from sklearn.metrics import f1_score
from datasets import load_dataset, disable_caching
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments


def compute_f1_metrics(pred):
    start_labels = pred.label_ids[0]
    start_preds = pred.predictions[0].argmax(-1)
    end_labels = pred.label_ids[1]
    end_preds = pred.predictions[1].argmax(-1)

    f1_start = f1_score(start_labels, start_preds, average='macro')
    f1_end = f1_score(end_labels, end_preds, average='macro')

    return {
        'f1_start': f1_start,
        'f1_end': f1_end,
    }

def process_samples(sample):
    tokenized_data = tokenizer(sample['document_plaintext'], sample['question_text'], truncation="only_first", padding="max_length")

    input_ids = tokenized_data["input_ids"]

    # We will label impossible answers with the index of the CLS token.
    cls_index = input_ids.index(tokenizer.cls_token_id)

    # If no answers are given, set the cls_index as answer.
    if sample["annotations.minimal_answers_start_byte"][0] == -1:
        start_position = cls_index
        end_position = cls_index
    else:
        # Start/end character index of the answer in the text.
        gold_text = sample["document_plaintext"][sample['annotations.minimal_answers_start_byte'][0]:sample['annotations.minimal_answers_end_byte'][0]]
        start_char = sample["annotations.minimal_answers_start_byte"][0]
        end_char = sample['annotations.minimal_answers_end_byte'][0] #start_char + len(gold_text)

        start_token = tokenized_data.char_to_token(start_char)
        end_token = tokenized_data.char_to_token(end_char - 1)

        # if start position is None, the answer passage has been truncated
        if start_token is None:
            start_token = tokenizer.model_max_length
        if end_token is None:
            end_token = tokenizer.model_max_length

        start_position = start_token
        end_position = end_token

    return {'input_ids': tokenized_data['input_ids'],
          'attention_mask': tokenized_data['attention_mask'],
          'start_positions': start_position,
          'end_positions': end_position}

if __name__ == "__main__":

    train_data = load_dataset('tydiqa', 'primary_task')
    tydiqa_data = train_data.filter(lambda example: example['language'] == 'english')

    # Flattening the datasets
    flattened_train_data = tydiqa_data['train'].flatten()
    flattened_test_data =  tydiqa_data['validation'].flatten()
    # Selecting a subset of the train dataset
    flattened_train_data = flattened_train_data.select(range(3000))
    # Selecting a subset of the test dataset
    flattened_test_data = flattened_test_data.select(range(1000))

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
    model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad",     
                                                        device_map="auto",        # puts the model on GPU(s) if available
                                                        low_cpu_mem_usage=True,   # avoids full CPU copy
                                                        torch_dtype="auto"       # picks an appropriate dtype for the device)
                                                        )

    disable_caching()  # avoid HF datasets cache attempts

    processed_train_data = flattened_train_data.map(
        process_samples,
        keep_in_memory=True,
        load_from_cache_file=False,
    )

    processed_test_data = flattened_test_data.map(
        process_samples,
        keep_in_memory=True,
        load_from_cache_file=False,
    )

    model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad",     
                                                        device_map="auto",        # puts the model on GPU(s) if available
                                                        low_cpu_mem_usage=True,   # avoids full CPU copy
                                                        torch_dtype="auto"       # picks an appropriate dtype for the device)
                                                        )

    columns_to_return = ['input_ids','attention_mask', 'start_positions', 'end_positions']
    processed_train_data.set_format(type='pt', columns=columns_to_return)
    processed_test_data.set_format(type='pt', columns=columns_to_return)

    training_args = TrainingArguments(
        output_dir='model_results5',          # output directory
        overwrite_output_dir=True,
        num_train_epochs=3,              # total number of training epochs
        per_device_train_batch_size=8,  # batch size per device during training
        per_device_eval_batch_size=8,   # batch size for evaluation
        warmup_steps=20,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir=None,            # directory for storing logs
        logging_steps=50
    )

    trainer = Trainer(
        model=model, # the instantiated 🤗 Transformers model to be trained
        args=training_args, # training arguments, defined above
        train_dataset=processed_train_data, # training dataset
        eval_dataset=processed_test_data, # evaluation dataset
        compute_metrics=compute_f1_metrics
    )

    trainer.train()

    # The evaluation may take around 30 seconds
    trainer.evaluate(processed_test_data)