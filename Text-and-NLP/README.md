# Text processing and NLP applications

This part of projects relevant to sequence models, concretely focus on text processing and NLP applications. We'll start from **corpus preparation**, **tokenization**, **embedding**, to ...
covering the main workflows of NLP task.

## Corpus and Preprocess

Corpus consists of a full context with predicted words and other symbols, you need to clean and tokenize them to build a vocabulary which is the foundation of text task. What would be taken into account includes:

* case sensitive
* punctuations
* numbers
* special characters
* emoji
* ...

There're various of tools for pre-processing the words including `NLTK`, `emoji` libraries etc. By using these tools make it much easier and efficient for data preparation.

```python
import nltk
nltk.download('punkt')

corpus ='Which team is the "CHAMPION" of the World Cup 2026? ❤️ ESPANA!!!'
# replace punctuations
data = re.sub(r'[,!?;-]+',corpus)
# tokenize
data = nltk.word_tokenize(data)
# turn to lower case
data = [ch.lower() for ch in data]
```

## Tokenization

The first step of processing text is to split sentences into small parts of units and convert them into numerics which can be understand by machines, that's **tokenization**.

There're different granularities and methods of tokenization, let's take 
a glance:

* Words
* **Subwords**
  * WordPiece
  * BPE
  * SentencePiece
* Characters

Subwords is a common way and we'll use it to tokenize sentences into small pieces.

### tokenization.py

In most cases we'll not build tokenizer from scratch, a pre-trained one would be a better choice. There're lots of popular models from `HuggineFace transformers`, we use `BERT` and `GPT` respectively to do practice.

**Remark:** It's a more common way to use `AutoTokenizer` to automatically matches the strategy and model.

```python
def get_tokenizer(tk_name):

    # `AutoTokenizer` is a better way in product, 
    # but for practice we specified tokenizer's type respectively
    if tk_name and tk_name.strip().lower() == "gpt":
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        # should set `eos_token` as padding for GPT2
        tokenizer.pad_token = tokenizer.eos_token
    elif tk_name and tk_name.strip().lower() == "bert":
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    else:
        tokenizer = AutoTokenizer.from_pretrained(tk_name)

    return tokenizer
```

> BERT Tokens: <br/>
[['[CLS]', 'i', "'", 'm', 'feeling', 'happy', 'today', 'because', 'doing', 'deep', '##lea', '##rn', '##ing', '[SEP]'],<br/>
 ['[CLS]', 'don', "'", 't', 'drop', 'garbage', 'anywhere', 'in', 'din', '##ning', 'room', '~', '[SEP]', '[PAD]']]<br/>
BERT Token IDs:<br/> tensor([[  101,  1045,  1005,  1049,  3110,  3407,  2651,  2138,  2725,  2784,
         19738,  6826,  2075,   102],
        [  101,  2123,  1005,  1056,  4530, 13044,  5973,  1999, 11586,  5582,
          2282,  1066,   102,     0]])

> GPT Tokens: <br/>
[['I', "'m", 'Ġfeeling', 'Ġhappy', 'Ġtoday', 'Ġbecause', 'Ġdoing', 'Ġdeep', 'learning', '<|endoftext|>'], <br/>
['Don', "'t", 'Ġdrop', 'Ġgarbage', 'Ġanywhere', 'Ġin', 'Ġdin', 'ning', 'Ġroom', '~']]<br/>
GPT Token IDs: <br/>tensor([[   40,  1101,  4203,  3772,  1909,   780,  1804,  2769, 40684, 50256],[ 3987,   470,  4268, 15413,  6609,   287, 16278,   768,  2119,    93]])


## Word Representations and Embeddings

Embedding models evolute from classic static to modern contextual ones. Here we'll create with a classic way of `CBOW` which is one strategy of `Word2Vec`, by the way, the other is `skip-gram`.

Same as before, in most of the cases you won't create embeddings from scratch. Usually you'll use them by mature models as library. 

### embedding_model.py


