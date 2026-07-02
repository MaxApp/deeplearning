# Description
Train a CNN model with CIFAR-100 datasets.
Classify limited subclasses instead of 100 categories.

### scripts descriptions
* **dataset_visualise.py**

    Given the pre-downloaded datasets, load and parse the files to read images and labels.

    Randomly choose some of the images and show them with matplotlib, in order to get an intuition with what form of data that will be processing.

    ![random_images](imgs/dataset_visualise_01.png)

* **nature_classifier.py**

    Build a model that can classify flowers, insects and small animals within small curated classes.

    **Technicals covering:**
    
    * data tranform & normalize
    * data augmentation
    * filter specific subclasses from datasets
    * convolutional layer
    * max pooling
    * flatten to fc layer
    * dropout regularization
    * mini-batch gradient clearing
    * evaluate model with validation set

    **Cautions:**
    
    * Be careful with accumulating the loss in batch and calculate the average loss for total. The loss value returned by loss function is the mean value per batch by default, in order to calculate the total loss based on every sample, you should calculate every batch total loss and accumulate them to get the final one. Otherwise you could calculate average loss based on batch, see `additional` notes below.
    * It's not promised the last batch is a full size, so you should count step by step to get the right size of the sample in a batch.
    * Don't use softmax as the last layer, use Linear layer directly instead. 
    
      `CrossEntropyLoss = LogSoftmax + NLLLoss`
    
    **Additional:**

    There're two different ways of calculating the average loss.
    
    * Batch average
    
      `running_loss += loss.item()`

      `running_loss / len(train_loader)`

    * Sample average

      `running_loss += loss.item() * batch_size`
      
      `running_loss / len(train_dataset)`      

    In this model, we are using the **sample average loss**
    ```python
    val_losses = 0.0
    correct = 0
    # Disable gradient descent
    with torch.no_grad():
        for val_images, labels in val_loader:
            outputs = model(val_images)
            batch_avg_loss = loss_func(outputs, labels)
            # accumulate total loss per batch
            val_losses += batch_avg_loss.item() * labels.size(0)
            predicted = torch.argmax(outputs, 1)
            correct += (predicted == labels).sum().item()

        # Calculate the average validation loss
        epoch_val_loss = val_losses / len(val_loader.dataset)
        # Calculate the accuracy
        epoch_accuracy = 100.0 * correct / len(val_loader.dataset)
    ```

    **Model training figures:** 
    
    Calculate each loss per epoch, as well as accuracy on validation dataset.

    ![training_result](imgs/nature_classifier_01.png)


* utils.py

    helper functions assist with main code.


 