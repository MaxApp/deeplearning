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

    Technicals covering:
    
    * data tranform & normalize
    * data augmentation
    * filter specific subclasses from datasets
    * convolutional layer
    * max pooling
    * flatten to fc layer
    * dropout regularization
    * mini-batch gradient clearing
    * evaluate model with validation set

    **cautions:**
    
    * Be careful with accumulating the loss in batch and calculate the average loss for total. The loss value returned by loss function is the mean value per batch by default, in order to calculate the total loss based on every sample, you should calculate every batch total loss and accumulate them to get the final one. Otherwise you could calculate average loss based on batch, see `additional` notes below.
    * It's not promised the last batch is a full size, so you should count step by step to get the right size of the sample in a batch.
    * Don't use softmax as the last layer, use Linear layer directly instead. 
    
      `CrossEntropyLoss = LogSoftmax + NLLLoss`
    
    **additional:**
    There're two different ways of calculating the average loss.
    
    * Batch average
    
      `running_loss += loss.item()`

      `running_loss / len(train_loader)`

    * Sample average

      `running_loss += loss.item() * batch_size`
      
      `running_loss / len(train_dataset)`      

    In this model, we are using the **sample average loss**
    ```python
    # Initialize running validation loss and correct predictions count
    running_val_loss = 0.0
    correct = 0
    total = 0
    # Disable gradient calculations for validation
    with torch.no_grad():
        # Iterate over batches of data in the validation loader
        for images, labels in val_loader:
            # Move images and labels to the specified device
            images, labels = images.to(device), labels.to(device)
            
            # Perform a forward pass to get model outputs
            outputs = model(images)
            
            # Calculate the validation loss for the batch
            val_loss = loss_function(outputs, labels)
            # Accumulate the validation loss
            running_val_loss += val_loss.item() * images.size(0)
            
            # Get the predicted class labels
            _, predicted = torch.max(outputs, 1)
            # Update the total number of samples
            total += labels.size(0)
            # Update the number of correct predictions
            correct += (predicted == labels).sum().item()
            
    # Calculate the average validation loss for the epoch
    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    # Append the epoch validation loss to the list
    val_losses.append(epoch_val_loss)
    
    # Calculate the validation accuracy for the epoch
    epoch_accuracy = 100.0 * correct / total
    # Append the epoch accuracy to the list
    val_accuracies.append(epoch_accuracy)
    ```

* utils.py

    helper functions assist with main code.


 