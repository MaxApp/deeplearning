# Description
Training CNN Models with CIFAR-100 datasets.

### scripts descriptions
* dataset_visualise.py

    Given the pre-downloaded datasets, load file and parse the images and labels.

    Randomly choose some of the images and show them with matplotlib, in order to get an intuition with what will be processing.

    ![random_images](imgs/dataset_visualise_01.png)

* nature_classifier.py

    Build a model that can classify flowers, insects and small animals within small curated classes.

    We'll use technicals:
    
    * data tranform
    * data augmentation
    * filter specific subclasses from datasets
    * convolutional layer
    * fc layer
    * dropout

* utils.py

    helper functions assist with main code.


 