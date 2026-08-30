# Practicals and Experiments of Machine Learning and Deep Learning

## Welcome and Foreword
Hi guys, welcome to this page! 😀

The resources and code labs here are my learning maps about machine learning and deep learning. They're organized in various folders by different regions and specilizations.

ML and AI is such a big topic that covers almost every aspect of our daily lives recently. The emerging techniques evolved rapidly than any other regions ever before. You can frequently see a new verison, a new model or a new application is released just in few months to even in few weeks by various companies, that's promising and thrilling.

As a classic programmer for many years，the ML and DL has made a deep impression on me and driven me to understand and master the technicals from start to deep. Despite a bit of sophisticated in theories at the beginning, I found more and more interests during progress and learned with more pleasure and ease.

As mentioned before, ML is a wide range in topics but typically seperated in **Visions**, **Texts** and **Audios** in form of inputs and processing format. Other than **Audios** I haven't refer yet, I marked some of the important learning point during practical labs. It's useful and helpful for realistic works.

## Overview of ML workflow

From the start point to the real world product being used, a model with production has a workflow to pregress regularly, basically it includes:

* Data Collection
* Data Cleaning and Preparation
* Model Archetecture
* Model Training
* Evaluation
* Compression and Deployment

In general different types of data use different methods to precess, train and evaluate a model, but the workflow usually keeps the same.

## Knowledge Maps

### Mathematics

Theoretically you need a basic knowledge of *probability*, *calculous* and *linear algebra* despite of that much. They will help you better understand what machine is doing during training and what the target is. To some extent, a well mastered math foundation will facilitate you go further and deeper on the journey.

Parts of the conceptions that would be used during ML includes:
* derivative
* vector
* matrix
* linear algebra
* probability distribution
* ...

In some applications to be more concretely:
* Naive Bayes
* Laplacian Smoothing
* Log Likelihood
* Cosine Similarity
* ...

### Programming Languages

`Python` is a popular programming language with a mess of tools and easy to learn. Although I also mastered *Java*, *C*, *Ruby*, *PHP*, *Javascripts* and others, I found Python is really a convenient one.

### Frameworks and Tools

`Pytorch` is being used during my learning, `TensorFlow` is another widely used one although. Both are most common frameworks for machine learning.

Other than learning framework itself, `Numpy`, `Pandas`, `scikit-learn` and `Matplotlib` is essential that you should master. They will help you processing data, tracking training and making comparisons. Visualization to your data is intuitive and important.

### Model Architectures and Components

Familiar with common methods and classic architectures like:

* CNN
    * LeNet-5
    * AlexNet
    * VGG-16
    * ResNet
    * Inception Net
    * MobileNet
    * EfficientNet
* RNN
    * GRU
    * LSTM
    * B-RNN
* NLP
    * Tokenization
    * Word Embedding（Vector Space Models）
        * KNN
        * ANN (LSH)
    * Sequence to Sequence Models
    * Probabilistic Models
        * Naive Bayes Classification Models（Laplacian Smoothing + Log Likelihood）
        * Markov Models
        * 
    * Transformer Models
* Transformer
    * Attention Model
    * Encoder
    * Decoder

### Training, Evaluating and Tuning

You could build your models in different architectures, even it's not based on neural network. Generally if you model based on neural network, you most likely to train it with `Gradient Descent`. In that way, you should master `Loss Function`, `Optimizer`, `Scheduler` etc during training. Also you should design how to evaluate your model's performance with metrics such as `Loss`, `Accuracy`, `Precision` and `Recall`. You should look out memory usage and efficiency too.

### Optimization and Deployment

After a model is well trained, you still need to do some extra work in order to refine it to deployment.

Transform your model to inference engine is a common way. You need to convert it to `ONNX` format or other for crossing platforms and gain a better performance.

Before that, you could also do optimizing with `pruning` and `quantization`. That can makes significant improving to your model while in deployment.

Beside, with a management platform such like `MLFlow` could help you training and monitoring your model.

## Conclusion

The page is just an overview of the whole labs, it is still being updated during learning. To get more details of each part, you could dive into folders for variety of regions.

All the annotations are personal understanding, not for teaching aims. If there are any errors, please don't hesitate to touch me.<br/>
Thanks for reading!


