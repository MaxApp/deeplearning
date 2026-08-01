# Siamese Network Application

Here comes in a different architecture of the network which ingest two or more inputs, learn a similarity function to measure distance between items in the same feature space.
The core technique is multi inputs going through an identical network and share the same parameters. The output embedding space eventually pull the similar items closer and push dissimilar items apart.

## Signature Verification

This is a common scenario with security implications to identify whether two signatures come from the same person, in other words to say it is genuine or forged.

~~We'll use datasets of [handwritten signatures](https://www.kaggle.com/datasets/divyanshrai/handwritten-signatures) from Kaggle.com.~~

It's a little bit hard to find satisfied dataset which is well organized and in right shape and format. We finally use a subset of `Signature Verification_v5.v11` and re-organized it somewhat by manually in order to get a right pair of signatures in genuine and forged.

The dataset is already augmented in rotation, brightness adjustment and blur.

![sig_samples](imgs/sig_sample.png)

We will train the model by using triplets: an **anchor**, a **positive**, and a **negative**.
The corresponding loss function will be `nn.TripletMarginLoss`. 





