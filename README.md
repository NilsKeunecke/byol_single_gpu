# BYOL, but trainable on a single GPU
This is a Tensorflow implementation of the BYOL paper. It uses weight standardization and group norm to make it trainable on a single consumer GPU. While the performance is not comparable to that of the same architecture pretrained on ImageNet, it can be trained on CIFAR10 on a normal GPU (tested on RTX 2070S). Performances are reported below:

## Performance on CIFAR10
After the self-supervised pretraining, the authors fine-tune the network on x% of the training data. The table below shows the classification accuracy. 50k images have been used for training/validation and 10k for testing.
1% Fine-tuned | 47.8% |
--- | --- | 
10% Fine-tuned | 64.8% |
100% Fine-tuned | 76.3%

## Papers:
- BYOL: https://arxiv.org/abs/2006.07733
- BYOL without batch statistics: https://arxiv.org/abs/2010.10241
- GroupNorm: https://arxiv.org/abs/1803.08494
- Weight standardization: https://arxiv.org/abs/1903.10520
