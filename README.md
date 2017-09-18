# Capstone-DL
---
## What is Convolutional Neural Network?
Convolutional Neural Networks are very similar to ordinary Neural Networks: they are made up of neurons that have learnable weights and biases. Each neuron receives some inputs, performs a dot product and optionally follows it with a non-linearity. The whole network expresses a single differentiable score function: from the raw image pixels on one end to class scores at the other. The difference is that ConvNet architectures make the explicit assumption that the inputs are images, which allows us to encode certain properties into the architecture. These then make the forward function more efficient to implement and vastly reduce the amount of parameters in the network.

Convolutional Neural Networks take advantage of the fact that the input consists of images and they constrain the architecture in a more sensible way. In particular, unlike a regular Neural Network, the layers of a ConvNet have neurons arranged in 3 dimensions: width, height, depth. For example, the input images in CIFAR-10 are an input volume of activations, and the volume has dimensions 32x32x3 (width, height, depth respectively). The final output layer for CIFAR-10 would have dimensions 1x1x10, because by the end of the ConvNet architecture we will reduce the full image into a single vector of class scores, arranged along the depth dimension.

## Layers used to build Convolutional Neural Network?
There are three main types of layers to build ConvNet architectures: Convolutional Layer, Pooling Layer, and Fully-Connected Layer. We will stack these layers to form a full ConvNet architecture.

e.g. Architecture of a simple ConvNet for CIFAR-10 classification.
- Input Layer [32 x 32 x 3] will hold the raw pixel values of the image, in this case an image of width 32, height 32, and three colour channels Red, Green and Blue.
- Conv Layer will compute the output of neurons that are connected to local regions in the input, each computing a dot product between their weights and a small region they are connected to. If we are using 12 filters, the resulting volume will be [32 x 32 x 12].
- RELU Layer will apply an elementwise activation function, such as the max(0, x) thresholding at zero. This leaves the size of the volume unchanged.
- POOL Layer will perform a downsampling operation along the spatial dimensions (width, height), resulting in volume such as [16 x 16 x 12].
- FC Layer will comp-ute the class scores, resulting in the volume of size [1 x 1 x 10], where each of the 10 numbers correspond to a class score, such as among the 10 categories of CIFAR-10.

In this way, ConvNet is able to transform the original image layer by layer from the original pixel values to the final class scores. (Note that the Conv / FC layers perform transformations that area function of not only the activations in the input volume, but also of the parameters, weights and biases of the neurons. On the other hand, the RELU / POOL layers will implement a fixed function. The parameters of Conv / FC layers will be trained with gradient descent so that the class scores that the ConvNet computes are consistent with the labels in the training set for each image.)


## Contributing changes
Please fork this repository and contribute back using pull requests.
Any contributions, large or small are welcomed and appreciated but will be thoroughly reviewed and discussed.
* See [CONTRIBUTING.md](CONTRIBUTING.md)

## Licensing
* See [LICENSE](LICENSE)
