# EmbeddedNN-PowerOptimization 
---------------------------------------------------------------------------------------------------------------------------------------------------
### Residual model  ###

AI85ResidualSimpleNet for CIFAR-10 Classification

AI85ResidualSimpleNet is a convolutional neural network (CNN) designed for CIFAR-10 image classification, optimized for deployment on AI hardware accelerators like the MAX78000. It leverages custom hardware-accelerated layers from the ai8x library to improve inference efficiency.

Key Features:

Residual Connections: Enhances gradient flow, allowing for deeper networks and better learning.

Fused Operations: Combines convolution, activation, and pooling operations into optimized layers for hardware acceleration.

Quantization: Supports quantization-aware training (QAT) for reducing model size and improving inference speed on low-resource hardware.

Batch Normalization Fusion: Fuses batch normalization parameters into convolutions to optimize inference performance.

Usage:

This model is designed for CIFAR-10 classification, but can be adapted for other tasks with minor modifications. It is suitable for edge devices and custom AI accelerators, providing a lightweight, fast solution for real-time image classification.

How to Use:

Instantiate the model using the ai85ressimplenet function.

Optionally apply quantization and batch normalization fusion for hardware optimization.

Train on the CIFAR-10 dataset or modify for other datasets.

--------------------------------------------------------------------------------------------------------------------------------------

### Faceid112 model ###

Reducing Energy Consumption of Neural Network Processing on Embedded Accelerators

It implements a lightweight convolutional neural network (CNN) for face recognition / face ID, targeted for low-power deployment (MAX78000 chip).

The design is inspired by MobileNet-style bottleneck blocks → efficient, low-compute, low-memory.

The final output is a low-dimensional embedding (e.g., 64-D vector) that represents a face in a normalized feature space.

-------------------------------------------------------------------------------------------------------------------------------------

###   Simplenet model  ###

A lightweight CNN based on SimpleNet, adapted for the MAX78000 microcontroller.

Designed to output L2-normalized embeddings.

Includes Quantization-Aware Training (QAT) and BatchNorm fusion → deployment-ready.


-----------------------------------------------------------------------------------------------------------------------------------
### Why Quantization aware training ??  ###

This method is well-suited for power optimization on embedded boards, especially those with specialized AI accelerators like the MAX78000. 

Why it's suitable for power optimization?

Quantization-Aware Training (QAT): 

By reducing the precision of weights and activations (e.g., to 8-bit or lower), quantization drastically lowers memory usage and computational requirements, which directly reduces power consumption.

Fused Layers:

Combining operations like convolution, activation (ReLU), and pooling into a single fused operation reduces intermediate memory access and computation steps, saving both energy and time.

Residual Connections:

Efficient network architecture helps achieve good accuracy with fewer parameters or layers, reducing the overall compute load.

Batch Normalization Fusion:

Folding batch normalization parameters into convolution layers reduces runtime operations, minimizing unnecessary computations during inference.

Custom AI Hardware Support:

The use of ai8x library layers optimized for specific embedded AI hardware means that the network takes advantage of hardware accelerators designed for low power consumption.



