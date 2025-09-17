# EmbeddedNN-PowerOptimization
Faceid112 model 

Reducing Energy Consumption of Neural Network Processing on Embedded Accelerators

It implements a lightweight convolutional neural network (CNN) for face recognition / face ID, targeted for low-power deployment (MAX78000 chip).

The design is inspired by MobileNet-style bottleneck blocks → efficient, low-compute, low-memory.

The final output is a low-dimensional embedding (e.g., 64-D vector) that represents a face in a normalized feature space.


Simplenet model

A lightweight CNN based on SimpleNet, adapted for the MAX78000 microcontroller.

Designed to output L2-normalized embeddings.

Includes Quantization-Aware Training (QAT) and BatchNorm fusion → deployment-ready.
