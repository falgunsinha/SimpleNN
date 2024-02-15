# 1. SimpleNN

Create a simple feed forward neural network using Tensorflow.  Input layer -> hidden layer -> output layer for IRIS dataset.
Normalize the inputs using Z-scalar normalization. Also show the Confusion Matrix.

# 2. CNN
Use the dataset data.csv. The dataset contains a grayscale images (8x8 pixels) in the first 64 columns. Reshape it with the numpy.reshape function to the shape of Nx8x8x1. N is length of the dataset. Last column (index 64) is a category in the range of $<0,3>$. Encode this column by the one-hot encoding.

Create convolution network hawing 2D convolution layer with relu activation, five filters, and kernel size 5x5. This layer will be followed by Maxpooling 2D (default parameters) layer, Flatten layer, and Dense layer with softmax activation. The first layer of the model will be Input(shape=(8,8,1)). Number of outputs of the Dense layer will be equal to 4.

Learn the network with optimizer Adam, loss function categorical_crossentropy, and metrics is accuracy for 20 epochs and batch size 30.
