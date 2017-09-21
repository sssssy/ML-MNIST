# easy MNIST

### A easy MNIST model to show how softmax regression works.

FILE EXPLANATIONS:

	input_data.py: Copyed from TENSORFLOW's GUIDE, to import train data set and test data set.

	MNIST_data: Automatically downloaded by input_Data.py, including 4 zipped floders, which contain all the basic data sets.

	read_data.py: Self-made program to unzip binary data set.Used to get random test data to show the model visually.
	
	test-images.idx3-ubyte: Unzipped from MNIST_data separately. Will be used in read_data.py.
	
	mnist.py(main): To run this model.

	WEIGHT output: Output a WEIGHT example when alpha was set as 0.01. Just to check if the training steps works.
