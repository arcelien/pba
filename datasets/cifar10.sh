# Run this bash script from the main directory to download the CIFAR-10 dataset.

wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzf cifar-10-python.tar.gz
mv cifar-10-batches-py ./datasets/
rm cifar-10-python.tar.gz
