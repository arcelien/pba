# Run this bash script from the main directory to download the CIFAR-100 dataset.

wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar -xzf cifar-100-python.tar.gz
mv cifar-100-python ./datasets/
rm cifar-100-python.tar.gz
