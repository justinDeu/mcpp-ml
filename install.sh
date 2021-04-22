#! /bin/bash

# clone the autodiff repo
git clone https://github.com/autodiff/autodiff.git

# wget the eigen tar, extract, remove the tar file
wget https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.tar.gz
tar -xvf eigen-3.3.9.tar.gz
rm eigen-3.3.9.tar.gz
