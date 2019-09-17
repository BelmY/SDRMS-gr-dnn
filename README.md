# gr-dnn

A GNU Radio out-of-tree module that make DNN inference using a specified ONNX model.

## Requirements

* GNU Radio ( >= 3.8)
* CMake ( >= 3.8)
* onnx ( >= 1.5.0)
* onnxruntime ( >= 0.5.0)
* git

## Installation from source

```
git clone https://gitlab.com/librespacefoundation/sdrmakerspace/gr-dnn
cd gr-dnn
mkdir build
cd build
cmake ..
make
sudo make install
```
