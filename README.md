# gr-dnn

A GNU Radio out-of-tree module that make DNN inference using a specified ONNX model.

## Requirements

* GNU Radio ( >= 3.8 (commit hash 8ecdef41e47641f820fff73e0ad11c0890544b71))
* CMake ( >= 3.8)
* onnx ( >= 1.4.1)
* onnxruntime ( >= 0.2.1)
* git

## Installation from source

```
git clone https://gitlab.com/librespacefoundation/sdrmakerspace/sdr_dnn
cd sdr_dnn/gnuradio-oot-blocks/ONNX/gr-dnn
mkdir build
cd build
cmake ..
make
sudo make install
```
