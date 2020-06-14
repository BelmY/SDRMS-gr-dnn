# gr-dnn

A GNU Radio out-of-tree (OOT) module that allows us to run machine learning inference using ONNX models.

## ONNX

[ONNX](https://onnx.ai/) is an interoperabe format for machine learning models supported by various machine learning (ML) and deep neural networkds (DNN) frameworks and tools.

[ONNX Runtime](https://microsoft.github.io/onnxruntime/) is a engine for Open Neural Network Exchange (ONNX) models. It also supports graph optimizations and accelerators.

[ONNX ML Tools](https://github.com/onnx/onnxmltools) enables you to convert models from different machine learning toolkits into ONNX.

## Requirements

* GNU Radio ( >= 3.8)
* CMake ( >= 3.8)
* Python (>= 3.6)
* ONNX ( == 1.6.0)
* ONNX Runtime ( == 1.0.0)
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

## Modules

There are two different block implementations both under the category **DNN**

### Message passing and Polimorphic types (PMTs)

This implementation uses the [message passing interface](https://wiki.gnuradio.org/index.php/Guided_Tutorial_Programming_Topics#5.3_Message_Passing).

This implementaion has some limitations due to the buffer size and PMTs. Due to these limitations **the block is not longer under development**.

You can find two examples on how to use the PMT block: 
- Using the MNIST dataset in [*mnist_demo_pmt.grc*](./examples/mnist_demo_pmt.grc) you can read more about this example [here](https://gitlab.com/librespacefoundation/sdrmakerspace/gr-dnn/-/wikis/examples/MNIST-dataset).
- Using a SDR device in [*pluto_onnx_pmt.grc*](./examples/mnist_demo_pmt.grc).

### Stream and Vector

This implementation uses [streams (synchronous block)](https://wiki.gnuradio.org/index.php/BlocksCodingGuide#Synchronous_Block) and vecors and it should be beared in mind when designing a flow graph with this block.

You can find two examples on hot wo use the stream block:
- Using the MNIST dataset in [*mnist_demo_vector.grc*](./examples/mnist_demo_vector.grc) 
- Using a SDR device in [*pluto_onnx_vector.grc*](./examples/pluto_onnx_vector.grc) you can read more about this example [here](https://gitlab.com/librespacefoundation/sdrmakerspace/gr-dnn/-/wikis/examples/SDR-with-ADALM-PLUTO).

## Disclaimer

Currently, the ONNX framework (specification, runtime and tools) is in active development and there might be some limitations in the current and problems in the future versions.