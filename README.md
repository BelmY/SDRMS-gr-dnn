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
* ONNX ( == 1.5.0)
* ONNX Runtime ( == 0.5.0)
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

This implementation uses the [message passing interface](https://wiki.gnuradio.org/index.php/Guided_Tutorial_Programming_Topics#5.3_Message_Passing) and it should be beared in mind when designing a flow graph with this block.

This implementaion has some limitations due to the buffer size and PMTs.

An example of its usage can be found in [*classifier_test.grc*](./examples/classifier_test.grc).

### Stream and Vector

This implementation uses [streams (synchronous block)](https://wiki.gnuradio.org/index.php/BlocksCodingGuide#Synchronous_Block) and vecors and it should be beared in mind when designing a flow graph with this block.

An example of its usage can be found in [*pluto_onnx_sync.grd*](./examples/pluto_onnx_sync.grc)

## Disclaimer

Currently, the ONNX framework (specification, runtime and tools) is in active development and there might be some limitations in the current and problems in the future versions.