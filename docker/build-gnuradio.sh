#/bin/bash
set -e

# Create base image for gnuradio 3.8
docker build -t gnuradio:base gnuradio-base
# Create pluto image to include pluto support to gnuradio 3.8
docker build -t gnuradio:pluto gnuradio-pluto
# Create dev image to include onnx support 
docker build -t gnuradio:onnx  gnuradio-onnx
# Create cuda image to include support for CUDA 10.1
docker build -t gnuradio:cuda gnuradio-cuda
