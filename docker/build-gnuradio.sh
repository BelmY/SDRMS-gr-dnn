#/bin/bash
set -e

# Create base image for gnuradio 3.8
docker build -t gnuradio:base gnuradio-base
# Create pluto image to include pluto support to gnuradio 3.8
docker build -t gnuradio:pluto gnuradio-pluto
# Create uhd image to include UHD support to gnuradio 3.8
docker build -t gnuradio:uhd gnuradio-uhd
# Create dev image to include onnx support
docker build -t gnuradio:onnx  gnuradio-onnx
# Create cuda image to include support for CUDA 10.1
docker build -t gnuradio:cuda gnuradio-cuda
# Create zigbee-demo image with gr-ieee802-11 from the wime project
docker build -t gnuradio:zigbee-demo gnuradio-zigbee-demo
