FROM gnuradio:pluto

# RUN apt-get update && \
#     apt-get install -qy \
#         python3-pip         \
#         python3-setuptools  \
#         python3-wheel

# Upgrade pip 
RUN python3 -m pip install --upgrade pip

# Install required ONNX packages
RUN pip install 'onnx>=1.5' 'onnxmltools>=1.5' 'onnxruntime>=0.5.0' --force-reinstall

# RUN ldconfig