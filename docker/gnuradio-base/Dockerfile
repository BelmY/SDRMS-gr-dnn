FROM debian:testing

ENV DEBIAN_FRONTEND noninteractive

RUN apt update  -q
RUN apt upgrade -y

# Install basics
RUN apt install -yq \
    build-essential \
    cmake           \
    doxygen         \
    git             \
    vim

#install gnuradio
RUN apt install -yq \
    gnuradio

# Install GUI dependencies
RUN apt install -yq \
    gir1.2-gtk-3.0  \
    python3-gi-cairo
    
RUN apt install -yq \
    python3-pip

ENV PYTHONPATH /usr/local/lib/python3/