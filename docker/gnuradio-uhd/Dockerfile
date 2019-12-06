FROM gnuradio:base

RUN apt-get update && apt-get install -y --no-install-recommends \
    libxml2         \
    libxml2-dev     \
    bison           \
    flex            \
    swig            \
    liborc-0.4-dev  \
    libaio-dev      \
    libgmp-dev      \
    libboost-regex-dev \
    libuhd3.14.1

ENV PYTHONPATH=/usr/local/lib/python3/dist-packages
