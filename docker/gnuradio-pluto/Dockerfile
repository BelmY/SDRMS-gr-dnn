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
    libboost-regex-dev

COPY build_pluto.sh /root/

RUN chmod +x /root/build_pluto.sh

RUN /root/build_pluto.sh

ENV PYTHONPATH=/usr/local/lib/python3/dist-packages