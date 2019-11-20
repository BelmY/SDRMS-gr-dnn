#!/bin/sh
set -e

export PREFIX=/opt

cd $PREFIX

git clone https://github.com/analogdevicesinc/libiio.git
cd libiio
cmake . && make && make install
cd ..

git clone https://github.com/analogdevicesinc/libad9361-iio.git
cd libad9361-iio
cmake . && make && make install
cd ..

git clone https://github.com/analogdevicesinc/gr-iio.git
cd gr-iio
git checkout upgrade-3.8
cmake . && make -j 8 && make install
cd ..

ln -s  /usr/local/lib/python3/dist-packages/iio /usr/local/lib/python3.7/dist-packages/iio

exit 0