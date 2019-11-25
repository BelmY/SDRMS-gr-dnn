#!/bin/sh
set -e

export PREFIX=/opt

cd $PREFIX

git clone https://github.com/bastibl/gr-foo.git
cd gr-foo
mkdir build
cd build
cmake ..
make
sudo make install
sudo ldconfig
cd ..
rm -rf gr-foo

git clone git://github.com/bastibl/gr-ieee802-15-4.git
cd gr-ieee802-15-4
mkdir build
cd build
cmake ..
make
sudo make install
sudo ldconfig
cd ..
rm -rf gr-ieee802-15-4

exit 0
