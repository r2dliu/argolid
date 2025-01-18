#!/bin/bash
# Usage: $bash install_prereq_linux.sh $INSTALL_DIR
# Default $INSTALL_DIR = ./local_install
#
if [ -z "$1" ]
then
      echo "No path to the Argolid source location provided"
      echo "Creating local_install directory"
      LOCAL_INSTALL_DIR="local_install"
else
     LOCAL_INSTALL_DIR=$1
fi

mkdir -p $LOCAL_INSTALL_DIR
mkdir -p $LOCAL_INSTALL_DIR/include

curl -L https://github.com/PolusAI/filepattern/archive/refs/tags/v2.0.4.zip -o v2.0.4.zip 
unzip v2.0.4.zip
cd filepattern-2.0.4
mkdir build
cd build
cmake -Dfilepattern_SHARED_LIB=ON -DCMAKE_PREFIX_PATH=../../$LOCAL_INSTALL_DIR -DCMAKE_INSTALL_PREFIX=../../$LOCAL_INSTALL_DIR ../src/filepattern/cpp
make install -j4
cd ../../

curl -L https://github.com/pybind/pybind11/archive/refs/tags/v2.11.1.zip -o v2.11.1.zip
unzip v2.11.1.zip
cd pybind11-2.11.1
mkdir build_man
cd build_man
cmake -DCMAKE_INSTALL_PREFIX=../../$LOCAL_INSTALL_DIR/  -DPYBIND11_TEST=OFF ..
make install -j4
cd ../../

if [[ "$OSTYPE" == "darwin"* ]]; then
      curl -L https://github.com/libjpeg-turbo/libjpeg-turbo/archive/refs/tags/3.1.0.zip -o 3.1.0.zip
      unzip 3.1.0.zip
      cd libjpeg-turbo-3.1.0
      mkdir build_man
      cd build_man
      cmake -DCMAKE_INSTALL_PREFIX=/usr/local -DENABLE_STATIC=FALSE -DCMAKE_BUILD_TYPE=Release ..
      sudo make install -j4
      cd ../../
fi
