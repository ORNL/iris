#!/bin/bash
source /auto/software/iris/setup_system.source

if [ ! -n "$IRIS_INSTALL_ROOT" ]; then
  IRIS_INSTALL_ROOT="$HOME/.iris"
fi

export IRIS_SRC_DIR=../..
export WORKING_DIR=`pwd`
export SYSTEM=`hostname`

source $IRIS_INSTALL_ROOT/setup.source
#start with a clean build of iris
rm -f $IRIS_INSTALL_ROOT/lib64/libiris.so ; rm -f $IRIS_INSTALL_ROOT/lib/libiris.so ;
cd $IRIS_SRC_DIR ; ./build.sh; [ $? -ne 0 ] && exit ; cd $WORKING_DIR ;

