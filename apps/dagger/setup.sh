#!/bin/bash
source /auto/software/iris/setup_system.source
export SYSTEM=$(hostname|cut -d . -f 1|sed 's/[0-9]*//g')
export MACHINE=${SYSTEM^}

if [ ! -n "$IRIS_INSTALL_ROOT" ]; then
	IRIS_INSTALL_ROOT="$HOME/.local"
fi

export IRIS_SRC_DIR=../..
export WORKING_DIR=`pwd`

#installed with:
#micromamba create -f dagger.yaml
micromamba activate dagger
#if we don't have a conda env set, then load it.
if [[ -z "$CONDA_PREFIX" ]] ; then
  echo "Please ensure this script is run from a conda session (hint: conda activate iris)"
  echo "Aborting..."
  exit
fi

#start with a clean build of iris
rm -f $IRIS_INSTALL_ROOT/lib64/libiris.so ; rm -f $IRIS_INSTALL_ROOT/lib/libiris.so ;
cd $IRIS_SRC_DIR ; ./build.sh; [ $? -ne 0 ] && exit ; cd $WORKING_DIR ;

source $IRIS_INSTALL_ROOT/setup.source
make clean

export SYSTEM=$(hostname|cut -d . -f 1|sed 's/[0-9]*//g')
export MACHINE=${SYSTEM^}

