#!/bin/bash
WORKING_DIRECTORY=`pwd`

#IRIS
cd ../.. ; ./build.sh;
cd $WORKING_DIRECTORY

#Charm-SYCL
git clone -b irisv3 git@code.ornl.gov:fujita/charm-sycl
[ $? -ne 0 ] && exit
cp build_charm_sycl.sh charm-sycl
cd charm-sycl
./build_charm_sycl.sh
[ $? -ne 0 ] && exit

cd $WORKING_DIRECTORY
echo "todo dpc++ and adaptivecpp!"
