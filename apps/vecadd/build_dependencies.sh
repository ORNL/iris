#!/bin/bash

#Charm-SYCL
git clone -b irisv3 git@code.ornl.gov:fujita/charm-sycl
[ $? -ne 0 ] && exit
cp build_charm_sycl.sh charm-sycl
cd charm-sycl
exec ./build_charm_sycl.sh
[ $? -ne 0 ] && exit

echo next one!
