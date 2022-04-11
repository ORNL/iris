VCMAKE="cmake3"
if ! command -v cmake3 &> /dev/null
then
    VCMAKE=cmake
fi
set -x;
${VCMAKE} ../ -DCMAKE_CXX_FLAGS="-g -fPIC" -DCMAKE_C_FLAGS="-g -fPIC"  -DCMAKE_INSTALL_PREFIX=$PWD/../install_host $@
