VCMAKE="cmake3"
if ! command -v cmake3 &> /dev/null
then
    VCMAKE=cmake
fi
set -x;
echo "Extra args: $@"
${VCMAKE} -DCMAKE_C_FLAGS="-fPIC -g" -DCMAKE_CXX_FLAGS="-fPIC -g" -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake -DSNAPDRAGON=ON -DANDROID_PLATFORM=28 -DANDROID_ABI=arm64-v8a -DANDROID_ARM_NEON=ON -DUSE_NDK=ON -DCMAKE_INSTALL_PREFIX=$PWD/../install -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_SHARED_LIBS=ON .. $@
