
# Compiling

To build all the supported targets (with conventional IRIS workflow) run:

```
  make
```

This will yield binaries for OpenMP (`kernel.openmp.so`), CUDA (`kernel.ptx`) and HIP (`kernel.hip`) as well as the IRIS host runner (`vecadd-iris`), and a non-accelerator baseline (`vecadd`) binary.

If you wish to evaluate the (Charm)SYCL workflow run:

```
  make vecadd-sycl
```

This will also produce the sycl (`vecadd-sycl`) binary with all the corresponding backend device kernels bundled into it---currently Charm-SYCL only supports CUDA and OpenMP.

# Running

IRIS environment variables can be set to select the available devices:

```
  IRIS_ARCHS=opencl,hip,openmp,cuda vecadd-iris
```

The order the backends are provided to the `IRIS_ARCHS` variable indicates the priority.

To run the Charm-SYCL test:

```
  CHARM_SYCL_RTS=IRIS IRIS_ARCHS=openmp ./vecadd-sycl
```

Or:

```
  CHARM_SYCL_RTS=IRIS IRIS_ARCHS=cuda ./vecadd-sycl
```

# Performance Evaluation

Run `benchmark_performance.sh`
This generates the samples into the `results` directory.

# Figure Generation

Converts the samples from the `results` directory into box-and-whisker plots and saves them in the `figures` directory.

Activate the plotting environment with:

`mamba activate plotting`

To create the mamba/conda environment:

`mamba env create --force ./plotting.yaml`

All conversion is performed by running:

`./plot_results.sh`

# Setup

## Charm-SYCL

Charm-SYCL was install with: `build.sh` as was IRIS from their respective top level directories.

## DPC++

DPC++ was installed (on zenith) with:

```
module load nvhpc/22.11 gnu/11.3.0

export DPCPP_HOME=$HOME/dpc++-workspace
mkdir $DPCPP_HOME
cd $DPCPP_HOME
git clone https://github.com/intel/llvm -b sycl
CC=gcc CXX=g++ python $DPCPP_HOME/llvm/buildbot/configure.py --cuda
CC=gcc CXX=g++ python $DPCPP_HOME/llvm/buildbot/compile.py
```

## OpenSYCL

OpenSYCL was installed (on zenith) with:

```
  spack install llvm@13.0.1
  #installs llvm to the following location:
  export LLVM_DIR=/home/9bj/spack/opt/spack/linux-ubuntu22.04-zen2/gcc-12.1.0/llvm-13.0.1-2a5yipjwa6bco5vymttszyqauqzainuc
  export CC=$LLVM_DIR/bin/clang ; export CXX=$LLVM_DIR/bin/clang++ ; export LDFLAGS=-L$LLVM_DIR/lib ; export CPPFLAGS="-I$LLVM_DIR/include"
  git clone https://github.com/OpenSYCL/OpenSYCL ; cd OpenSYCL ; rm -rf ~/.opensycl
  cmake -DCMAKE_INSTALL_PREFIX=/home/9bj/.opensycl -DCMAKE_C_COMPILER=$LLVM_DIR/bin/clang -DCMAKE_CXX_COMPILER=$LLVM_DIR/bin/clang++ -DWITH_ROCM_BACKEND=OFF -DWITH_CUDA_BACKEND=OFF -DWITH_ACCELERATED_CPU=OFF -DBUILD_CLANG_PLUGIN=OFF -DCLANG_INCLUDE_PATH=$LLVM_DIR/include -DCLANG_EXECUTABLE_PATH=$LLVM_DIR/bin/clang++ -DCMAKE_CXX_FLAGS="-fuse-ld=lld"
  make -j`nproc`
  make install
```

**BROKEN**
```
    module load gcc/12.1.0

    module load llvm/13.0.1
    git clone https://github.com/OpenSYCL/OpenSYCL
    cd OpenSYCL
    cmake -DCMAKE_INSTALL_PREFIX=/home/9bj/.opensycl -DLLVM_ROOT=/auto/software/swtree/ubuntu20.04/x86_64/llvm/13.0.1/ -DLLVM_DIR=/auto/software/swtree/ubuntu20.04/x86_64/llvm/13.0.1/lib/cmake -DWITH_ACCELERATED_CPU=ON -DCUDA_TOOLKIT_ROOT_DIR=/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/cuda -DWITH_CUDA_BACKEND=ON
    make install
```


