
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

# Setup

OpenSYCL was installed (on zenith) with:

```
    module load llvm/13.0.1
    git clone https://github.com/OpenSYCL/OpenSYCL
    cd OpenSYCL
    cmake -DCMAKE_INSTALL_PREFIX=/home/9bj/.opensycl -DLLVM_ROOT=/auto/software/swtree/ubuntu20.04/x86_64/llvm/13.0.1/ -DLLVM_DIR=/auto/software/swtree/ubuntu20.04/x86_64/llvm/13.0.1/lib/cmake -DWITH_ACCELERATED_CPU=ON -DCUDA_TOOLKIT_ROOT_DIR=/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/cuda -DWITH_CUDA_BACKEND=ON
    make install
```

