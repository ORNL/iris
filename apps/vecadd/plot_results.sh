#!/bin/bash
rm -rf figures ; mkdir figures
python plot_results.py iris_openmp "IRIS OpenMP"
python plot_results.py iris_cuda "IRIS CUDA"
python plot_results.py iris_hip "IRIS HIP"
python plot_results.py dpc++_cuda "DPC++ CUDA"
python plot_results.py charmsycl_hip "Charm-SYCL IRIS HIP"
python plot_results.py charmsycl_hip_directly "Charm-SYCL (Internal) HIP"
python plot_results.py charmsycl_cuda "Charm-SYCL IRIS CUDA"
python plot_results.py charmsycl_cuda_directly "Charm-SYCL (Internal) CUDA"
python plot_results.py charmsycl_openmp "Charm-SYCL IRIS OpenMP"
python plot_results.py charmsycl_openmp_directly "Charm-SYCL (Internal) OpenMP"
python plot_results.py opensycl_openmp "OpenSYCL OpenMP"
pdfunite figures/iris_cuda.pdf figures/charmsycl_cuda.pdf figures/charmsycl_cuda_directly.pdf figures/dpc++_cuda.pdf figures/cuda_comparison.pdf
pdfunite figures/iris_openmp.pdf figures/charmsycl_openmp.pdf figures/charmsycl_openmp_directly.pdf figures/opensycl_openmp.pdf figures/openmp_comparison.pdf
pdfunite figures/iris_hip.pdf figures/charmsycl_hip.pdf figures/charmsycl_hip_directly.pdf figures/hip_comparison.pdf
