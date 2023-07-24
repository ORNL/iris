#!/bin/bash
rm -rf figures/empty ; mkdir -p figures; mkdir figures/empty
python plot_results.py empty/iris_openmp "IRIS OpenMP" True
python plot_results.py empty/iris_cuda "IRIS CUDA" True
python plot_results.py empty/iris_hip "IRIS HIP" True
python plot_results.py empty/dpc++_cuda "DPC++ CUDA" True
python plot_results.py empty/charmsycl_cuda "Charm-SYCL IRIS CUDA" True
python plot_results.py empty/charmsycl_cuda_directly "Charm-SYCL (Internal) CUDA" True
python plot_results.py empty/charmsycl_hip "Charm-SYCL IRIS HIP" True
python plot_results.py empty/charmsycl_hip_directly "Charm-SYCL (Internal) HIP" True
python plot_results.py empty/charmsycl_openmp "Charm-SYCL IRIS OpenMP" True
python plot_results.py empty/charmsycl_openmp_directly "Charm-SYCL (Internal) OpenMP" True
python plot_results.py empty/opensycl_openmp "OpenSYCL OpenMP" True
pdfunite figures/empty/iris_cuda.pdf figures/empty/charmsycl_cuda.pdf figures/empty/charmsycl_cuda_directly.pdf figures/empty/dpc++_cuda.pdf figures/empty/cuda_comparison.pdf
pdfunite figures/empty/iris_hip.pdf figures/empty/charmsycl_hip.pdf figures/empty/charmsycl_hip_directly.pdf figures/empty/hip_comparison.pdf
pdfunite figures/empty/iris_openmp.pdf figures/empty/charmsycl_openmp.pdf figures/empty/charmsycl_openmp_directly.pdf figures/empty/opensycl_openmp.pdf figures/empty/openmp_comparison.pdf
pdfunite figures/empty/iris_hip_bar.pdf figures/empty/charmsycl_hip_bar.pdf figures/empty/charmsycl_hip_directly_bar.pdf figures/empty/hip_comparison_breakdown.pdf
pdfunite figures/empty/iris_cuda_bar.pdf figures/empty/charmsycl_cuda_bar.pdf figures/empty/charmsycl_cuda_directly_bar.pdf figures/empty/dpc++_cuda_bar.pdf figures/empty/cuda_comparison_breakdown.pdf
pdfunite figures/empty/iris_openmp_bar.pdf figures/empty/charmsycl_openmp_bar.pdf figures/empty/charmsycl_openmp_directly_bar.pdf figures/empty/opensycl_openmp_bar.pdf figures/empty/openmp_comparison_breakdown.pdf
