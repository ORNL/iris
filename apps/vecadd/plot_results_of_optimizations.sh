#!/bin/bash
mkdir -p figures
python plot_results.py iris_openmp_data_memory "IRIS OpenMP (Data-Memory)"
python plot_results.py iris_cuda_data_memory "IRIS CUDA (Data-Memory)"
python plot_results.py iris_hip_data_memory "IRIS HIP (Data-Memory)"
python plot_results.py dpc++_cuda_discard_write "DPC++ CUDA (Discard-Write)"
python plot_results.py charmsycl_cuda_discard_write "Charm-SYCL IRIS CUDA (Discard-Write)"
python plot_results.py charmsycl_cuda_directly_discard_write "Charm-SYCL (Internal) CUDA (Discard-Write)"
python plot_results.py charmsycl_hip_discard_write "Charm-SYCL IRIS HIP (Discard-Write)"
python plot_results.py charmsycl_hip_directly_discard_write "Charm-SYCL (Internal) HIP (Discard-Write)"
python plot_results.py charmsycl_openmp_discard_write "Charm-SYCL IRIS OpenMP (Discard-Write)"
python plot_results.py charmsycl_openmp_directly_discard_write "Charm-SYCL (Internal) OpenMP (Discard-Write)"
python plot_results.py opensycl_openmp_discard_write "OpenSYCL OpenMP (Discard-Write)"
pdfunite figures/iris_cuda_data_memory.pdf figures/charmsycl_cuda_discard_write.pdf figures/charmsycl_cuda_directly_discard_write.pdf figures/dpc++_cuda_discard_write.pdf figures/cuda_comparison_optimization.pdf
pdfunite figures/iris_hip_data_memory.pdf figures/charmsycl_hip_discard_write.pdf figures/charmsycl_hip_directly_discard_write.pdf figures/hip_comparison_optimization.pdf
pdfunite figures/iris_openmp_data_memory.pdf figures/charmsycl_openmp_discard_write.pdf figures/charmsycl_openmp_directly_discard_write.pdf figures/opensycl_openmp_discard_write.pdf figures/openmp_comparison_optimization.pdf
pdfunite figures/iris_cuda.pdf figures/iris_cuda_data_memory.pdf figures/charmsycl_cuda.pdf figures/charmsycl_cuda_discard_write.pdf figures/charmsycl_cuda_directly.pdf figures/charmsycl_cuda_directly_discard_write.pdf figures/dpc++_cuda.pdf figures/dpc++_cuda_discard_write.pdf figures/cuda_comparison_optimization_full.pdf
pdfunite figures/iris_hip.pdf figures/iris_hip_data_memory.pdf figures/charmsycl_hip.pdf figures/charmsycl_hip_discard_write.pdf figures/charmsycl_hip_directly.pdf figures/charmsycl_hip_directly_discard_write.pdf figures/hip_comparison_optimization_full.pdf
pdfunite figures/iris_openmp.pdf figures/iris_openmp_data_memory.pdf figures/charmsycl_openmp.pdf figures/charmsycl_openmp_discard_write.pdf figures/charmsycl_openmp_directly.pdf figures/charmsycl_openmp_directly_discard_write.pdf figures/opensycl_openmp.pdf figures/opensycl_openmp_discard_write.pdf figures/openmp_comparison_optimization_full.pdf
