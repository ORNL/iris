#!/bin/bash
mkdir -p figures
python plot_results.py iris_openmp_data_memory "IRIS OpenMP (Data-Memory)"
python plot_results.py iris_cuda_data_memory "IRIS CUDA (Data-Memory)"
python plot_results.py dpc++_cuda_discard_write "DPC++ CUDA (Discard-Write)"
python plot_results.py charmsycl_cuda_discard_write "Charm-SYCL IRIS CUDA (Discard-Write)"
python plot_results.py charmsycl_cuda_directly_discard_write "Charm-SYCL (Internal) CUDA (Discard-Write)"
python plot_results.py charmsycl_openmp_discard_write "Charm-SYCL IRIS OpenMP (Discard-Write)"
python plot_results.py charmsycl_cpu_directly_discard_write "Charm-SYCL (Internal) CPU (Discard-Write)"
python plot_results.py opensycl_openmp_discard_write "OpenSYCL OpenMP (Discard-Write)"
