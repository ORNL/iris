#!/bin/bash
rm -rf figures ; mkdir figures
python plot_results.py iris_openmp "IRIS OpenMP"
python plot_results.py iris_cuda "IRIS CUDA"
python plot_results.py dpc++_cuda "DPC++ CUDA"
python plot_results.py charmsycl_cuda "Charm-SYCL IRIS CUDA"
python plot_results.py charmsycl_cuda_directly "Charm-SYCL (Internal) CUDA"
python plot_results.py charmsycl_openmp "Charm-SYCL IRIS OpenMP"
python plot_results.py charmsycl_cpu_directly "Charm-SYCL (Internal) CPU"
python plot_results.py opensycl_openmp "OpenSYCL OpenMP"
