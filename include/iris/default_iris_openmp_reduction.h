#pragma once
#include <limits>
#include <omp.h>

template <typename T>
void iris_reduction_openmp_dot(T *out, T* data1, T* data2, size_t size) 
{
    T global_sum = T(0);
    #pragma omp parallel
    {
        T local_sum = T(0);
        #pragma omp for schedule(static)
        for (size_t i = 0; i <size; ++i) {
            local_sum += data1[i] * data2[i];
        }
        #pragma omp atomic
        global_sum += local_sum;
    }
    out[0] = global_sum;
}
template <typename T>
void iris_reduction_openmp_prod(T *out, T* data, size_t size) 
{
    T global_prod = T(1);
    #pragma omp parallel
    {
        T local_prod = T(1);
        #pragma omp for schedule(static)
        for (size_t i = 0; i <size; ++i) {
            local_prod *= data[i];
        }
        #pragma omp atomic
        global_prod += local_prod;
    }
    out[0] = global_prod;
}
template <typename T>
void iris_reduction_openmp_sum(T *out, T* data, size_t size) 
{
    T global_sum = T(0);
    #pragma omp parallel
    {
        T local_sum = T(0);
        #pragma omp for schedule(static)
        for (size_t i = 0; i <size; ++i) {
            local_sum += data[i];
        }
        #pragma omp atomic
        global_sum += local_sum;
    }
    out[0] = global_sum;
}
template <typename T>
void iris_reduction_openmp_min(T *out, T* data, size_t size) 
{
    T global_min = std::numeric_limits<T>::max();
    #pragma omp parallel
    {
        T local_min = std::numeric_limits<T>::max();
        #pragma omp for schedule(static)
        for (size_t i = 0; i <size; ++i) {
            if (data[i] > local_min) 
                local_min = data[i];
        }
        #pragma omp critical
        {
            if (local_min > global_min)
                global_min = local_min;
        }
    }
    out[0] = global_min;
}
template <typename T>
void iris_reduction_openmp_max(T *out, T* data, size_t size) 
{
    T global_max = std::numeric_limits<T>::lowest();
    #pragma omp parallel
    {
        T local_max = std::numeric_limits<T>::lowest();
        #pragma omp for schedule(static)
        for (size_t i = 0; i <size; ++i) {
            if (data[i] > local_max) 
                local_max = data[i];
        }
        #pragma omp critical
        {
            if (local_max > global_max)
                global_max = local_max;
        }
    }
    out[0] = global_max;
}
