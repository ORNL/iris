// rope_omp_raw.hpp
#pragma once

#include <cmath>
#include <cstddef>
#include <omp.h>

template<typename T>
void iris_rope_cpu(
    T* x,               // interleaved complex buffer: [re,im, re,im, …]
    std::size_t cols,
    std::size_t rows,
    int pos,
    T theta_scale
) 
{
    //printf(" rows:%lu cols:%lu pos:%d theta_scale:%f\n", rows, cols, pos, theta_scale);
    #pragma omp parallel for collapse(2) schedule(static)
    for (std::size_t head = 0; head < rows; ++head) {
        for (std::size_t i = 0; i < cols; ++i) {
            // compute θ = (pos-1) * theta_scale^(i-1)
            //theta = Float32(pos - 1) * (theta_scale)^(i - 1)
            T exponent = (i == 0 ? T(1) : std::pow(theta_scale, T(i)));
            T theta    = T(pos - 1) * exponent;
            T c = std::cos(theta);
            T s = std::sin(theta);

            // index into interleaved buffer
            std::size_t idx2 = 2 * (i + head * cols);
            T a = x[idx2 + 0];  // real
            T b = x[idx2 + 1];  // imag

            // (a+ib)*(c+is) = (a*c - b*s) + i(a*s + b*c)
            x[idx2 + 0] = a*c - b*s;
            x[idx2 + 1] = a*s + b*c;
            //printf("(%d, %d) a,b: (%f+i%f) out:%f+i%f theta:%f\n", i, head, a, b, x[idx2], x[idx2+1], theta);
        }
    }
}

