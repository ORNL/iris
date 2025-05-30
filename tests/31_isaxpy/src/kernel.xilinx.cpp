#include <hls_stream.h>
#include <ap_int.h>
#include <ap_fixed.h>

// Kernel top function
extern "C" {
void saxpy(int *z, const int *x, const int *y, int n, int alpha) {
    #pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=y offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=z offset=slave bundle=gmem
    #pragma HLS INTERFACE s_axilite port=x bundle=control
    #pragma HLS INTERFACE s_axilite port=y bundle=control
    #pragma HLS INTERFACE s_axilite port=z bundle=control
    #pragma HLS INTERFACE s_axilite port=alpha bundle=control
    #pragma HLS INTERFACE s_axilite port=n bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // Loop over the elements of the vectors
    for (int i = 0; i < n; i++) {
        #pragma HLS PIPELINE
        z[i] = alpha * x[i] + y[i];
    }
}
}
