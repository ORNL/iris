#include "iris/iris.h"

int main(int argc, char **argv)
{
    // IRIS Initialization
    iris_init(&argc, &argv, 1);

    // Create and initialize Src memory
    int N=16;
    int *src_data = new int[N];
    for(int i=0; i<N; i++) src_data[i] = i+1;

    // Create and initialize Src memory
    int *dst_data = new int[N];
    for(int i=0; i<N; i++) dst_data[i] = i+1;

    // Create IRIS DMEM objects for source and destination
    iris::DMemArray<int> src(src_data, N);
    iris::DMemArray<int> dst(dst_data, N);

    // Create IRIS task
    iris::Task task();
    
    // Create DMEM2DMEM command in task
    task.dmem2dmem(src, dst);

    // Flush destination DMEM to host memory
    task.flush(dst);

    // Submit the task
    task.submit(sync=true);

    // Compare output
    int nerrors = 0;
    for(int i=0; i<N; i++) {
        if (src[i] != dst[i]) nerrors++;
    }

    // Clear
    DMem.release(src);
    DMem.release(dst);
    delete [] src_data;
    delete [] dst_data;

    // IRIS finalize
    iris_finalize();
    return nerrors;
}