#include "iris/iris.h"

int main(int argc, char **argv)
{
    // IRIS Initialization
    iris_init(&argc, &argv, 1);

    // Create and initialize Src memory
    int N=16;
    int *src_data = (int*)malloc(sizeof(int)*N);
    for(int i=0; i<N; i++) src_data[i] = i+1;

    // Create and initialize Src memory
    int *dst_data = (int*)malloc(sizeof(int)*N);
    for(int i=0; i<N; i++) dst_data[i] = i+1;

    // Create IRIS DMEM objects for source and destination
    iris_dmem src = iris_data_mem_create_struct(src_data, N*sizeof(int));
    iris_dmem dst = iris_data_mem_create_struct(dst_data, N*sizeof(int));

    // Create IRIS task
    iris_task task = iris_task_create_struct();

    // Create DMEM2DMEM command in task
    iris_task_dmem2dmem(task, src, dst);

    // Flush destination DMEM to host memory
    iris_task_dmem_flush_out(task, dst);

    // Submit the task
    iris_task_submit(task, iris_default, NULL, 1);

    // Compare output
    int nerrors = 0;
    for(int i=0; i<N; i++) {
        if (src[i] != dst[i]) nerrors++;
    }

    // Clear
    iris_mem_release(src);
    iris_mem_release(dst);
    free(src_data);
    free(dst_data);

    // IRIS finalize
    iris_finalize();
    return nerrors;
}