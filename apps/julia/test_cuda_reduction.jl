using CUDA

# Define a kernel that computes a block-wise dot product
function kernel_dot(a, b, partial_sum)
    # Allocate dynamic shared memory: one Float32 per thread
    sdata = @cuDynamicSharedMem(Float32, blockDim().x)
    tid = threadIdx().x
    i = (blockIdx().x - 1) * blockDim().x + tid

    # Each thread computes a[i] * b[i] (if within bounds)
    tmp = (i <= length(a)) ? a[i] * b[i] : 0.0f0
    sdata[tid] = tmp
    sync_threads()

    # Perform reduction using a for loop instead of while loop
    nsteps = floor(Int, log2(blockDim().x))
    for s in 1:nsteps
        offset = div(blockDim().x, 2^s)
        if tid <= offset
             sdata[tid] += sdata[tid + offset]
        end
        sync_threads()
    end
    # The first thread in the block writes the block’s result to global memory
    if tid == 1
        partial_sum[blockIdx().x] = sdata[1]
    end
    return
end

# A helper function to launch the kernel and compute the final dot product
function gpu_dot(a::Vector{Float32}, b::Vector{Float32})
    n = length(a)
    # Transfer data to the GPU
    d_a = CuArray(a)
    d_b = CuArray(b)

    # Define threads per block and compute the number of blocks
    threads = 256
    blocks = cld(n, threads)
    
    # Allocate an array on the GPU to hold per-block partial sums
    d_partial = CuArray(zeros(Float32, blocks))

    # Calculate the shared memory size in bytes (one Float32 per thread)
    shmem = threads * sizeof(Float32)

    # Launch the kernel
    @cuda threads=threads blocks=blocks shmem=shmem kernel_dot(d_a, d_b, d_partial)
    CUDA.synchronize()
    
    # Sum the per-block results on the CPU (for large arrays, you might launch another kernel)
    result = sum(Array(d_partial))
    return result
end

# Example usage:
a = rand(Float32, 1000)
b = rand(Float32, 1000)
result = gpu_dot(a, b)
println("Dot product (reduction of a[i]*b[i]): ", result)

# For verification, compare with the CPU computed dot product:
println("CPU dot product: ", sum(a .* b))

