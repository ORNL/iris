# deps/build.jl
using Pkg
using CxxWrap  # or any other packages you need

println("Starting C++ build process...")

# For example, using a system call to run CMake:
run(`cmake -S . -B build`)
run(`cmake --build build`)

println("C++ build process finished.")
# Create the destination directory if needed and copy the built library there.
lib_dest = joinpath(@__DIR__, "..", "lib64")
mkpath(lib_dest)  # Ensure directory exists
cp(joinpath("build", "libiris.so"), joinpath(lib_dest, "libiris.so"); force=true)

println("CMake build finished and libiris.so copied to lib64/")
