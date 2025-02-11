#!/bin/bash
set -x
HOME=${HOME:-/}
JULIA_CACHE_NAME=${JULIA_CACHE_NAME:-.julia}
JULIA_CACHE_DEBUG_NAME=${JULIA_CACHE_DEBUG_NAME:-$JULIA_CACHE_NAME.debug}
JULIA_CACHE_PATH=$HOME/$JULIA_CACHE_NAME
JULIA_CACHE_DEBUG_PATH=$HOME/$JULIA_CACHE_DEBUG_NAME
CWD=`pwd`
SKIP_SETUP=${SKIP_SETUP:-0}
mkdir -p $JULIA_CACHE_PATH
mkdir -p $JULIA_CACHE_DEBUG_PATH
if [ "x$SKIP_SETUP" = "x0" ]; then
    mkdir -p $CWD/software
    cd software
    git clone https://github.com/JuliaLang/julia.git
fi
if [ ! -e CUDA.jl ]; then
    git clone https://github.com/JuliaGPU/CUDA.jl.git
    cd CUDA.jl
    git pull
    cd ..
fi
if [ ! -e AMDGPU.jl ]; then
    git clone https://github.com/JuliaGPU/AMDGPU.jl.git
    cd AMDGPU.jl
    git pull
    cd ..
    sed -e s/'LLD_jll = "\(15, [^"]*\)"/LLD_jll = "\1, 18, 19\"/g' -e s/'LLVM_jll = \"\(15, [^"]*\)\"'/'LLVM_jll = \"\1, 18\"'/g -i AMDGPU.jl/Project.toml
fi
if [ ! -e GPUCompiler.jl ]; then
    git clone https://github.com/JuliaGPU/GPUCompiler.jl.git
    cd GPUCompiler.jl
    git pull
    cd ..
fi
if [ ! -e LLD_jll.jl ]; then
    git clone https://github.com/JuliaBinaryWrappers/LLD_jll.jl
    cd LLD_jll.jl
    git pull
    cd ..
fi
if [ "x$SKIP_SETUP" = "x0" ]; then
    cp -r julia julia.debug
    cd julia
    ln -s $CWD/software/GPUCompiler.jl .
    ln -s $CWD/software/CUDA.jl .
    ln -s $CWD/software/AMDGPU.jl .
    ln -s $CWD/software/LLD_jll.jl .
    make -j
    echo "export CWD=$CWD" > setup.source
    echo "export JULIA_CACHE_NAME=$JULIA_CACHE_NAME" >> setup.source
    echo "export JULIA_CACHE_PATH=$JULIA_CACHE_PATH" >> setup.source
    echo 'export JULIA=$CWD/software/julia/usr' >> setup.source
    echo 'export JULIA_DEPOT_PATH=$HOME/$JULIA_CACHE_NAME:$JULIA_CACHE_PATH/' >> setup.source
    echo 'export PATH=$JULIA/bin:$PATH' >> setup.source
    echo 'export LD_LIBRARY_PATH=$JULIA/lib:$LD_LIBRARY_PATH' >> setup.source
    cd ..
    cd julia.debug
    ln -s $CWD/software/GPUCompiler.jl .
    ln -s $CWD/software/CUDA.jl .
    ln -s $CWD/software/AMDGPU.jl .
    ln -s $CWD/software/LLD_jll.jl .
    make cleanall
    make -j debug
    echo "export CWD=$CWD" > setup.source
    echo "export JULIA_CACHE_NAME=$JULIA_CACHE_DEBUG_NAME" >> setup.source
    echo "export JULIA_CACHE_PATH=$JULIA_CACHE_DEBUG_PATH" >> setup.source
    echo 'export JULIA=$CWD/software/julia/usr' >> setup.source
    echo 'export JULIA_DEPOT_PATH=$HOME/$JULIA_CACHE_NAME:$JULIA_CACHE_PATH/' >> setup.source
    echo 'export PATH=$JULIA/bin:$PATH' >> setup.source
    echo 'export LD_LIBRARY_PATH=$JULIA/lib:$LD_LIBRARY_PATH' >> setup.source
    cd ..
fi
if [ "x$SKIP_SETUP" = "x0" ]; then
    cd $CWD/software
    . $CWD/software/julia/setup.source
fi
export JULIA_DEPOT_PATH=$JULIA_CACHE_PATH
julia -e "using Pkg; Pkg.add(PackageSpec(name=\"LLD_jll\", version=\"19.1.1\"))"
julia -e "using Pkg; Pkg.add(PackageSpec(path=\"./GPUCompiler.jl\"))" 
julia -e "using Pkg; Pkg.develop(PackageSpec(path=\"./AMDGPU.jl\"))" 
julia -e "using Pkg; Pkg.add(PackageSpec(path=\"./CUDA.jl\"))" 
julia -e "using Pkg; Pkg.add(\"Requires\")"
julia -e "using Pkg; Pkg.add(\"Dagger\")"
for f in `find $JULIA_CACHE_PATH -name "libcuda.so"`; do 
    rm -f $f ; ln -s /usr/lib/x86_64-linux-gnu/libcuda.so $f ;
done
if [ "x$SKIP_SETUP" = "x0" ]; then
    . $CWD/software/julia.debug/setup.source
    export JULIA_DEPOT_PATH=$JULIA_CACHE_PATH.debug
    julia-debug -e "using Pkg; Pkg.add(PackageSpec(name=\"LLD_jll\", version=\"19.1.1\"))"
    julia-debug -e "using Pkg; Pkg.add(PackageSpec(path=\"./GPUCompiler.jl\"))" 
    julia-debug -e "using Pkg; Pkg.develop(PackageSpec(path=\"./AMDGPU.jl\"))" 
    julia-debug -e "using Pkg; Pkg.add(PackageSpec(path=\"./CUDA.jl\"))" 
    julia-debug -e "using Pkg; Pkg.add(\"Requires\")"
    julia-debug -e "using Pkg; Pkg.add(\"Dagger\")"
    for f in `find $JULIA_CACHE_DEBUG_PATH -name "libcuda.so"`; do 
        rm -f $f ; ln -s /usr/lib/x86_64-linux-gnu/libcuda.so $f ;
    done
fi

