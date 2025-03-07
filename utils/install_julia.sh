#!/bin/bash
set -x
HOME=${HOME:-/}
JULIA_CACHE_NAME=${JULIA_CACHE_NAME:-.julia}
JULIA_CACHE_DEBUG_NAME=${JULIA_CACHE_DEBUG_NAME:-$JULIA_CACHE_NAME.debug}
JULIA_CACHE_PATH=$HOME/$JULIA_CACHE_NAME
JULIA_CACHE_DEBUG_PATH=$HOME/$JULIA_CACHE_DEBUG_NAME
CWD=`pwd`
SKIP_SETUP=${SKIP_SETUP:-0}
SKIP_DEBUG=${SKIP_DEBUG:-0}
JULIA_INSTALL_DIR=${JULIA_INSTALL_DIR:-$CWD/software}
echo "CWD: $CWD"
echo "JULIA_INSTALL_DIR: $JULIA_INSTALL_DIR"
mkdir -p $JULIA_CACHE_PATH
mkdir -p $JULIA_CACHE_DEBUG_PATH
if [ "x$SKIP_SETUP" = "x0" ]; then
    mkdir -p $JULIA_INSTALL_DIR
    cd $JULIA_INSTALL_DIR
    git clone https://github.com/JuliaLang/julia.git
    cd julia
    git checkout cbc47c9fb852773e379cce062d6bd2aec743a7cf
    cd ..
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
    sed -e s/'LLD_jll = "\(15, [^"]*\)"/LLD_jll = "\1, 18, 19\"/g' -e s/'LLVM_jll = \"\(15, [^"]*\)\"'/'LLVM_jll = \"\1, 18, 19\"'/g -i AMDGPU.jl/Project.toml
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
    if [ "x$SKIP_DEBUG" = "x0" ]; then
        cp -r julia julia.debug
    fi
    cd julia
    ln -s $JULIA_INSTALL_DIR/GPUCompiler.jl .
    ln -s $JULIA_INSTALL_DIR/CUDA.jl .
    ln -s $JULIA_INSTALL_DIR/AMDGPU.jl .
    ln -s $JULIA_INSTALL_DIR/LLD_jll.jl .
    make -j
    echo "export JULIA_INSTALL_DIR=$JULIA_INSTALL_DIR" > setup.source
    echo "export JULIA_CACHE_NAME=$JULIA_CACHE_NAME" >> setup.source
    echo "export JULIA_CACHE_PATH=$JULIA_CACHE_PATH" >> setup.source
    echo 'export JULIA=$JULIA_INSTALL_DIR/julia/usr' >> setup.source
    echo 'export JULIA_DEPOT_PATH=$HOME/$JULIA_CACHE_NAME:$JULIA_CACHE_PATH/' >> setup.source
    echo 'export PATH=$JULIA/bin:$PATH' >> setup.source
    echo 'export LD_LIBRARY_PATH=$JULIA/lib:$LD_LIBRARY_PATH' >> setup.source
    cd ..
    if [ "x$SKIP_DEBUG" = "x0" ]; then
        cd julia.debug
        ln -s $JULIA_INSTALL_DIR/GPUCompiler.jl .
        ln -s $JULIA_INSTALL_DIR/CUDA.jl .
        ln -s $JULIA_INSTALL_DIR/AMDGPU.jl .
        ln -s $JULIA_INSTALL_DIR/LLD_jll.jl .
        make cleanall
        make -j debug
        echo "export JULIA_INSTALL_DIR=$JULIA_INSTALL_DIR" > setup.source
        echo "export JULIA_CACHE_NAME=$JULIA_CACHE_DEBUG_NAME" >> setup.source
        echo "export JULIA_CACHE_PATH=$JULIA_CACHE_DEBUG_PATH" >> setup.source
        echo 'export JULIA=$JULIA_INSTALL_DIR/julia/usr' >> setup.source
        echo 'export JULIA_DEPOT_PATH=$HOME/$JULIA_CACHE_NAME:$JULIA_CACHE_PATH/' >> setup.source
        echo 'export PATH=$JULIA/bin:$PATH' >> setup.source
        echo 'export LD_LIBRARY_PATH=$JULIA/lib:$LD_LIBRARY_PATH' >> setup.source
        cd ..
    fi
fi
if [ "x$SKIP_SETUP" = "x0" ]; then
    cd $JULIA_INSTALL_DIR
    . $JULIA_INSTALL_DIR/julia/setup.source
fi
export JULIA_DEPOT_PATH=$JULIA_CACHE_PATH
julia -e "using Pkg; Pkg.add(PackageSpec(name=\"LLD_jll\", version=\"19.1.7\"))"
julia -e "using Pkg; Pkg.add(PackageSpec(path=\"./GPUCompiler.jl\"))" 
julia -e "using Pkg; Pkg.develop(PackageSpec(path=\"./AMDGPU.jl\"))" 
julia -e "using Pkg; Pkg.add(PackageSpec(path=\"./CUDA.jl\"))" 
julia -e "using Pkg; Pkg.add(\"Requires\")"
julia -e "using Pkg; Pkg.add(\"Dagger\")"
if [ -e /usr/lib/x86_64-linux-gnu/libcuda.so ]; then
    for f in `find $JULIA_CACHE_PATH -name "libcuda.so"`; do 
        rm -f $f ; ln -s /usr/lib/x86_64-linux-gnu/libcuda.so $f ;
    done
fi
if [ "x$SKIP_SETUP" = "x0" ]; then
    if [ "x$SKIP_DEBUG" = "x0" ]; then
        . $JULIA_INSTALL_DIR/julia.debug/setup.source
        export JULIA_DEPOT_PATH=$JULIA_CACHE_PATH.debug
        julia-debug -e "using Pkg; Pkg.add(PackageSpec(name=\"LLD_jll\", version=\"19.1.7\"))"
        julia-debug -e "using Pkg; Pkg.add(PackageSpec(path=\"./GPUCompiler.jl\"))" 
        julia-debug -e "using Pkg; Pkg.develop(PackageSpec(path=\"./AMDGPU.jl\"))" 
        julia-debug -e "using Pkg; Pkg.add(PackageSpec(path=\"./CUDA.jl\"))" 
        julia-debug -e "using Pkg; Pkg.add(\"Requires\")"
        julia-debug -e "using Pkg; Pkg.add(\"Dagger\")"
        for f in `find $JULIA_CACHE_DEBUG_PATH -name "libcuda.so"`; do 
            rm -f $f ; ln -s /usr/lib/x86_64-linux-gnu/libcuda.so $f ;
        done
    fi
fi

