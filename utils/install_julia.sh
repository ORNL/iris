#!/bin/bash
set -x
HOME=${HOME:-/}
JULIA_CACHE_NAME=${JULIA_CACHE_NAME:-.julia}
JULIA_CACHE_DEBUG_NAME=${JULIA_CACHE_DEBUG_NAME:-$JULIA_CACHE_NAME.debug}
JULIA_CACHE_PATH=$HOME/$JULIA_CACHE_NAME
JULIA_CACHE_DEBUG_PATH=$HOME/$JULIA_CACHE_DEBUG_NAME
CWD=`pwd`
SKIP_SETUP=${SKIP_SETUP:-0}
SKIP_JULIA=${SKIP_JULIA:-0}
SKIP_DEBUG=${SKIP_DEBUG:-0}
JULIA_INSTALL_DIR=${JULIA_INSTALL_DIR:-$CWD/software}
echo "CWD: $CWD"
echo "JULIA_INSTALL_DIR: $JULIA_INSTALL_DIR"
mkdir -p $JULIA_CACHE_PATH
mkdir -p $JULIA_CACHE_DEBUG_PATH
if [ "x$SKIP_SETUP" = "x0" ]; then
    if [ "x$SKIP_JULIA" = "x0" ]; then
        mkdir -p $JULIA_INSTALL_DIR
        cd $JULIA_INSTALL_DIR
        git clone https://github.com/JuliaLang/julia.git
        cd julia
        git checkout cbc47c9fb852773e379cce062d6bd2aec743a7cf
        cd $CWD
    fi
fi
if [ ! -e AMDGPU.jl ]; then
    git clone https://github.com/JuliaGPU/AMDGPU.jl.git
    cd AMDGPU.jl
    git pull
    git checkout 98fdeb619977152c9bc58e72ab716cfb81457f47
    cd ..
    sed -e s/'LLD_jll = "\(15, [^"]*\)"/LLD_jll = "\1, 18, 19\"/g' -e s/'LLVM_jll = \"\(15, [^"]*\)\"'/'LLVM_jll = \"\1, 18, 19\"'/g -i AMDGPU.jl/Project.toml
fi
#if [ ! -e LLD_jll.jl ]; then
#    git clone https://github.com/JuliaBinaryWrappers/LLD_jll.jl
#    cd LLD_jll.jl
#    git pull
#    git checkout 655a902a8c354fd5d367392b98b167d799431acc
#    cd ..
#fi
#if [ ! -e CUDA.jl ]; then
#    git clone https://github.com/JuliaGPU/CUDA.jl.git
#    cd CUDA.jl
#    git pull
#     git checkout 3d45d8537ef0f34ba7067a709f47390d0e42508e
#    cd ..
#fi
#if [ ! -e GPUCompiler.jl ]; then
#    git clone https://github.com/JuliaGPU/GPUCompiler.jl.git
#    cd GPUCompiler.jl
#    git pull
#     git checkout 8ae152be7337fdd2217a2472b1816647f6e3c730
#    cd ..
#fi
#if [ ! -e KernelAbstractions.jl ]; then
#    git clone https://github.com/JuliaGPU/KernelAbstractions.jl
#    cd KernelAbstractions.jl
#    git pull
#    git checkout f198f5c8ab1a60fa265279406ab458cfc29f8239
#    cd ..
#fi
if [ "x$SKIP_SETUP" = "x0" ]; then
    if [ "x$SKIP_JULIA" = "x0" ]; then
        if [ "x$SKIP_DEBUG" = "x0" ]; then
            cp -r julia julia.debug
        fi
        cd julia
        ln -s $JULIA_INSTALL_DIR/AMDGPU.jl .
        #ln -s $JULIA_INSTALL_DIR/LLD_jll.jl .
        #ln -s $JULIA_INSTALL_DIR/GPUCompiler.jl .
        #ln -s $JULIA_INSTALL_DIR/CUDA.jl .
        #ln -s $JULIA_INSTALL_DIR/KernelAbstractions.jl .
        make -j
        echo "export JULIA_INSTALL_DIR=$JULIA_INSTALL_DIR" > setup.source
        echo "export JULIA_CACHE_NAME=$JULIA_CACHE_NAME" >> setup.source
        echo "export JULIA_CACHE_PATH=$JULIA_CACHE_PATH" >> setup.source
        echo 'export JULIA=$JULIA_INSTALL_DIR/julia/usr' >> setup.source
        echo 'export JULIA_DEPOT_PATH=$HOME/$JULIA_CACHE_NAME:$JULIA_CACHE_PATH/' >> setup.source
        echo 'export PATH=$JULIA/bin:$PATH' >> setup.source
        echo 'export LD_LIBRARY_PATH=$JULIA/lib:$LD_LIBRARY_PATH' >> setup.source
        cd ..
    fi
    if [ "x$SKIP_DEBUG" = "x0" ]; then
        if [ "x$SKIP_JULIA" = "x0" ]; then
            cd julia.debug
            ln -s $JULIA_INSTALL_DIR/AMDGPU.jl .
            #ln -s $JULIA_INSTALL_DIR/LLD_jll.jl .
            #ln -s $JULIA_INSTALL_DIR/GPUCompiler.jl .
            #ln -s $JULIA_INSTALL_DIR/CUDA.jl .
            #ln -s $JULIA_INSTALL_DIR/KernelAbstractions.jl .
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
fi
if [ "x$SKIP_SETUP" = "x0" ]; then
    if [ "x$SKIP_JULIA" = "x0" ]; then
        cd $JULIA_INSTALL_DIR
        . $JULIA_INSTALL_DIR/julia/setup.source
        cd $CWD
    fi
fi
export JULIA_DEPOT_PATH=$JULIA_CACHE_PATH
#julia -e "using Pkg; Pkg.add(PackageSpec(name=\"LLD_jll\", version=\"19.1.7\"))"
#julia -e "using Pkg; Pkg.add(PackageSpec(path=\"./GPUCompiler.jl\"))" 
#julia -e "using Pkg; Pkg.add(PackageSpec(path=\"./CUDA.jl\"))" 
#julia -e 'using Pkg; Pkg.add(PackageSpec(url="https://github.com/JuliaBinaryWrappers/LLD_jll.jl", rev="655a902a8c354fd5d367392b98b167d799431acc"))'
#julia -e 'using Pkg; Pkg.add(PackageSpec(url="https://github.com/JuliaGPU/GPUCompiler.jl", rev="8ae152be7337fdd2217a2472b1816647f6e3c730"))'
julia -e 'using Pkg; Pkg.add(PackageSpec(url="https://github.com/JuliaGPU/GPUCompiler.jl"))'
#julia -e 'using Pkg; Pkg.add(PackageSpec(url="https://github.com/JuliaGPU/CUDA.jl", rev="3d45d8537ef0f34ba7067a709f47390d0e42508e"))'
julia -e 'using Pkg; Pkg.add(PackageSpec(url="https://github.com/JuliaGPU/CUDA.jl"))'
julia -e "using Pkg; Pkg.develop(PackageSpec(path=\"./AMDGPU.jl\"))" 
julia -e "using Pkg; Pkg.add(\"Requires\")"
julia -e "using Pkg; Pkg.add(\"Dagger\")"
julia -e "using Pkg; Pkg.add(\"StatsBase\")"
julia -e "using Pkg; Pkg.add(\"ProgressMeter\")"
julia -e "using Pkg; Pkg.add(\"Distributions\")"
julia -e 'using Pkg; Pkg.add(PackageSpec(url="https://github.com/JuliaORNL/JACC.jl", rev="v0.3.1"))'
julia -e 'using Pkg; Pkg.add(PackageSpec(url="https://github.com/miniskar/LoopVectorization.jl", rev="6aea6fb4effd100df12f73f429a5b09ce7a9ab93"))'
julia -e 'using Pkg; Pkg.add(PackageSpec(url="https://github.com/JuliaGPU/KernelAbstractions.jl", rev="f198f5c8ab1a60fa265279406ab458cfc29f8239"))'
#julia -e "using Pkg; Pkg.add(PackageSpec(path=\"./KernelAbstractions.jl\"))" 
if [ -e /usr/lib/x86_64-linux-gnu/libcuda.so ]; then
    for f in `find $JULIA_CACHE_PATH -name "libcuda.so"`; do 
        rm -f $f ; ln -s /usr/lib/x86_64-linux-gnu/libcuda.so $f ;
    done
fi
if [ "x$SKIP_SETUP" = "x0" ]; then
    if [ "x$SKIP_DEBUG" = "x0" ]; then
        if [ "x$SKIP_JULIA" = "x0" ]; then
            . $JULIA_INSTALL_DIR/julia.debug/setup.source
        fi
        export JULIA_DEPOT_PATH=$JULIA_CACHE_PATH.debug
        #julia-debug -e "using Pkg; Pkg.add(PackageSpec(name=\"LLD_jll\", version=\"19.1.7\"))"
        #julia-debug -e "using Pkg; Pkg.add(PackageSpec(path=\"./GPUCompiler.jl\"))" 
        #julia-debug -e "using Pkg; Pkg.add(PackageSpec(path=\"./CUDA.jl\"))" 
		#julia-debug -e 'using Pkg; Pkg.add(PackageSpec(url="https://github.com/JuliaBinaryWrappers/LLD_jll.jl", rev="655a902a8c354fd5d367392b98b167d799431acc", version="19.1.7"))'
        julia-debug -e 'using Pkg; Pkg.add(PackageSpec(url="https://github.com/JuliaGPU/GPUCompiler.jl", rev="8ae152be7337fdd2217a2472b1816647f6e3c730"))'
        julia-debug -e 'using Pkg; Pkg.add(PackageSpec(url="https://github.com/JuliaGPU/CUDA.jl", rev="3d45d8537ef0f34ba7067a709f47390d0e42508e"))'
        julia-debug -e "using Pkg; Pkg.develop(PackageSpec(path=\"./AMDGPU.jl\"))" 
        julia-debug -e "using Pkg; Pkg.add(\"Requires\")"
        julia-debug -e "using Pkg; Pkg.add(\"Dagger\")"
        julia-debug -e "using Pkg; Pkg.add(\"StatsBase\")"
        julia-debug -e "using Pkg; Pkg.add(\"ProgressMeter\")"
        julia-debug -e "using Pkg; Pkg.add(\"Distributions\")"
        julia-debug -e 'using Pkg; Pkg.add(PackageSpec(url="https://github.com/JuliaORNL/JACC.jl", rev="v0.3.1"))'
        julia-debug -e 'using Pkg; Pkg.add(PackageSpec(url="https://github.com/miniskar/LoopVectorization.jl", rev="6aea6fb4effd100df12f73f429a5b09ce7a9ab93"))'
        julia-debug -e 'using Pkg; Pkg.add(PackageSpec(url="https://github.com/JuliaGPU/KernelAbstractions.jl", rev="f198f5c8ab1a60fa265279406ab458cfc29f8239"))'
        for f in `find $JULIA_CACHE_DEBUG_PATH -name "libcuda.so"`; do 
            rm -f $f ; ln -s /usr/lib/x86_64-linux-gnu/libcuda.so $f ;
        done
    fi
fi

