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
JULIA_NIGHTLY=${JULIA_NIGHTLY:-0}
JULIA_INSTALL_DIR=${JULIA_INSTALL_DIR:-$CWD/software}
date_variable=$(date +"%d-%b-%y")
JULIA_NIGHTLY_DIR=julia-nightly-${date_variable}
JULIA_DIR=julia-${date_variable}
JULIA_DEBUG_DIR=julia-debug-${date_variable}

julia_version=$(julia -v)
if [ "x$JULIA_NIGHTLY" = "x1" ]; then
    if [ -e julia-latest-linux-x86_64.tar.gz ]; then
        rm -f julia-latest-linux-x86_64.tar.gz
    fi
    wget https://julialangnightlies-s3.julialang.org/bin/linux/x86_64/julia-latest-linux-x86_64.tar.gz
    mkdir -p $JULIA_NIGHTLY_DIR
    tar -xvzf julia-latest-linux-x86_64.tar.gz --strip-components=1 -C ${JULIA_NIGHTLY_DIR}
    cd $JULIA_NIGHTLY_DIR
    echo "export JULIA=$CWD/$JULIA_NIGHTLY_DIR" > setup.source
    echo "export JULIA_CACHE_NAME=$JULIA_CACHE_NAME" >> setup.source
    echo "export JULIA_CACHE_PATH=$JULIA_CACHE_PATH" >> setup.source
    echo 'export JULIA_DEPOT_PATH=$HOME/$JULIA_CACHE_NAME:$JULIA_CACHE_PATH/' >> setup.source
    echo 'export PATH=$JULIA/bin:$PATH' >> setup.source
    echo 'export LD_LIBRARY_PATH=$JULIA/lib:$LD_LIBRARY_PATH' >> setup.source
    source setup.source
    cd ..
fi
echo "CWD: $CWD"
echo "JULIA_INSTALL_DIR: $JULIA_INSTALL_DIR"
mkdir -p $JULIA_CACHE_PATH
mkdir -p $JULIA_CACHE_DEBUG_PATH
if [ "x$SKIP_SETUP" = "x0" ]; then
    if [ "x$SKIP_JULIA" = "x0" ]; then
        mkdir -p $JULIA_INSTALL_DIR
        cd $JULIA_INSTALL_DIR
        CWD=`pwd`
        git clone https://github.com/JuliaLang/julia.git $JULIA_DIR
        cd $JULIA_DIR
        git checkout 3360a44 
        cd $CWD
    fi
fi
if [ "x$SKIP_SETUP" = "x0" ]; then
    if [ "x$SKIP_JULIA" = "x0" ]; then
        if [ "x$SKIP_DEBUG" = "x0" ]; then
            cp -r $JULIA_DIR $JULIA_DEBUG_DIR 
        fi
        cd $JULIA_DIR
        #ln -s $JULIA_INSTALL_DIR/AMDGPU.jl .
        #ln -s $JULIA_INSTALL_DIR/LLD_jll.jl .
        #ln -s $JULIA_INSTALL_DIR/GPUCompiler.jl .
        #ln -s $JULIA_INSTALL_DIR/CUDA.jl .
        #ln -s $JULIA_INSTALL_DIR/KernelAbstractions.jl .
        make -j
        echo "export JULIA=$JULIA_INSTALL_DIR/$JULIA_DIR/usr" > setup.source
        echo "export JULIA_DIR=$JULIA_DIR" > setup.source
        echo "export JULIA_CACHE_NAME=$JULIA_CACHE_NAME" >> setup.source
        echo "export JULIA_CACHE_PATH=$JULIA_CACHE_PATH" >> setup.source
        echo 'export JULIA_DEPOT_PATH=$HOME/$JULIA_CACHE_NAME:$JULIA_CACHE_PATH/' >> setup.source
        echo 'export PATH=$JULIA/bin:$PATH' >> setup.source
        echo 'export LD_LIBRARY_PATH=$JULIA/lib:$LD_LIBRARY_PATH' >> setup.source
        cd $CWD
    fi
    if [ "x$SKIP_DEBUG" = "x0" ]; then
        if [ "x$SKIP_JULIA" = "x0" ]; then
            cd $JULIA_DEBUG_DIR
            #ln -s $JULIA_INSTALL_DIR/AMDGPU.jl .
            #ln -s $JULIA_INSTALL_DIR/LLD_jll.jl .
            #ln -s $JULIA_INSTALL_DIR/GPUCompiler.jl .
            #ln -s $JULIA_INSTALL_DIR/CUDA.jl .
            #ln -s $JULIA_INSTALL_DIR/KernelAbstractions.jl .
            make cleanall
            make -j debug
			echo "export JULIA=$JULIA_INSTALL_DIR/$JULIA_DEBUG_DIR/usr" > setup.source
            echo "export JULIA_DEBUG_DIR=$JULIA_DEBUG_DIR" > setup.source
            echo "export JULIA_CACHE_NAME=$JULIA_CACHE_DEBUG_NAME" >> setup.source
            echo "export JULIA_CACHE_PATH=$JULIA_CACHE_DEBUG_PATH" >> setup.source
            echo 'export JULIA=$JULIA_INSTALL_DIR/$JULIA_DEBUG_DIR/usr' >> setup.source
            echo 'export JULIA_DEPOT_PATH=$HOME/$JULIA_CACHE_NAME:$JULIA_CACHE_PATH/' >> setup.source
            echo 'export PATH=$JULIA/bin:$PATH' >> setup.source
            echo 'export LD_LIBRARY_PATH=$JULIA/lib:$LD_LIBRARY_PATH' >> setup.source
            cd $CWD
        fi
    fi
fi
if [ "x$SKIP_SETUP" = "x0" ]; then
    if [ "x$SKIP_JULIA" = "x0" ]; then
        cd $JULIA_INSTALL_DIR
        . $JULIA_INSTALL_DIR/$JULIA_DIR/setup.source
        cd $CWD
    fi
fi
export JULIA_DEPOT_PATH=$JULIA_CACHE_PATH
if [ "x$julia_version" = "xjulia version 1.12.0-beta1" ]; then
GPUCompiler=${GPUCompiler:-master}
AMDGPURepository="JuliaGPU"
AMDGPU=${AMDGPU:-v1.2.7}
else
GPUCompiler=${GPUCompiler:-v1.3.2}
AMDGPURepository="miniskar"
AMDGPU=${AMDGPU:-dev}
fi
CUDA=${CUDA:-v5.7.1}
LoopVectorization=${LoopVectorization:-6aea6fb4eff}
KernelAbstractions=${KernelAbstractions:-v0.9.34}
#julia -e "using Pkg; Pkg.add(PackageSpec(name=\"LLD_jll\", version=\"19.1.7\"))"
julia -e "using Pkg; Pkg.add(PackageSpec(url=\"https://github.com/JuliaGPU/GPUCompiler.jl\", rev=\"$GPUCompiler\"))"
julia -e "using Pkg; Pkg.add(PackageSpec(url=\"https://github.com/JuliaGPU/CUDA.jl\", rev=\"$CUDA\"))"
julia -e "using Pkg; Pkg.add(PackageSpec(url=\"https://github.com/$AMDGPURepository/AMDGPU.jl\", rev=\"$AMDGPU\"))"
julia -e "using Pkg; Pkg.add(\"Requires\")"
julia -e "using Pkg; Pkg.add(\"Dagger\")"
julia -e "using Pkg; Pkg.add(\"StatsBase\")"
julia -e "using Pkg; Pkg.add(\"ProgressMeter\")"
julia -e "using Pkg; Pkg.add(\"Distributions\")"
julia -e "using Pkg; Pkg.add(PackageSpec(url=\"https://github.com/miniskar/JACC.jl\"))"
julia -e "using Pkg; Pkg.add(PackageSpec(url=\"https://github.com/miniskar/LoopVectorization.jl\", rev=\"$LoopVectorization\"))"
julia -e "using Pkg; Pkg.add(PackageSpec(url=\"https://github.com/JuliaGPU/KernelAbstractions.jl\", rev=\"$KernelAbstractions\"))"
#julia -e 'using Pkg; Pkg.add(PackageSpec(url="https://github.com/JuliaGPU/KernelAbstractions.jl", rev="f198f5c8ab1a60fa265279406ab458cfc29f8239"))"
#julia -e "using Pkg; Pkg.add(PackageSpec(path=\"./KernelAbstractions.jl\"))" 
if [ -e /usr/lib/x86_64-linux-gnu/libcuda.so ]; then
    for f in `find $JULIA_CACHE_PATH -name "libcuda.so"`; do 
        rm -f $f ; ln -s /usr/lib/x86_64-linux-gnu/libcuda.so $f ;
    done
fi
if [ "x$SKIP_SETUP" = "x0" ]; then
    if [ "x$SKIP_DEBUG" = "x0" ]; then
        if [ "x$SKIP_JULIA" = "x0" ]; then
            . $JULIA_INSTALL_DIR/$JULIA_DEBUG_DIR/setup.source
        fi
        export JULIA_DEPOT_PATH=$JULIA_CACHE_PATH.debug
        #julia-debug -e "using Pkg; Pkg.add(PackageSpec(name=\"LLD_jll\", version=\"19.1.7\"))"
        julia-debug -e "using Pkg; Pkg.add(PackageSpec(url=\"https://github.com/JuliaGPU/GPUCompiler.jl\", rev=\"$GPUCompiler\"))"
        julia-debug -e "using Pkg; Pkg.add(PackageSpec(url=\"https://github.com/JuliaGPU/CUDA.jl\", rev=\"$CUDA\"))"
        julia-debug -e "using Pkg; Pkg.add(PackageSpec(url=\"https://github.com/$AMDGPURepository/AMDGPU.jl\", rev=\"$AMDGPU\"))"
        julia-debug -e "using Pkg; Pkg.add(\"Requires\")"
        julia-debug -e "using Pkg; Pkg.add(\"Dagger\")"
        julia-debug -e "using Pkg; Pkg.add(\"StatsBase\")"
        julia-debug -e "using Pkg; Pkg.add(\"ProgressMeter\")"
        julia-debug -e "using Pkg; Pkg.add(\"Distributions\")"
        julia-debug -e "using Pkg; Pkg.add(PackageSpec(url=\"https://github.com/miniskar/JACC.jl\"))"
        julia-debug -e "using Pkg; Pkg.add(PackageSpec(url=\"https://github.com/miniskar/LoopVectorization.jl\", rev=\"$LoopVectorization\"))"
        julia-debug -e "using Pkg; Pkg.add(PackageSpec(url=\"https://github.com/JuliaGPU/KernelAbstractions.jl\", rev=\"$KernelAbstractions\"))"
        for f in `find $JULIA_CACHE_DEBUG_PATH -name "libcuda.so"`; do 
            rm -f $f ; ln -s /usr/lib/x86_64-linux-gnu/libcuda.so $f ;
        done
    fi
fi

