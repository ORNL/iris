TEST=test20_cublas

CFLAGS		+= -I ${MATH_LIBS}/include -I ${NVIDIA_PATH}/include
LDINC		+= -L ${MATH_LIBS}/lib64 	-L ${NVIDIA_PATH}/lib64
LDFLAGS	+= -lcublas -lcuda

include ../Makefile.tests

kernel.ptx: kernel.cu
	$(NVCC) -ptx $^

