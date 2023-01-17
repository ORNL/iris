IRIS=$(HOME)/work/iris-rts

LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
LOCAL_MODULE:=kernel.omp.so
LOCAL_MODULE_FILENAME:=kernel.omp
LOCAL_C_INCLUDES:=$(IRIS)/include
LOCAL_SRC_FILES:= $(IRIS)/apps/saxpy/kernel.omp.c
LOCAL_LDFLAGS:=-fopenmp
include $(BUILD_SHARED_LIBRARY)

