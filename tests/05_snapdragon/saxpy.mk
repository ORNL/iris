IRIS=$(HOME)/work/iris-rts

LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
LOCAL_MODULE := libOpenCL
LOCAL_SRC_FILES := ../libOpenCL.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libiris
LOCAL_SRC_FILES:= ../obj/local/armeabi-v7a/libiris.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE:=saxpy
LOCAL_STATIC_LIBRARIES := libiris
LOCAL_SHARED_LIBRARIES := libOpenCL
LOCAL_C_INCLUDES:=$(IRIS)/include
LOCAL_SRC_FILES:= $(IRIS)/apps/saxpy/saxpy-iris.cpp
LOCAL_LDFLAGS:=-fopenmp
include $(BUILD_EXECUTABLE)

