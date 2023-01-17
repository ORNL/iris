IRIS=$(HOME)/work/iris-rts
BRT=$(IRIS)/src/runtime

IRIS_RT_FILES:= \
$(BRT)/CAPI.cpp \
$(BRT)/Command.cpp \
$(BRT)/Consistency.cpp \
$(BRT)/Device.cpp \
$(BRT)/DeviceOpenCL.cpp \
$(BRT)/DeviceOpenMP.cpp \
$(BRT)/FilterTaskSplit.cpp \
$(BRT)/History.cpp \
$(BRT)/HubClient.cpp \
$(BRT)/Hub.cpp \
$(BRT)/Kernel.cpp \
$(BRT)/Mem.cpp \
$(BRT)/MemRange.cpp \
$(BRT)/Message.cpp \
$(BRT)/Platform.cpp \
$(BRT)/Policies.cpp \
$(BRT)/PolicyAll.cpp \
$(BRT)/PolicyAny.cpp \
$(BRT)/Policy.cpp \
$(BRT)/PolicyData.cpp \
$(BRT)/PolicyDefault.cpp \
$(BRT)/PolicyDevice.cpp \
$(BRT)/PolicyProfile.cpp \
$(BRT)/PolicyRandom.cpp \
$(BRT)/Polyhedral.cpp \
$(BRT)/Profiler.cpp \
$(BRT)/ProfilerDOT.cpp \
$(BRT)/ProfilerGoogleCharts.cpp \
$(BRT)/Reduction.cpp \
$(BRT)/Retainable.cpp \
$(BRT)/Scheduler.cpp \
$(BRT)/Task.cpp \
$(BRT)/TaskQueue.cpp \
$(BRT)/Thread.cpp \
$(BRT)/Timer.cpp \
$(BRT)/Utils.cpp \
$(BRT)/Worker.cpp

LOCAL_PATH:= $(call my-dir)

include $(CLEAR_VARS)
LOCAL_MODULE := libiris
LOCAL_C_INCLUDES:=$(IRIS)/include -I$(BRT)
LOCAL_CFLAGS:=-DUSE_OPENCL -DUSE_OPENMP
LOCAL_SRC_FILES:= $(IRIS_RT_FILES)
LOCAL_MODULE_FILENAME:=libiris
include $(BUILD_STATIC_LIBRARY)

