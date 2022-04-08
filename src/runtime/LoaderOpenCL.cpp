#include "LoaderOpenCL.h"
#include "Debug.h"

namespace iris {
namespace rt {

LoaderOpenCL::LoaderOpenCL() {
}

LoaderOpenCL::~LoaderOpenCL() {
}

int LoaderOpenCL::LoadFunctions() {
  LOADFUNC(clGetPlatformIDs);
  LOADFUNC(clGetPlatformInfo);
  LOADFUNC(clGetDeviceIDs);
  LOADFUNC(clGetDeviceInfo);
  LOADFUNC(clCreateContext);
  LOADFUNC(clCreateBuffer);
  LOADFUNC(clReleaseMemObject);
  LOADFUNC(clCreateProgramWithSource);
  LOADFUNC(clCreateProgramWithBinary);
  LOADFUNCSILENT(clCreateProgramWithIL);
  LOADFUNC(clReleaseProgram);
  LOADFUNC(clBuildProgram);
  LOADFUNC(clGetProgramInfo);
  LOADFUNC(clGetProgramBuildInfo);
  LOADFUNC(clCreateKernel);
  LOADFUNC(clSetKernelArg);
  LOADFUNC(clFinish);
  LOADFUNC(clEnqueueReadBuffer);
  LOADFUNC(clEnqueueWriteBuffer);
  LOADFUNC(clEnqueueReadBufferRect);
  LOADFUNC(clEnqueueWriteBufferRect);
  LOADFUNC(clEnqueueNDRangeKernel);
  LOADFUNC(clCreateCommandQueue);
  LOADFUNC(clSetEventCallback);
  return IRIS_SUCCESS;
}

} /* namespace rt */
} /* namespace iris */

