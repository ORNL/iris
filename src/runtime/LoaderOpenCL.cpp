#include "LoaderOpenCL.h"
#include "Debug.h"

namespace brisbane {
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
  LOADFUNC(clEnqueueNDRangeKernel);
  LOADFUNC(clCreateCommandQueue);
  LOADFUNC(clSetEventCallback);
  return BRISBANE_OK;
}

} /* namespace rt */
} /* namespace brisbane */

