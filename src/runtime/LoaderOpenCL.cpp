#include "LoaderOpenCL.h"
#include "Debug.h"
#include "Utils.h"

namespace iris {
namespace rt {

LoaderOpenCL::LoaderOpenCL() {
}

LoaderOpenCL::~LoaderOpenCL() {
}

const char* LoaderOpenCL::library() {
  std::string* return_string = new std::string();
  bool proper_file_path = false;
  const char* icd_vendors = getenv("OCL_ICD_VENDORS");
  const char* icd_preferred_path = getenv("OPENCL_VENDOR_PATH");

  if(icd_vendors){
    bool empty_vendor_lib = !strcmp(icd_vendors, "");
    if (empty_vendor_lib) icd_vendors = nullptr;
  }
  if(icd_preferred_path){
    bool empty_path = !strcmp(icd_preferred_path, "");
    if (empty_path) icd_preferred_path = nullptr;
    std::string buf(icd_preferred_path);
    std::string ending = "/";
    proper_file_path = std::equal(ending.rbegin(), ending.rend(), buf.rbegin());
  }

  if (icd_vendors && icd_preferred_path){
    if (proper_file_path) *return_string = std::string(icd_preferred_path) + std::string(icd_vendors);
    else *return_string = std::string(icd_preferred_path) + "/" + std::string(icd_vendors);
  }
  else if (icd_vendors) *return_string = icd_vendors;
  else if (icd_preferred_path){
    if(proper_file_path) *return_string = std::string(icd_preferred_path) + "libOpenCL.so";
    else *return_string = std::string(icd_preferred_path) + "/" + "libOpenCL.so";
  }
  else *return_string = "libOpenCL.so";
  //printf("using path = %s\n",return_string->c_str());
  return return_string->c_str();
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

