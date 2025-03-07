#include "LoaderOpenCL.h"
#include "Debug.h"
#include "Utils.h"
#include <fstream>

namespace iris {
namespace rt {

LoaderOpenCL::LoaderOpenCL() {
}

LoaderOpenCL::~LoaderOpenCL() {
}

const char* LoaderOpenCL::library() {
  std::string* return_string = new std::string();
  const char* icd_preferred_path = getenv("OPENCL_VENDOR_PATH");//OPENCL_VENDOR_PATH=/my/local/icd/file/path/libOpenCL.so
  if(icd_preferred_path){
    ifstream f(icd_preferred_path);
    if (f.good()) return(icd_preferred_path);
  }
  //else use OpenCL ICD Loader logic
  const char* icd_vendors = getenv("OCL_ICD_VENDORS"); //OCL_ICD_VENDORS=/my/local/icd/search/path
  const char* icd_filenames = getenv("OCL_ICD_FILENAMES");//OCL_ICD_FILENAMES=libVendorA.so:libVendorB.so

  if(icd_vendors){
    bool empty_vendor_lib = (0 == strcmp(icd_vendors, ""));
    if (empty_vendor_lib) icd_vendors = nullptr;
  }
  if(icd_filenames){
    bool empty_vendor_filename = (0 == strcmp(icd_filenames, ""));
    if (empty_vendor_filename) icd_filenames = nullptr;
  }
  if (icd_vendors && icd_filenames){
    *return_string = std::string(icd_vendors) + "/" + std::string(icd_filenames);
  }
  else if (icd_filenames){ *return_string = icd_filenames;}
  else if (icd_vendors){
    *return_string = std::string(icd_vendors) + "/" + "libOpenCL.so";
  }
  else {*return_string = "libOpenCL.so";}
  //printf("using : %s\n",return_string->c_str());

  //clear the use of the ICD loader at other levels
  unsetenv("OCL_ICD_VENDORS");
  unsetenv("OCL_ICD_FILENAMES");
  //if it's a good value force all the OpenCL worker devices to use the same backend
  if(strcmp(return_string->c_str(),"libOpenCL.so") != 0) setenv("OPENCL_VENDOR_PATH",return_string->c_str(),1);
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
  LOADFUNC(clGetCommandQueueInfo);
  LOADFUNC(clCreateProgramWithBinary);
  LOADFUNCSILENT(clCreateProgramWithIL);
  LOADFUNC(clReleaseProgram);
  LOADFUNC(clReleaseEvent);
  LOADFUNC(clBuildProgram);
  LOADFUNC(clEnqueueMarkerWithWaitList);
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
#ifdef CL_VERSION_2_0
  LOADFUNC(clCreateCommandQueueWithProperties);
#else
  LOADFUNC(clCreateCommandQueue);
#endif
  LOADFUNC(clEnqueueMarker);
  LOADFUNC(clEnqueueWaitForEvents);
  LOADFUNC(clWaitForEvents);
  LOADFUNC(clReleaseCommandQueue);
  LOADFUNC(clReleaseContext);
  LOADFUNC(clSetEventCallback);
  LOADFUNC(clEnqueueFillBuffer);
  LOADFUNC(clGetEventProfilingInfo);
  return IRIS_SUCCESS;
}

} /* namespace rt */
} /* namespace iris */

