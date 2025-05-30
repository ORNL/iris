#ifndef IRIS_SRC_RT_LOADER_OPENCL_H
#define IRIS_SRC_RT_LOADER_OPENCL_H

#include "Loader.h"
#define CL_TARGET_OPENCL_VERSION 220
#include <iris/CL/cl.h>

namespace iris {
namespace rt {

typedef void (*OpenCLCallBack)(cl_event, cl_int, void*);
class LoaderOpenCL : public Loader {
public:
  LoaderOpenCL();
  ~LoaderOpenCL();

  const char* library();
  int LoadFunctions();

CL_API_ENTRY cl_int CL_API_CALL
(*clReleaseEvent)(cl_event event) CL_API_SUFFIX__VERSION_1_0;
cl_int
(*clGetPlatformIDs)(cl_uint          num_entries,
                 cl_platform_id * platforms,
                 cl_uint *        num_platforms) CL_API_SUFFIX__VERSION_1_0;

CL_API_ENTRY cl_int CL_API_CALL
(*clEnqueueMarkerWithWaitList)(cl_command_queue  command_queue,
                            cl_uint           num_events_in_wait_list,
                            const cl_event *  event_wait_list,
                            cl_event *        event) CL_API_SUFFIX__VERSION_1_2;

CL_API_ENTRY cl_int CL_API_CALL
(*clGetPlatformInfo)(cl_platform_id   platform,
                  cl_platform_info param_name,
                  size_t           param_value_size,
                  void *           param_value,
                  size_t *         param_value_size_ret) CL_API_SUFFIX__VERSION_1_0;

CL_API_ENTRY cl_int CL_API_CALL
(*clGetEventProfilingInfo)(cl_event            event,
                        cl_profiling_info   param_name,
                        size_t              param_value_size,
                        void *              param_value,
                        size_t *            param_value_size_ret) CL_API_SUFFIX__VERSION_1_0;
CL_API_ENTRY cl_int CL_API_CALL
(*clGetDeviceIDs)(cl_platform_id   platform,
               cl_device_type   device_type,
               cl_uint          num_entries,
               cl_device_id *   devices,
               cl_uint *        num_devices) CL_API_SUFFIX__VERSION_1_0;

CL_API_ENTRY cl_int CL_API_CALL
(*clGetDeviceInfo)(cl_device_id    device,
                cl_device_info  param_name,
                size_t          param_value_size,
                void *          param_value,
                size_t *        param_value_size_ret) CL_API_SUFFIX__VERSION_1_0;

CL_API_ENTRY cl_context CL_API_CALL
(*clCreateContext)(const cl_context_properties * properties,
                cl_uint              num_devices,
                const cl_device_id * devices,
                void (CL_CALLBACK * pfn_notify)(const char * errinfo,
                                                const void * private_info,
                                                size_t       cb,
                                                void *       user_data),
                void *               user_data,
                cl_int *             errcode_ret) CL_API_SUFFIX__VERSION_1_0;

CL_API_ENTRY cl_int CL_API_CALL
    (*clWaitForEvents)(cl_uint             num_events,
            const cl_event *    event_list) CL_API_SUFFIX__VERSION_1_0;
CL_API_ENTRY cl_int CL_API_CALL (*clEnqueueMarker)(	cl_command_queue command_queue,
        cl_event *event);

CL_API_ENTRY cl_int CL_API_CALL (*clEnqueueWaitForEvents)(	cl_command_queue command_queue,
 	cl_uint num_events,
 	const cl_event *event_list);
CL_API_ENTRY cl_mem CL_API_CALL
(*clCreateBuffer)(cl_context   context,
               cl_mem_flags flags,
               size_t       size,
               void *       host_ptr,
               cl_int *     errcode_ret) CL_API_SUFFIX__VERSION_1_0;

CL_API_ENTRY cl_int CL_API_CALL
(*clReleaseMemObject)(cl_mem memobj) CL_API_SUFFIX__VERSION_1_0;

CL_API_ENTRY cl_int CL_API_CALL
(*clGetCommandQueueInfo)(cl_command_queue      command_queue,
                      cl_command_queue_info param_name,
                      size_t                param_value_size,
                      void *                param_value,
                      size_t *              param_value_size_ret) CL_API_SUFFIX__VERSION_1_0;

CL_API_ENTRY cl_program CL_API_CALL
(*clCreateProgramWithSource)(cl_context        context,
                          cl_uint           count,
                          const char **     strings,
                          const size_t *    lengths,
                          cl_int *          errcode_ret) CL_API_SUFFIX__VERSION_1_0;

CL_API_ENTRY cl_program CL_API_CALL
(*clCreateProgramWithBinary)(cl_context                     context,
                          cl_uint                        num_devices,
                          const cl_device_id *           device_list,
                          const size_t *                 lengths,
                          const unsigned char **         binaries,
                          cl_int *                       binary_status,
                          cl_int *                       errcode_ret) CL_API_SUFFIX__VERSION_1_0;

CL_API_ENTRY cl_program CL_API_CALL
(*clCreateProgramWithIL)(cl_context    context,
                     const void*    il,
                     size_t         length,
                     cl_int*        errcode_ret) CL_API_SUFFIX__VERSION_2_1;

CL_API_ENTRY cl_int CL_API_CALL
(*clReleaseProgram)(cl_program program) CL_API_SUFFIX__VERSION_1_0;

CL_API_ENTRY cl_int CL_API_CALL
(*clBuildProgram)(cl_program           program,
               cl_uint              num_devices,
               const cl_device_id * device_list,
               const char *         options,
               void (CL_CALLBACK *  pfn_notify)(cl_program program,
                                                void * user_data),
               void *               user_data) CL_API_SUFFIX__VERSION_1_0;

CL_API_ENTRY cl_int CL_API_CALL
(*clGetProgramInfo)(cl_program         program,
                 cl_program_info    param_name,
                 size_t             param_value_size,
                 void *             param_value,
                 size_t *           param_value_size_ret) CL_API_SUFFIX__VERSION_1_0;

CL_API_ENTRY cl_int CL_API_CALL
(*clGetProgramBuildInfo)(cl_program            program,
                      cl_device_id          device,
                      cl_program_build_info param_name,
                      size_t                param_value_size,
                      void *                param_value,
                      size_t *              param_value_size_ret) CL_API_SUFFIX__VERSION_1_0;

CL_API_ENTRY cl_kernel CL_API_CALL
(*clCreateKernel)(cl_program      program,
               const char *    kernel_name,
               cl_int *        errcode_ret) CL_API_SUFFIX__VERSION_1_0;

CL_API_ENTRY cl_int CL_API_CALL
(*clSetKernelArg)(cl_kernel    kernel,
               cl_uint      arg_index,
               size_t       arg_size,
               const void * arg_value) CL_API_SUFFIX__VERSION_1_0;

CL_API_ENTRY cl_int CL_API_CALL
(*clFinish)(cl_command_queue command_queue) CL_API_SUFFIX__VERSION_1_0;

CL_API_ENTRY cl_int CL_API_CALL
(*clEnqueueReadBuffer)(cl_command_queue    command_queue,
                    cl_mem              buffer,
                    cl_bool             blocking_read,
                    size_t              offset,
                    size_t              size,
                    void *              ptr,
                    cl_uint             num_events_in_wait_list,
                    const cl_event *    event_wait_list,
                    cl_event *          event) CL_API_SUFFIX__VERSION_1_0;

CL_API_ENTRY cl_int CL_API_CALL
(*clEnqueueReadBufferRect)(cl_command_queue command_queue,
                               cl_mem buffer,
                               cl_bool blocking_read,
                               const size_t * buffer_origin,
                               const size_t * host_origin,
                               const size_t * region,
                               size_t buffer_row_pitch,
                               size_t buffer_slice_pitch,
                               size_t host_row_pitch,
                               size_t host_slice_pitch,
                               void *ptr,
                               cl_uint num_events_in_wait_list,
                               const cl_event *event_wait_list,
                               cl_event *event) CL_API_SUFFIX__VERSION_1_0;
CL_API_ENTRY cl_int CL_API_CALL
(*clEnqueueWriteBuffer)(cl_command_queue   command_queue,
                     cl_mem             buffer,
                     cl_bool            blocking_write,
                     size_t             offset,
                     size_t             size,
                     const void *       ptr,
                     cl_uint            num_events_in_wait_list,
                     const cl_event *   event_wait_list,
                     cl_event *         event) CL_API_SUFFIX__VERSION_1_0;

CL_API_ENTRY cl_int CL_API_CALL
(*clEnqueueWriteBufferRect)(cl_command_queue command_queue,
                                cl_mem buffer,
                                cl_bool blocking_write,
                                const size_t * buffer_origin,
                                const size_t * host_origin,
                                const size_t *region,
                                size_t buffer_row_pitch,
                                size_t buffer_slice_pitch,
                                size_t host_row_pitch,
                                size_t host_slice_pitch,
                                void *ptr,
                                cl_uint num_events_in_wait_list,
                                const cl_event *event_wait_list,
                                cl_event *event) CL_API_SUFFIX__VERSION_1_0;

CL_API_ENTRY cl_int CL_API_CALL
(*clEnqueueNDRangeKernel)(cl_command_queue command_queue,
                       cl_kernel        kernel,
                       cl_uint          work_dim,
                       const size_t *   global_work_offset,
                       const size_t *   global_work_size,
                       const size_t *   local_work_size,
                       cl_uint          num_events_in_wait_list,
                       const cl_event * event_wait_list,
                       cl_event *       event) CL_API_SUFFIX__VERSION_1_0;

typedef cl_bitfield         cl_command_queue_properties;
CL_API_ENTRY CL_EXT_PREFIX__VERSION_1_2_DEPRECATED cl_command_queue CL_API_CALL
(*clCreateCommandQueue)(cl_context                     context,
                     cl_device_id                   device,
                     cl_command_queue_properties    properties,
                     cl_int *                       errcode_ret) CL_EXT_SUFFIX__VERSION_1_2_DEPRECATED;
#ifdef CL_VERSION_2_0
typedef cl_properties       cl_queue_properties;
CL_API_ENTRY cl_command_queue CL_API_CALL
(*clCreateCommandQueueWithProperties)(cl_context               context,
                                   cl_device_id             device,
                                   const cl_queue_properties *    properties,
                                   cl_int *                 errcode_ret) CL_API_SUFFIX__VERSION_2_0;
#else
CL_API_ENTRY CL_EXT_PREFIX__VERSION_1_2_DEPRECATED cl_command_queue CL_API_CALL
(*clCreateCommandQueue)(cl_context                     context,
                     cl_device_id                   device,
                     cl_command_queue_properties    properties,
                     cl_int *                       errcode_ret) CL_EXT_SUFFIX__VERSION_1_2_DEPRECATED;
#endif
CL_API_ENTRY cl_int CL_API_CALL
(*clReleaseContext)(cl_context context) CL_API_SUFFIX__VERSION_1_0;
CL_API_ENTRY cl_int CL_API_CALL
(*clReleaseCommandQueue)(cl_command_queue command_queue) CL_API_SUFFIX__VERSION_1_0;

CL_API_ENTRY cl_int CL_API_CALL
(*clSetEventCallback)(cl_event    event,
                   cl_int      command_exec_callback_type,
                   void (CL_CALLBACK * pfn_notify)(cl_event event,
                                                   cl_int   event_command_status,
                                                   void *   user_data),
                   void *      user_data) CL_API_SUFFIX__VERSION_1_1;
CL_API_ENTRY cl_int CL_API_CALL
    (*clEnqueueFillBuffer)(cl_command_queue   command_queue,
            cl_mem             buffer,
            const void *       pattern,
            size_t             pattern_size,
            size_t             offset,
            size_t             size,
            cl_uint            num_events_in_wait_list,
            const cl_event *   event_wait_list,
            cl_event *         event) CL_API_SUFFIX__VERSION_1_2;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_LOADER_OPENCL_H */

