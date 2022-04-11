#ifndef IRIS_SRC_RT_LOADER_LEVEL_ZERO_H
#define IRIS_SRC_RT_LOADER_LEVEL_ZERO_H

#include "Loader.h"
#include <iris/level_zero/ze_api.h>
#include <pthread.h>

namespace iris {
namespace rt {

class LoaderLevelZero: public Loader {
public:
  LoaderLevelZero();
  ~LoaderLevelZero();

  const char* library() { return "libze_loader.so"; }
  int LoadFunctions();

ZE_APIEXPORT ze_result_t ZE_APICALL
(*zeInit)(
    ze_init_flags_t flags                           ///< [in] initialization flags.
                                                    ///< must be 0 (default) or a combination of ::ze_init_flag_t.
    );

ZE_APIEXPORT ze_result_t ZE_APICALL
(*zeDriverGet)(
    uint32_t* pCount,                               ///< [in,out] pointer to the number of driver instances.
                                                    ///< if count is zero, then the loader shall update the value with the
                                                    ///< total number of drivers available.
                                                    ///< if count is greater than the number of drivers available, then the
                                                    ///< loader shall update the value with the correct number of drivers available.
    ze_driver_handle_t* phDrivers                   ///< [in,out][optional][range(0, *pCount)] array of driver instance handles.
                                                    ///< if count is less than the number of drivers available, then the loader
                                                    ///< shall only retrieve that number of drivers.
    );

ZE_APIEXPORT ze_result_t ZE_APICALL
(*zeDeviceGet)(
    ze_driver_handle_t hDriver,                     ///< [in] handle of the driver instance
    uint32_t* pCount,                               ///< [in,out] pointer to the number of devices.
                                                    ///< if count is zero, then the driver shall update the value with the
                                                    ///< total number of devices available.
                                                    ///< if count is greater than the number of devices available, then the
                                                    ///< driver shall update the value with the correct number of devices available.
    ze_device_handle_t* phDevices                   ///< [in,out][optional][range(0, *pCount)] array of handle of devices.
                                                    ///< if count is less than the number of devices available, then driver
                                                    ///< shall only retrieve that number of devices.
    );

ZE_APIEXPORT ze_result_t ZE_APICALL
(*zeDeviceGetProperties)(
    ze_device_handle_t hDevice,                     ///< [in] handle of the device
    ze_device_properties_t* pDeviceProperties       ///< [in,out] query result for device properties
    );

ZE_APIEXPORT ze_result_t ZE_APICALL
(*zeContextCreate)(
    ze_driver_handle_t hDriver,                     ///< [in] handle of the driver object
    const ze_context_desc_t* desc,                  ///< [in] pointer to context descriptor
    ze_context_handle_t* phContext                  ///< [out] pointer to handle of context object created
    );

ZE_APIEXPORT ze_result_t ZE_APICALL
(*zeContextDestroy)(
    ze_context_handle_t hContext                    ///< [in][release] handle of context object to destroy
    );

ZE_APIEXPORT ze_result_t ZE_APICALL
(*zeMemAllocDevice)(
    ze_context_handle_t hContext,                   ///< [in] handle of the context object
    const ze_device_mem_alloc_desc_t* device_desc,  ///< [in] pointer to device memory allocation descriptor
    size_t size,                                    ///< [in] size in bytes to allocate; must be less-than
                                                    ///< ::ze_device_properties_t.maxMemAllocSize.
    size_t alignment,                               ///< [in] minimum alignment in bytes for the allocation; must be a power of
                                                    ///< two.
    ze_device_handle_t hDevice,                     ///< [in] handle of the device
    void** pptr                                     ///< [out] pointer to device allocation
    );

ZE_APIEXPORT ze_result_t ZE_APICALL
(*zeMemFree)(
    ze_context_handle_t hContext,                   ///< [in] handle of the context object
    void* ptr                                       ///< [in][release] pointer to memory to free
    );

ZE_APIEXPORT ze_result_t ZE_APICALL
(*zeCommandQueueCreate)(
    ze_context_handle_t hContext,                   ///< [in] handle of the context object
    ze_device_handle_t hDevice,                     ///< [in] handle of the device object
    const ze_command_queue_desc_t* desc,            ///< [in] pointer to command queue descriptor
    ze_command_queue_handle_t* phCommandQueue       ///< [out] pointer to handle of command queue object created
    );

ZE_APIEXPORT ze_result_t ZE_APICALL
(*zeCommandListCreate)(
    ze_context_handle_t hContext,                   ///< [in] handle of the context object
    ze_device_handle_t hDevice,                     ///< [in] handle of the device object
    const ze_command_list_desc_t* desc,             ///< [in] pointer to command list descriptor
    ze_command_list_handle_t* phCommandList         ///< [out] pointer to handle of command list object created
    );

ZE_APIEXPORT ze_result_t ZE_APICALL
(*zeCommandListCreateImmediate)(
    ze_context_handle_t hContext,                   ///< [in] handle of the context object
    ze_device_handle_t hDevice,                     ///< [in] handle of the device object
    const ze_command_queue_desc_t* altdesc,         ///< [in] pointer to command queue descriptor
    ze_command_list_handle_t* phCommandList         ///< [out] pointer to handle of command list object created
    );

ZE_APIEXPORT ze_result_t ZE_APICALL
(*zeCommandListReset)(
    ze_command_list_handle_t hCommandList           ///< [in] handle of command list object to reset
    );

ZE_APIEXPORT ze_result_t ZE_APICALL
(*zeCommandListDestroy)(
    ze_command_list_handle_t hCommandList           ///< [in][release] handle of command list object to destroy
    );

ZE_APIEXPORT ze_result_t ZE_APICALL
(*zeCommandListAppendMemoryCopy)(
    ze_command_list_handle_t hCommandList,          ///< [in] handle of command list
    void* dstptr,                                   ///< [in] pointer to destination memory to copy to
    const void* srcptr,                             ///< [in] pointer to source memory to copy from
    size_t size,                                    ///< [in] size in bytes to copy
    ze_event_handle_t hSignalEvent,                 ///< [in][optional] handle of the event to signal on completion
    uint32_t numWaitEvents,                         ///< [in][optional] number of events to wait on before launching; must be 0
                                                    ///< if `nullptr == phWaitEvents`
    ze_event_handle_t* phWaitEvents                 ///< [in][optional][range(0, numWaitEvents)] handle of the events to wait
                                                    ///< on before launching
    );

ZE_APIEXPORT ze_result_t ZE_APICALL
(*zeCommandListAppendLaunchKernel)(
    ze_command_list_handle_t hCommandList,          ///< [in] handle of the command list
    ze_kernel_handle_t hKernel,                     ///< [in] handle of the kernel object
    const ze_group_count_t* pLaunchFuncArgs,        ///< [in] thread group launch arguments
    ze_event_handle_t hSignalEvent,                 ///< [in][optional] handle of the event to signal on completion
    uint32_t numWaitEvents,                         ///< [in][optional] number of events to wait on before launching; must be 0
                                                    ///< if `nullptr == phWaitEvents`
    ze_event_handle_t* phWaitEvents                 ///< [in][optional][range(0, numWaitEvents)] handle of the events to wait
                                                    ///< on before launching
    );

ZE_APIEXPORT ze_result_t ZE_APICALL
(*zeCommandListAppendSignalEvent)(
    ze_command_list_handle_t hCommandList,          ///< [in] handle of the command list
    ze_event_handle_t hEvent                        ///< [in] handle of the event
    );

ZE_APIEXPORT ze_result_t ZE_APICALL
(*zeCommandQueueExecuteCommandLists)(
    ze_command_queue_handle_t hCommandQueue,        ///< [in] handle of the command queue
    uint32_t numCommandLists,                       ///< [in] number of command lists to execute
    ze_command_list_handle_t* phCommandLists,       ///< [in][range(0, numCommandLists)] list of handles of the command lists
                                                    ///< to execute
    ze_fence_handle_t hFence                        ///< [in][optional] handle of the fence to signal on completion
    );

ZE_APIEXPORT ze_result_t ZE_APICALL
(*zeCommandQueueSynchronize)(
    ze_command_queue_handle_t hCommandQueue,        ///< [in] handle of the command queue
    uint64_t timeout                                ///< [in] if non-zero, then indicates the maximum time (in nanoseconds) to
                                                    ///< yield before returning ::ZE_RESULT_SUCCESS or ::ZE_RESULT_NOT_READY;
                                                    ///< if zero, then immediately returns the status of the command queue;
                                                    ///< if UINT64_MAX, then function will not return until complete or device
                                                    ///< is lost.
                                                    ///< Due to external dependencies, timeout may be rounded to the closest
                                                    ///< value allowed by the accuracy of those dependencies.
    );

ZE_APIEXPORT ze_result_t ZE_APICALL
(*zeModuleCreate)(
    ze_context_handle_t hContext,                   ///< [in] handle of the context object
    ze_device_handle_t hDevice,                     ///< [in] handle of the device
    const ze_module_desc_t* desc,                   ///< [in] pointer to module descriptor
    ze_module_handle_t* phModule,                   ///< [out] pointer to handle of module object created
    ze_module_build_log_handle_t* phBuildLog        ///< [out][optional] pointer to handle of module's build log.
    );

ZE_APIEXPORT ze_result_t ZE_APICALL
(*zeKernelCreate)(
    ze_module_handle_t hModule,                     ///< [in] handle of the module
    const ze_kernel_desc_t* desc,                   ///< [in] pointer to kernel descriptor
    ze_kernel_handle_t* phKernel                    ///< [out] handle of the Function object
    );

ZE_APIEXPORT ze_result_t ZE_APICALL
(*zeKernelSetArgumentValue)(
    ze_kernel_handle_t hKernel,                     ///< [in] handle of the kernel object
    uint32_t argIndex,                              ///< [in] argument index in range [0, num args - 1]
    size_t argSize,                                 ///< [in] size of argument type
    const void* pArgValue                           ///< [in][optional] argument value represented as matching arg type. If
                                                    ///< null then argument value is considered null.
    );

ZE_APIEXPORT ze_result_t ZE_APICALL
(*zeFenceCreate)(
    ze_command_queue_handle_t hCommandQueue,        ///< [in] handle of command queue
    const ze_fence_desc_t* desc,                    ///< [in] pointer to fence descriptor
    ze_fence_handle_t* phFence                      ///< [out] pointer to handle of fence object created
    );

ZE_APIEXPORT ze_result_t ZE_APICALL
(*zeFenceDestroy)(
    ze_fence_handle_t hFence                        ///< [in][release] handle of fence object to destroy
    );

ZE_APIEXPORT ze_result_t ZE_APICALL
(*zeEventPoolCreate)(
    ze_context_handle_t hContext,                   ///< [in] handle of the context object
    const ze_event_pool_desc_t* desc,               ///< [in] pointer to event pool descriptor
    uint32_t numDevices,                            ///< [in][optional] number of device handles; must be 0 if `nullptr ==
                                                    ///< phDevices`
    ze_device_handle_t* phDevices,                  ///< [in][optional][range(0, numDevices)] array of device handles which
                                                    ///< have visibility to the event pool.
                                                    ///< if nullptr, then event pool is visible to all devices supported by the
                                                    ///< driver instance.
    ze_event_pool_handle_t* phEventPool             ///< [out] pointer handle of event pool object created
    );

ZE_APIEXPORT ze_result_t ZE_APICALL
(*zeEventCreate)(
    ze_event_pool_handle_t hEventPool,              ///< [in] handle of the event pool
    const ze_event_desc_t* desc,                    ///< [in] pointer to event descriptor
    ze_event_handle_t* phEvent                      ///< [out] pointer to handle of event object created
    );

ZE_APIEXPORT ze_result_t ZE_APICALL
(*zeEventDestroy)(
    ze_event_handle_t hEvent                        ///< [in][release] handle of event object to destroy
    );

ZE_APIEXPORT ze_result_t ZE_APICALL
(*zeEventHostSynchronize)(
    ze_event_handle_t hEvent,                       ///< [in] handle of the event
    uint64_t timeout                                ///< [in] if non-zero, then indicates the maximum time (in nanoseconds) to
                                                    ///< yield before returning ::ZE_RESULT_SUCCESS or ::ZE_RESULT_NOT_READY;
                                                    ///< if zero, then operates exactly like ::zeEventQueryStatus;
                                                    ///< if UINT64_MAX, then function will not return until complete or device
                                                    ///< is lost.
                                                    ///< Due to external dependencies, timeout may be rounded to the closest
                                                    ///< value allowed by the accuracy of those dependencies.
    );

  void Lock();
  void Unlock();

private:
  pthread_mutex_t mutex_;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_LOADER_LEVEL_ZERO_H */

