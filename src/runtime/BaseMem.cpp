#include "BaseMem.h"
#include "Device.h"

namespace iris {
namespace rt {
void BaseMem::HostRecordEvent(int devno, int stream) 
{
    Device *dev = archs_dev_[devno];
    HostLock();
    void **host_event = GetHostCompletionEventPtr(true);
    dev->CreateEvent(host_event, iris_event_disable_timing);
    d2h_events_[devno] = *host_event; 
    dev->RecordEvent(host_event, stream);
    //printf("host_event:%p host_event_ptr:%p d2h:%p\n", host_event, *host_event, d2h_events_[devno]);
    _event_debug("dev:[%d][%s] mem:%lu host_event:%p host_event_ptr:%p d2h:%p stream:%d", dev->devno(), dev->name(), uid(), *host_event, host_event, d2h_events_[devno], stream);
    SetHostWriteDevice(devno);
    SetHostWriteStream(stream);
    HostUnlock();
}
void *BaseMem::RecordEvent(int devno, int stream, bool new_entry) {
    Device *dev = archs_dev_[devno];
    void **event_ptr = GetCompletionEventPtr(devno, new_entry);
    if (*event_ptr == NULL) {
        dev->CreateEvent(event_ptr, iris_event_disable_timing);
    }
    else {
    _event_debug("Reusing dev:[%d][%s] mem:%lu event:%p event_ptr:%p stream:%d",  dev->devno(), dev->name(), uid(), *event_ptr, event_ptr, stream);
    }
    _trace(" devno:%d stream:%d uid:%lu event:%p", devno, stream, uid(), GetCompletionEvent(devno)); 
    _event_debug("dev:[%d][%s] mem:%lu event:%p event_ptr:%p stream:%d",  dev->devno(), dev->name(), uid(), *event_ptr, event_ptr, stream);
    dev->RecordEvent(event_ptr, stream);
    SetWriteDevice(devno);
    SetWriteStream(devno, stream);
    SetWriteDeviceEvent(devno, *event_ptr);
    return *event_ptr;
}
void BaseMem::HardHostWriteSynchronize(Device *dev, void *event) {
    dev->EventSynchronize(event);
    SetHostWriteStream(-1);
    SetHostWriteDevice(-1);
}
void BaseMem::HardDeviceWriteSynchronize(Device *dev, void *event) {
    dev->EventSynchronize(event);
    SetWriteStream(dev->devno(), -1);
    SetWriteDevice(-1);
}
void BaseMem::WaitForEvent(int devno, int stream, int dep_devno) {
    assert(GetCompletionEvent(devno) != NULL);
    Device *dev = archs_dev_[devno];
    dev->WaitForEvent(GetCompletionEvent(devno), stream, iris_event_wait_default);
}
void BaseMem::DestroyEvent(int devno, void *event) {
    Device *dev = archs_dev_[devno];
    dev->DestroyEvent(event);
}

}
}
