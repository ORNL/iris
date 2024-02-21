#include "BaseMem.h"
#include "Device.h"

namespace iris {
namespace rt {
void BaseMem::HostRecordEvent(int devno, int stream) 
{
    HostLock();
    if (d2h_events_[devno] == NULL) {
        Device *dev = archs_dev_[devno];
        dev->CreateEvent(&d2h_events_[devno], iris_event_default);
        archs_dev_[devno]->RecordEvent(&d2h_events_[devno], stream);
    }
    void **host_event = GetHostCompletionEventPtr();
    //printf("host_event:%p host_event_ptr:%p d2h:%p\n", host_event, *host_event, d2h_events_[devno]);
    if (*host_event != d2h_events_[devno]) {
        _debug2("Recording event mem:%lu dev:%d stream:%d event:%p\n", uid(), devno, stream, d2h_events_[devno]);
        *host_event = d2h_events_[devno];
    }
    HostUnlock();
}
void BaseMem::RecordEvent(int devno, int stream) {
    if (GetCompletionEvent(devno) == NULL) {
        Device *dev = archs_dev_[devno];
        dev->CreateEvent(GetCompletionEventPtr(devno), iris_event_disable_timing);
    }
    _trace(" devno:%d stream:%d uid:%lu event:%p\n", devno, stream, uid(), GetCompletionEvent(devno)); 
    archs_dev_[devno]->RecordEvent(GetCompletionEventPtr(devno), stream);
}
void BaseMem::WaitForEvent(int devno, int stream, int dep_devno) {
    assert(GetCompletionEvent(devno) != NULL);
    Device *dev = archs_dev_[devno];
    dev->WaitForEvent(GetCompletionEvent(devno), stream, iris_event_disable_timing);
}
void BaseMem::DestroyEvent(int devno, void *event) {
    Device *dev = archs_dev_[devno];
    dev->DestroyEvent(event);
}

}
}
