#include "BaseMem.h"
#include "Device.h"

namespace iris {
namespace rt {
void BaseMem::RecordEvent(int devno, int stream) {
    if (GetCompletionEvent(devno) == NULL) {
        Device *dev = archs_dev_[devno];
        dev->CreateEvent(GetCompletionEventPtr(devno), iris_event_disable_timing);
    }
    _trace(" devno:%d stream:%d uid:%lu event:%p\n", devno, stream, uid(), GetCompletionEvent(devno)); 
    archs_dev_[devno]->RecordEvent(GetCompletionEvent(devno), stream);
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
