#include "BaseMem.h"
#include "Device.h"

namespace iris {
namespace rt {
void BaseMem::HostRecordEvent(int devno, int stream) 
{
    Device *dev = archs_dev_[devno];
    HostLock();
    void **host_event = GetHostCompletionEventPtr(true);
    dev->ResetContext();
    dev->CreateEvent(host_event, iris_event_disable_timing);
    dev->RecordEvent(host_event, stream);
    d2h_events_[devno] = *host_event; 
    ASSERT(*host_event != NULL);
    //printf("host_event:%p host_event_ptr:%p d2h:%p\n", host_event, *host_event, d2h_events_[devno]);
    _event_debug("dev:[%d][%s] mem:%lu host_event:%p host_event_ptr:%p d2h:%p stream:%d", dev->devno(), dev->name(), uid(), *host_event, host_event, d2h_events_[devno], stream);
    SetHostWriteDevice(devno);
    SetHostWriteStream(stream);
    HostUnlock();
}
void *BaseMem::RecordEvent(int devno, int stream, bool new_entry) {
    Device *dev = archs_dev_[devno];
    void **event_ptr = GetCompletionEventPtr(devno, new_entry);
    _event_debug("Created mem:%lu devno:%d *event:%p event:%p", uid(), devno, *event_ptr, event_ptr);
    dev->ResetContext();
    if (*event_ptr == NULL) {
        dev->CreateEvent(event_ptr, iris_event_disable_timing);
    }
    else {
    _event_debug("Reusing dev:[%d][%s] mem:%lu event:%p event_ptr:%p stream:%d",  dev->devno(), dev->name(), uid(), *event_ptr, event_ptr, stream);
    }
    _trace(" devno:%d stream:%d uid:%lu event:%p", devno, stream, uid(), GetCompletionEvent(devno)); 
    _event_debug("dev:[%d][%s] mem:%lu event:%p event_ptr:%p stream:%d",  dev->devno(), dev->name(), uid(), *event_ptr, event_ptr, stream);
    dev->RecordEvent(event_ptr, stream);
    _event_debug("Recorded event mem:%lu devno:%d *event:%p event:%p", uid(), devno, *event_ptr, event_ptr);
    SetWriteDevice(devno);
    SetWriteStream(devno, stream);
    SetWriteDeviceEvent(devno, *event_ptr);
    return *event_ptr;
}
void BaseMem::HardHostWriteEventSynchronize(Device *dev, void *event) {
    dev->ResetContext();
    dev->EventSynchronize(event);
    SetHostWriteStream(-1);
    SetHostWriteDevice(-1);
}
void BaseMem::HardDeviceWriteEventSynchronize(Device *dev, void *event) {
    dev->ResetContext();
    dev->EventSynchronize(event);
    SetWriteStream(dev->devno(), -1);
    SetWriteDevice(-1);
}
void BaseMem::WaitForEvent(int devno, int stream, int dep_devno) {
    assert(GetCompletionEvent(devno) != NULL);
    Device *dev = archs_dev_[devno];
    dev->ResetContext();
    dev->WaitForEvent(GetCompletionEvent(devno), stream, iris_event_wait_default);
}
void BaseMem::DestroyEvent(int devno, void *event) {
    Device *dev = archs_dev_[devno];
    dev->ResetContext();
    dev->DestroyEvent(event);
}
pair<bool, int8_t> BaseMem::IsResetPossibleWithMemset() {
    return IsResetPossibleWithMemset(reset_data_);
}
pair<bool, int8_t> BaseMem::IsResetPossibleWithMemset(ResetData & reset_data) {
    if (reset_data.reset_type_ == iris_reset_memset) return make_pair<bool, int8_t>(true, 0); //It will read value from reset_data->value_.u8
    auto false_pair = std::make_pair(false, 0);
    if (reset_data.reset_type_ != iris_reset_assign) return false_pair;
    IRISValue *value = &reset_data.value_;
    int8_t vi8 = reset_data.value_.i8;
    auto v_pair = std::make_pair(true, vi8);
    switch(element_type_) {
        case iris_int8: 
        case iris_bool: 
        case iris_uint8: 
            if (value->i8 == 0 || value->i8 == -1) return v_pair; else break;
        case iris_int16: 
        case iris_uint16: 
            if (value->i16 == 0 || value->i16 == -1) return v_pair; else break;
        case iris_int32: 
        case iris_uint32: 
        case iris_float: 
            if (value->i32 == 0 || value->i32 == -1) return v_pair; else break;
        case iris_double: 
        case iris_int64: 
        case iris_uint64: 
            if (value->i64 == 0 || value->i64 == -1) return v_pair; else break;
        case iris_unknown: break;
    }
    return false_pair;
}

}
}
