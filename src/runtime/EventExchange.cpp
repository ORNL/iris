#include "EventExchange.h"
#include "Device.h"

namespace iris
{
    namespace rt {
        void DestCentricGPUEventExchange::Wait()
        {
            _debug2("Waiting for fire:%d mem:%lu src_mem_stream:%d src_dev:%d dev:%d mem_stream:%d object:%p", fired, uid_, src_mem_stream_, src_devno_, devno_, dest_mem_stream_, this);
            dest_dev_->ResetContext();
            dest_dev_->WaitForEvent(dest_event_, dest_mem_stream_, iris_event_wait_default);
            _debug2("Completed Waiting for fired:%d mem:%lu src_mem_stream:%d src_dev:%d dev:%d mem_stream:%d object:%p", fired, uid_, src_mem_stream_, src_devno_, devno_, dest_mem_stream_, this);
        }
        void DestCentricGPUEventExchange::Fire()
        {
            fired = true;
            _debug2("Broadcast fire:%d mem:%lu src_mem_stream:%d src_dev:%d dev:%d mem_stream:%d object:%p", fired, uid_, src_mem_stream_, src_devno_, devno_, dest_mem_stream_, this);
            dest_dev_->ResetContext();
            dest_dev_->RecordEvent(&dest_event_, 0);
            _debug2("Broadcast completed fire:%d mem:%lu src_mem_stream:%d src_dev:%d dev:%d mem_stream:%d object:%p", fired, uid_, src_mem_stream_, src_devno_, devno_, dest_mem_stream_, this);
        }
        void SrcCentricGPUEventExchange::Wait()
        {
            _debug2("Waiting for fire:%d mem:%lu src_mem_stream:%d src_dev:%d dev:%d mem_stream:%d object:%p src_event:%p", fired, uid_, src_mem_stream_, src_devno_, devno_, dest_mem_stream_, this, src_event_);
            //src_dev_->ResetContext();
            //src_dev_->WaitForEvent(dest_event_, src_mem_stream_, iris_event_wait_default);
            src_dev_->ResetContext();
            src_dev_->EventSynchronize(src_event_);
            dest_dev_->ResetContext();
            _debug2("Completed Waiting for fired:%d mem:%lu src_mem_stream:%d src_dev:%d dev:%d mem_stream:%d object:%p src_event:%p", fired, uid_, src_mem_stream_, src_devno_, devno_, dest_mem_stream_, this, src_event_);
        }
        void SrcCentricGPUEventExchange::Fire()
        {
            fired = true;
            _debug2("Broadcast fire:%d mem:%lu src_mem_stream:%d src_dev:%d dev:%d mem_stream:%d object:%p src_event:%p", fired, uid_, src_mem_stream_, src_devno_, devno_, dest_mem_stream_, this, src_event_);
            _debug2("Broadcast completed fire:%d mem:%lu src_mem_stream:%d src_dev:%d dev:%d mem_stream:%d object:%p src_event:%p", fired, uid_, src_mem_stream_, src_devno_, devno_, dest_mem_stream_, this, src_event_);
        }
    }
}
