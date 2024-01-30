#include "EventExchange.h"
#include "Device.h"

namespace iris
{
    namespace rt {
        void DestCentricGPUEventExchange::Wait()
        {
            _info("Waiting for fire:%d mem:%lu src_mem_stream:%d src_dev:%d dev:%d mem_stream:%d object:%p", fired, uid_, src_mem_stream_, src_devno_, devno_, dest_mem_stream_, this);
            dest_dev_->ResetContext();
            dest_dev_->WaitForEvent(dest_event_, dest_mem_stream_, iris_event_wait_default);
            _info("Completed Waiting for fired:%d mem:%lu src_mem_stream:%d src_dev:%d dev:%d mem_stream:%d object:%p", fired, uid_, src_mem_stream_, src_devno_, devno_, dest_mem_stream_, this);
        }
        void DestCentricGPUEventExchange::Fire()
        {
            fired = true;
            _info("Broadcast fire:%d mem:%lu src_mem_stream:%d src_dev:%d dev:%d mem_stream:%d object:%p", fired, uid_, src_mem_stream_, src_devno_, devno_, dest_mem_stream_, this);
            dest_dev_->ResetContext();
            dest_dev_->RecordEvent(dest_event_, 0);
            _info("Broadcast completed fire:%d mem:%lu src_mem_stream:%d src_dev:%d dev:%d mem_stream:%d object:%p", fired, uid_, src_mem_stream_, src_devno_, devno_, dest_mem_stream_, this);
        }
        void SrcCentricGPUEventExchange::Wait()
        {
            _info("Waiting for fire:%d mem:%lu src_mem_stream:%d src_dev:%d dev:%d mem_stream:%d object:%p", fired, uid_, src_mem_stream_, src_devno_, devno_, dest_mem_stream_, this);
            src_dev_->ResetContext();
            //src_dev_->WaitForEvent(dest_event_, src_mem_stream_, iris_event_wait_default);
            src_dev_->EventSychronize(src_event_);
            dest_dev_->ResetContext();
            _info("Completed Waiting for fired:%d mem:%lu src_mem_stream:%d src_dev:%d dev:%d mem_stream:%d object:%p", fired, uid_, src_mem_stream_, src_devno_, devno_, dest_mem_stream_, this);
        }
        void SrcCentricGPUEventExchange::Fire()
        {
            fired = true;
            _info("Broadcast fire:%d mem:%lu src_mem_stream:%d src_dev:%d dev:%d mem_stream:%d object:%p", fired, uid_, src_mem_stream_, src_devno_, devno_, dest_mem_stream_, this);
            _info("Broadcast completed fire:%d mem:%lu src_mem_stream:%d src_dev:%d dev:%d mem_stream:%d object:%p", fired, uid_, src_mem_stream_, src_devno_, devno_, dest_mem_stream_, this);
        }
    }
}
