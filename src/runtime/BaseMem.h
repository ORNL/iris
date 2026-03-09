#ifndef __IRIS_SRC_BASE_MEM_H__
#define __IRIS_SRC_BASE_MEM_H__

#include "Config.h"
#include "Retainable.h"
#ifdef AUTO_PAR
#include "Task.h"
#include <vector>
#endif //AUTO_PAR

#include "AsyncData.h"
#include <utility>
#include <string>
using namespace std;
#define HANDLE_IN_BASE

namespace iris {
namespace rt {
    class Device;
#ifdef AUTO_PAR
    class Task;
#endif
    enum MemHandlerType{
        IRIS_MEM = 0x1,
        IRIS_DMEM = 0x2,
        IRIS_DMEM_REGION = 0x4,
    };
    class BaseMem;
    using BaseMemDevice = AsyncData<BaseMem>;
    //This is an abstract class
    class BaseMem : public Retainable<struct _iris_mem, BaseMem> {
        public:
            BaseMem(MemHandlerType type, int ndevs) {
                name_ = "MEM";
                source_mem_ = NULL;
                handler_type_ = type;
                reset_ = false;
                enable_reset_ = false;
                host_reset_ = false;
                reset_data_.value_.u64 = 0;
                reset_data_.start_.u64 = 0;
                reset_data_.step_.u64 = 1;
                reset_data_.reset_type_ = iris_reset_memset;
                size_ = 0;
                own_events_ = true;
                element_type_ = iris_unknown;
                ndevs_ = ndevs;
                write_dev_ = -1;
                device_map_ = new BaseMemDevice[ndevs_+1];
                d2h_events_ = new void *[ndevs_];
                write_event_ = new void *[ndevs_+1];
                for (int i = 0; i < ndevs_+1; i++) {
                    if (i == ndevs_)
                        device_map_[i].Init(this, -1);
                    else
                        device_map_[i].Init(this, i);
                    write_event_[i] = NULL;
                }
                for (int i = 0; i < ndevs_; i++) {
                  archs_[i] = NULL;
                  archs_off_[i] = NULL;
                  archs_dev_[i] = NULL;
                  is_usm_[i] = false;
                  d2h_events_[i] = NULL;
                  recommended_stream_[i] = -1;
                  pthread_mutex_init(&host_write_lock_[i], NULL);
                }
                pthread_mutex_init(&host_mutex_, NULL);
                // Special device map for host
                //Retain();
                set_object_track(Platform::GetPlatform()->mem_track_ptr());
                track()->TrackObject(this, uid());
                _trace("Memory object is Created :%lu:%p", uid(), this);
            }
            /*
            static void CompleteCallback(void *stream, int status, BaseMemDevice *data)
            {
                data->EnableCompleted();
            }
            */
            void SetMemHandlerType(MemHandlerType type) { handler_type_ = type; }
            MemHandlerType GetMemHandlerType() { return handler_type_; }
            virtual void HostUnlock() {
                BaseMem *source_mem = (BaseMem*)get_source_mem();
                if (source_mem != NULL && !own_events_) 
                    return source_mem->HostUnlock();
                pthread_mutex_unlock(&host_mutex_);
            }
            virtual void HostLock() {
                BaseMem *source_mem = (BaseMem*)get_source_mem();
                if (source_mem != NULL && !own_events_) 
                    return source_mem->HostLock();
                pthread_mutex_lock(&host_mutex_);
            }
            virtual void HostWriteLock(int devno) {
                BaseMem *source_mem = (BaseMem*)get_source_mem();
                if (source_mem != NULL && !own_events_) 
                    return source_mem->HostWriteLock(devno);
                pthread_mutex_lock(&host_write_lock_[devno]);
            }
            virtual void HostWriteUnLock(int devno) {
                BaseMem *source_mem = (BaseMem*)get_source_mem();
                if (source_mem != NULL && !own_events_) 
                    return source_mem->HostWriteUnLock(devno);
                pthread_mutex_unlock(&host_write_lock_[devno]);
            }
            virtual void HardHostWriteEventSynchronize(Device *dev, void *event);
            virtual int GetHostWriteStream() { 
                BaseMem *source_mem = (BaseMem*)get_source_mem();
                if (source_mem != NULL && !own_events_) 
                    return source_mem->GetHostWriteStream();
                return device_map_[ndevs_].GetWriteStream(); 
            }
            virtual void SetHostWriteStream(int stream) { 
                BaseMem *source_mem = (BaseMem*)get_source_mem();
                if (source_mem != NULL && !own_events_) 
                    return source_mem->SetHostWriteStream(stream);
                device_map_[ndevs_].SetWriteStream(stream); 
            }
            virtual int GetHostWriteDevice() { 
                BaseMem *source_mem = (BaseMem*)get_source_mem();
                if (source_mem != NULL && !own_events_) 
                    return source_mem->GetHostWriteDevice();
                return device_map_[ndevs_].devno(); 
            }
            virtual void SetHostWriteDevice(int dev) { 
                BaseMem *source_mem = (BaseMem*)get_source_mem();
                if (source_mem != NULL && !own_events_) 
                    return source_mem->SetHostWriteDevice(dev);
                device_map_[ndevs_].set_devno(dev); 
            }
            virtual void HostRecordEvent(int devno, int stream);
            virtual void *GetHostCompletionEvent() { 
                BaseMem *source_mem = (BaseMem*)get_source_mem();
                if (source_mem != NULL && !own_events_) 
                    return source_mem->GetHostCompletionEvent();
                return device_map_[ndevs_].GetCompletionEvent(); 
            }
            virtual void **GetHostCompletionEventPtr(bool new_entry=false) { 
                BaseMem *source_mem = (BaseMem*)get_source_mem();
                if (source_mem != NULL && !own_events_) 
                    return source_mem->GetHostCompletionEventPtr(new_entry);
                return device_map_[ndevs_].GetCompletionEventPtr(new_entry); 
            }
            virtual void *GetDeviceSpecificHostCompletionEvent(int devno) {
                BaseMem *source_mem = (BaseMem*)get_source_mem();
                if (source_mem != NULL && !own_events_) 
                    return source_mem->GetDeviceSpecificHostCompletionEvent(devno);
                ASSERT(devno >=0 && devno < ndevs_);
                return d2h_events_[devno];
            }
            virtual ~BaseMem() { 
                pthread_mutex_destroy(&host_mutex_);
                _trace("Memory object is getting deleted:%lu:%p", uid(), this);
                for(int i=0; i<ndevs_; i++) {
                    pthread_mutex_destroy(&host_write_lock_[i]);
                    stack<void *> & stk = device_map_[i].GetCompletionStack();
                    int n = stk.size();
                    for(int j=0; j<n; j++) { 
                        void *top = stk.top();
                        if (top != NULL) 
                            DestroyEvent(i, top);
                        stk.pop();
                    } 
                }
                delete [] d2h_events_;
                delete [] write_event_;
                delete [] device_map_;
                //track()->UntrackObject(this, uid());
            }
            virtual int  get_dev_affinity()  { return -1; }
            virtual void* arch(Device* dev, void *host=NULL) = 0;
            virtual void* arch(int devno, void *host=NULL) = 0;
            virtual void** arch_ptr(Device *dev, void *host=NULL) = 0;
            virtual void** arch_ptr(int devno, void *host=NULL) = 0;
            virtual void init_reset(bool reset=true) { reset_ = reset; enable_reset_ = reset; }
            inline bool is_reset() { return reset_ & enable_reset_; }
            inline void disable_reset() { enable_reset_ = false; }
            inline void enable_reset() { enable_reset_ = true; }
            inline void use_parent_events() { own_events_ = false; }
            inline void use_own_events() { own_events_ = true; }
            inline ResetData & reset_data() { return reset_data_; }
            inline void set_reset_type(int reset_type) { 
                reset_data_.reset_type_ = reset_type; 
            } 
            inline void set_reset_assign(IRISValue value) {
                reset_ = true;
                enable_reset_ = true;
                reset_data_.reset_type_ = iris_reset_assign; 
                reset_data_.value_ = value;
            } 
            inline void set_reset_arith_seq(IRISValue start, IRISValue increment) { 
                reset_ = true;
                enable_reset_ = true;
                reset_data_.reset_type_ = iris_reset_arith_seq; 
                reset_data_.start_ = start;
                reset_data_.step_ = increment;
            } 
            inline void set_reset_geom_seq(IRISValue start, IRISValue step) { 
                reset_ = true;
                enable_reset_ = true;
                reset_data_.reset_type_ = iris_reset_geom_seq; 
                reset_data_.start_ = start;
                reset_data_.step_ = step;
            } 
            inline void set_reset_seed(long long seed) {
                reset_data_.seed_ = seed;
            }
            inline void set_reset_min(IRISValue min) {
                reset_data_.p1_ = min;
            }
            inline void set_reset_max(IRISValue max) {
                reset_data_.p2_ = max;
            }
            inline void set_reset_mean(IRISValue mean) {
                reset_data_.p1_ = mean;
            }
            inline void set_reset_stddev(IRISValue stddev) {
                reset_data_.p2_ = stddev;
            }
            inline void** archs_off() { return archs_off_; }
            virtual void** archs() { return archs_; }
            virtual void **get_arch_ptr(int devno) { return &archs_[devno]; }
            virtual void *get_arch(int devno) { return archs_[devno]; }
            virtual void set_arch(int devno, void *ptr) { archs_[devno] = ptr; }
            inline size_t size() { return size_; }
            virtual size_t full_size() { return size_; }
            void dev_unlock(int devno) {
              BaseMem *source_mem = (BaseMem*)get_source_mem();
              if (source_mem != NULL && !own_events_) 
                  return source_mem->dev_unlock(devno);
              ASSERT(devno >=0 && devno < ndevs_+1);
              device_map_[devno].Unlock();
            }
            void dev_lock(int devno) {
              BaseMem *source_mem = (BaseMem*)get_source_mem();
              if (source_mem != NULL && !own_events_) 
                  return source_mem->dev_lock(devno);
              ASSERT(devno >=0 && devno < ndevs_+1);
              device_map_[devno].Lock();
            }
            void host_unlock() {
              BaseMem *source_mem = (BaseMem*)get_source_mem();
              if (source_mem != NULL && !own_events_) 
                  return source_mem->host_unlock();
              device_map_[ndevs_].Unlock();
            }
            void host_lock() {
              BaseMem *source_mem = (BaseMem*)get_source_mem();
              if (source_mem != NULL && !own_events_) 
                  return source_mem->host_lock();
              device_map_[ndevs_].Lock();
            }
            void clear_streams() {
                BaseMem *source_mem = (BaseMem*)get_source_mem();
                if (source_mem != NULL && !own_events_) 
                    return source_mem->clear_streams();
                for(int i=0; i<ndevs_+1; i++) {
                    device_map_[i].Clear();
                    write_event_[i] = NULL;
                }
                device_map_[ndevs_].ClearDevice();
                write_dev_ = -1;
            }
            void clear_read_tracks() {
                for(int i=0; i<ndevs_; i++) {
                    device_map_[i].ClearReadTrack();
                }
            }
            void clear_read_tracks(int devno) {
                ASSERT(devno < ndevs_);
                device_map_[devno].ClearReadTrack();
            }
            void insert_read_track(int devno, int stream, Task *task, void *event) {
                ASSERT(devno < ndevs_);
                device_map_[devno].InsertReadTrack(stream, task, event);
            }
            map<int, DataReadTrack> & read_tracks(int devno) {
                ASSERT(devno < ndevs_);
                return device_map_[devno].ReadTracks();
            }
            void ClearAndAddWaitEvent(int devno, void *event) { 
                BaseMem *source_mem = (BaseMem*)get_source_mem();
                if (source_mem != NULL && !own_events_) 
                    return source_mem->ClearAndAddWaitEvent(devno, event);
                ASSERT(devno >=0 && devno < ndevs_+1);
                device_map_[devno].ClearAndAddWaitEvent(event); 
            }
            void AddWaitEvent(int devno, void *event) { 
                BaseMem *source_mem = (BaseMem*)get_source_mem();
                if (source_mem != NULL && !own_events_) 
                    return source_mem->AddWaitEvent(devno, event);
                ASSERT(devno >=0 && devno < ndevs_+1);
                device_map_[devno].AddWaitEvent(event); 
            }
            void ClearWaitEvents(int devno) { 
                BaseMem *source_mem = (BaseMem*)get_source_mem();
                if (source_mem != NULL && !own_events_) 
                    return source_mem->ClearWaitEvents(devno);
                ASSERT(devno >=0 && devno < ndevs_+1);
                device_map_[devno].ClearWaitEvents(); 
            }
            vector<void *> & GetWaitEvents(int devno) { 
                BaseMem *source_mem = (BaseMem*)get_source_mem();
                if (source_mem != NULL && !own_events_) 
                    return source_mem->GetWaitEvents(devno);
                ASSERT(devno >=0 && devno < ndevs_+1);
                return device_map_[devno].GetWaitEvents(); 
            }
            void SetWriteDevice(int devno) { write_dev_ = devno; }
            int GetWriteDevice() { return write_dev_; }
            void SetWriteDeviceEvent(int devno, void *event) { 
                BaseMem *source_mem = (BaseMem*)get_source_mem();
                if (source_mem != NULL && !own_events_) 
                    return source_mem->SetWriteDeviceEvent(devno, event);
                ASSERT(devno >=0 && devno < ndevs_+1);
                write_event_[devno] = event; 
            }
            void *GetWriteDeviceEvent(int devno) { 
                BaseMem *source_mem = (BaseMem*)get_source_mem();
                if (source_mem != NULL && !own_events_) 
                    return source_mem->GetWriteDeviceEvent(devno);
                ASSERT(devno >=0 && devno < ndevs_+1);
                return write_event_[devno]; 
            }
            int GetWriteStream(int devno) {
                BaseMem *source_mem = (BaseMem*)get_source_mem();
                if (source_mem != NULL && !own_events_) 
                    return source_mem->GetWriteStream(devno);
                ASSERT(devno >=0 && devno < ndevs_+1);
                return device_map_[devno].GetWriteStream(); 
            }
            void SetWriteStream(int devno, int stream) { 
                BaseMem *source_mem = (BaseMem*)get_source_mem();
                if (source_mem != NULL && !own_events_) 
                    return source_mem->SetWriteStream(devno, stream);
                ASSERT(devno >=0 && devno < ndevs_+1);
                device_map_[devno].SetWriteStream(stream); 
            }
            void HardDeviceWriteEventSynchronize(Device *dev, void *event);
            bool IsProactive(int devno) { 
                BaseMem *source_mem = (BaseMem*)get_source_mem();
                if (source_mem != NULL && !own_events_) 
                    return source_mem->IsProactive(devno);
                ASSERT(devno >=0 && devno < ndevs_+1);
                return device_map_[devno].IsProactive(); 
            }
            void EnableProactive(int devno) { 
                BaseMem *source_mem = (BaseMem*)get_source_mem();
                if (source_mem != NULL && !own_events_) 
                    return source_mem->EnableProactive(devno);
                ASSERT(devno >=0 && devno < ndevs_+1);
                device_map_[devno].EnableProactive(); 
            }
            void DisableProactive(int devno) { 
                BaseMem *source_mem = (BaseMem*)get_source_mem();
                if (source_mem != NULL && !own_events_) 
                    return source_mem->DisableProactive(devno);
                ASSERT(devno >=0 && devno < ndevs_+1);
                device_map_[devno].DisableProactive(); 
            }
            EventExchange *GetEventExchange(int devno) { 
                BaseMem *source_mem = (BaseMem*)get_source_mem();
                if (source_mem != NULL && !own_events_) 
                    return source_mem->GetEventExchange(devno);
                ASSERT(devno >=0 && devno < ndevs_+1);
                return device_map_[devno].exchange(); 
            }
            void *GetCompletionEvent(int devno) { 
                BaseMem *source_mem = (BaseMem*)get_source_mem();
                if (source_mem != NULL && !own_events_) 
                    return source_mem->GetCompletionEvent(devno);
                ASSERT(devno >=0 && devno < ndevs_+1);
                return device_map_[devno].GetCompletionEvent(); 
            }
            void **GetCompletionEventPtr(int devno, bool new_entry=false) { 
                BaseMem *source_mem = (BaseMem*)get_source_mem();
                if (source_mem != NULL && !own_events_) 
                    return source_mem->GetCompletionEventPtr(devno, new_entry);
                ASSERT(devno >=0 && devno < ndevs_+1);
                return device_map_[devno].GetCompletionEventPtr(new_entry); 
            }
            void clear_d2h_events() { 
                BaseMem *source_mem = (BaseMem*)get_source_mem();
                if (source_mem != NULL && !own_events_) 
                    return source_mem->clear_d2h_events();
                for (int i=0; i<ndevs_; i++) d2h_events_[i] = NULL; 
            }
            void set_d2h_event(int devno, void *host_event) {
                BaseMem *source_mem = (BaseMem*)get_source_mem();
                if (source_mem != NULL && !own_events_) 
                    return source_mem->set_d2h_event();
                d2h_events_[devno] = host_event;
            }
#ifdef AUTO_PAR
  	    inline Task* get_current_writing_task() { return current_writing_task_;}
  	    inline void set_current_writing_task(Task* task) { current_writing_task_ = task;}
  	    void add_to_read_task_list(Task* task) { read_task_list_.push_back(task); }
  	    std::vector<Task*>* get_read_task_list() { return &read_task_list_; }
  	    void erase_all_read_task_list() {
		    read_task_list_.erase(read_task_list_.begin(), 
		    read_task_list_.end()); }

#ifdef AUTO_FLUSH
  	    inline Task* get_flush_task() { return flush_task_;}
  	    inline void set_flush_task(Task* flush_task) { flush_task_ = flush_task;}
#endif 

#endif
            inline int ndevs() { return ndevs_; }
            virtual inline void clear() { }
        void set_recommended_stream(int devno, int stream) { 
#ifdef HANDLE_IN_BASE
            BaseMem *source_mem = (BaseMem*)get_source_mem();
            if (source_mem != NULL && !own_events_) 
                source_mem->set_recommended_stream(devno, stream);
            else
#endif
                recommended_stream_[devno] = stream; 
        }
        int recommended_stream(int devno) { 
#ifdef HANDLE_IN_BASE
            BaseMem *source_mem = (BaseMem*)get_source_mem();
            if (source_mem != NULL && !own_events_) 
                return source_mem->recommended_stream(devno);
#endif
            return recommended_stream_[devno]; 
        }
        int *recommended_stream_ptr(int devno) { return &recommended_stream_[devno]; }
        void *RecordEvent(int devno, int stream, bool new_entry=false);
        void WaitForEvent(int devno, int stream, int dep_devno);
        void DestroyEvent(int devno, void *event);
        bool is_usm(int devno) { return is_usm_[devno]; }
        void set_usm_flag(int devno, bool flag=true) { is_usm_[devno] = flag; }
        int element_type() { return element_type_; }
        bool is_host_reset() { return host_reset_; }
        void set_element_type(int t) { element_type_ = t; }
        pair<bool, int8_t> IsResetPossibleWithMemset();
        pair<bool, int8_t> IsResetPossibleWithMemset(ResetData & reset_data);
        pair<bool, int8_t> IsResetPossibleWithMemset(ResetData & reset_data, int element_type);
        virtual void set_source_mem(BaseMem *mem) { source_mem_ = mem; }
        const char* name() { return name_.c_str(); }
        void set_name(const char* name);
        void set_name(string name);
        BaseMem *get_source_mem(bool deep=true) { 
            if (source_mem_ != NULL && deep) {
                if (source_mem_->get_source_mem() == NULL)
                    return source_mem_;
                else
                    return source_mem_->get_source_mem();
            }
            return source_mem_; 
        }
        protected:
            BaseMem *source_mem_;
            int element_type_;
            int recommended_stream_[IRIS_MAX_NDEVS];
            MemHandlerType handler_type_;
            void* archs_[IRIS_MAX_NDEVS];
            Device* archs_dev_[IRIS_MAX_NDEVS];
            void* archs_off_[IRIS_MAX_NDEVS];
            size_t size_;
            int ndevs_;
            bool  reset_;
            bool  enable_reset_;
            bool host_reset_;
            bool  is_usm_[IRIS_MAX_NDEVS];
            ResetData reset_data_;
            BaseMemDevice *device_map_;
            int write_dev_;
            void **write_event_;
            void **d2h_events_;
            string name_;
            bool own_events_;
            pthread_mutex_t host_mutex_;
            pthread_mutex_t host_write_lock_[IRIS_MAX_NDEVS];
#ifdef AUTO_PAR
  	    Task* current_writing_task_;
  	    std::vector<Task*> read_task_list_;
#ifdef AUTO_FLUSH
  	    Task* flush_task_;
#endif 
#endif
 
    };
}
}

#endif // __IRIS_SRC_BASE_MEM_H__
