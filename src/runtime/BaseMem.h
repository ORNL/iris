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
using namespace std;

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
                element_type_ = iris_unknown;
                ndevs_ = ndevs;
                write_dev_ = -1;
                device_map_ = new BaseMemDevice[ndevs_+1];
                d2h_events_ = new void *[ndevs_];
                write_event_ = new void *[ndevs+1];
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
            void HostUnlock() {
                pthread_mutex_unlock(&host_mutex_);
            }
            void HostLock() {
                pthread_mutex_lock(&host_mutex_);
            }
            void HostWriteLock(int devno) {
                pthread_mutex_lock(&host_write_lock_[devno]);
            }
            void HostWriteUnLock(int devno) {
                pthread_mutex_unlock(&host_write_lock_[devno]);
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
            inline void** archs() { return archs_; }
            inline void **get_arch_ptr(int devno) { return &archs_[devno]; }
            inline void *get_arch(int devno) { return archs_[devno]; }
            inline void set_arch(int devno, void *ptr) { archs_[devno] = ptr; }
            inline size_t size() { return size_; }
            void dev_unlock(int devno) {
              device_map_[devno].Unlock();
            }
            void dev_lock(int devno) {
              device_map_[devno].Lock();
            }
            void host_unlock() {
              device_map_[ndevs_].Unlock();
            }
            void host_lock() {
              device_map_[ndevs_].Lock();
            }
            void clear_streams() {
                for(int i=0; i<ndevs_+1; i++) {
                    device_map_[i].Clear();
                    write_event_[i] = NULL;
                }
                device_map_[ndevs_].ClearDevice();
                write_dev_ = -1;
            }
            void ClearAndAddWaitEvent(int devno, void *event) { 
                ASSERT(devno >=0 && devno < ndevs_+1);
                device_map_[devno].ClearAndAddWaitEvent(event); 
            }
            void AddWaitEvent(int devno, void *event) { 
                ASSERT(devno >=0 && devno < ndevs_+1);
                device_map_[devno].AddWaitEvent(event); 
            }
            void ClearWaitEvents(int devno) { 
                ASSERT(devno >=0 && devno < ndevs_+1);
                device_map_[devno].ClearWaitEvents(); 
            }
            vector<void *> & GetWaitEvents(int devno) { 
                ASSERT(devno >=0 && devno < ndevs_+1);
                return device_map_[devno].GetWaitEvents(); 
            }
            void SetWriteDevice(int devno) { write_dev_ = devno; }
            int GetWriteDevice() { return write_dev_; }
            void SetWriteDeviceEvent(int devno, void *event) { 
                ASSERT(devno >=0 && devno < ndevs_+1);
                write_event_[devno] = event; 
            }
            void *GetWriteDeviceEvent(int devno) { 
                ASSERT(devno >=0 && devno < ndevs_+1);
                return write_event_[devno]; 
            }
            int GetWriteStream(int devno) { 
                ASSERT(devno >=0 && devno < ndevs_+1);
                return device_map_[devno].GetWriteStream(); 
            }
            void SetWriteStream(int devno, int stream) { 
                ASSERT(devno >=0 && devno < ndevs_+1);
                device_map_[devno].SetWriteStream(stream); 
            }
            void HardDeviceWriteEventSynchronize(Device *dev, void *event);
            void HardHostWriteEventSynchronize(Device *dev, void *event);
            bool IsProactive(int devno) { 
                ASSERT(devno >=0 && devno < ndevs_+1);
                return device_map_[devno].IsProactive(); 
            }
            void EnableProactive(int devno) { 
                ASSERT(devno >=0 && devno < ndevs_+1);
                device_map_[devno].EnableProactive(); 
            }
            void DisableProactive(int devno) { 
                ASSERT(devno >=0 && devno < ndevs_+1);
                device_map_[devno].DisableProactive(); 
            }
            EventExchange *GetEventExchange(int devno) { 
                ASSERT(devno >=0 && devno < ndevs_+1);
                return device_map_[devno].exchange(); 
            }
            void *GetCompletionEvent(int devno) { 
                ASSERT(devno >=0 && devno < ndevs_+1);
                return device_map_[devno].GetCompletionEvent(); 
            }
            void **GetCompletionEventPtr(int devno, bool new_entry=false) { 
                ASSERT(devno >=0 && devno < ndevs_+1);
                return device_map_[devno].GetCompletionEventPtr(new_entry); 
            }
            int GetHostWriteStream() { 
                return device_map_[ndevs_].GetWriteStream(); 
            }
            void SetHostWriteStream(int stream) { 
                device_map_[ndevs_].SetWriteStream(stream); 
            }
            int GetHostWriteDevice() { 
                return device_map_[ndevs_].devno(); 
            }
            void SetHostWriteDevice(int dev) { 
                device_map_[ndevs_].set_devno(dev); 
            }
            void HostRecordEvent(int devno, int stream);
            void *GetHostCompletionEvent() { 
                return device_map_[ndevs_].GetCompletionEvent(); 
            }
            void **GetHostCompletionEventPtr(bool new_entry=false) { 
                return device_map_[ndevs_].GetCompletionEventPtr(new_entry); 
            }
            void *GetDeviceSpecificHostCompletionEvent(int devno) {
                ASSERT(devno >=0 && devno < ndevs_);
                return d2h_events_[devno];
            }
            void clear_d2h_events() { 
                for (int i=0; i<ndevs_; i++) d2h_events_[i] = NULL; 
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
        void set_recommended_stream(int devno, int stream) { recommended_stream_[devno] = stream; }
        int recommended_stream(int devno) { return recommended_stream_[devno]; }
        void *RecordEvent(int devno, int stream, bool new_entry=false);
        void WaitForEvent(int devno, int stream, int dep_devno);
        void DestroyEvent(int devno, void *event);
        bool is_usm(int devno) { return is_usm_[devno]; }
        void set_usm_flag(int devno, bool flag=true) { is_usm_[devno] = flag; }
        int element_type() { return element_type_; }
        void set_element_type(int t) { element_type_ = t; }
        pair<bool, int8_t> IsResetPossibleWithMemset();
        pair<bool, int8_t> IsResetPossibleWithMemset(ResetData & reset_data);
        void set_source_mem(BaseMem *mem) { source_mem_ = (DataMem *)mem; }
        DataMem *get_source_mem() { return source_mem_; }
        protected:
            DataMem *source_mem_;
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
