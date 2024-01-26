#ifndef __IRIS_SRC_BASE_MEM_H__
#define __IRIS_SRC_BASE_MEM_H__

#include "Config.h"
#include "Retainable.h"
#ifdef AUTO_PAR
#include "Task.h"
#include <vector>
#endif //AUTO_PAR

#include "AsyncData.h"


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
                handler_type_ = type;
                reset_ = false;
                size_ = 0;
                ndevs_ = ndevs;
                device_map_ = new BaseMemDevice[ndevs_+1];
                for (int i = 0; i < ndevs_; i++) {
                  archs_[i] = NULL;
                  archs_off_[i] = NULL;
                  archs_dev_[i] = NULL;
                  device_map_[i].Init(this, i);
                  recommended_stream_[i] = -1;
                }
                // Special device map for host
                device_map_[ndevs_].Init(this, -1);
                Retain();
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
            virtual ~BaseMem() { 
                _trace("Memory object is deleted:%lu:%p", uid(), this);
                for(int i=0; i<ndevs_; i++) {
                    if (GetCompletionEvent(i) != NULL) 
                        DestroyEvent(i, GetCompletionEvent(i));
                }
                delete [] device_map_;
                //track()->UntrackObject(this, uid());
            }
            virtual void* arch(Device* dev, void *host=NULL) = 0;
            virtual void* arch(int devno, void *host=NULL) = 0;
            virtual void** arch_ptr(Device *dev, void *host=NULL) = 0;
            virtual void** arch_ptr(int devno, void *host=NULL) = 0;
            virtual void init_reset(bool reset=true) { reset_ = reset; }
            inline bool is_reset() { return reset_; }
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
                }
                device_map_[ndevs_].ClearDevice();
            }
            void ClearAndAddWaitEvent(int devno, void *event) { device_map_[devno].ClearAndAddWaitEvent(event); }
            void AddWaitEvent(int devno, void *event) { device_map_[devno].AddWaitEvent(event); }
            void ClearWaitEvents(int devno) { device_map_[devno].ClearWaitEvents(); }
            vector<void *> & GetWaitEvents(int devno) { return device_map_[devno].GetWaitEvents(); }
            int GetWriteStream(int devno) { return device_map_[devno].GetWriteStream(); }
            void SetWriteStream(int devno, int stream) { device_map_[devno].SetWriteStream(stream); }
            bool IsProactive(int devno) { return device_map_[devno].IsProactive(); }
            void EnableProactive(int devno) { device_map_[devno].EnableProactive(); }
            void DisableProactive(int devno) { device_map_[devno].DisableProactive(); }
            EventExchange *GetEventExchange(int devno) { return device_map_[devno].exchange(); }
            void *GetCompletionEvent(int devno) { return device_map_[devno].GetCompletionEvent(); }
            void **GetCompletionEventPtr(int devno) { return device_map_[devno].GetCompletionEventPtr(); }
            void *GetHostCompletionEvent() { return device_map_[ndevs_].GetCompletionEvent(); }
            void **GetHostCompletionEventPtr(int devno) { return device_map_[ndevs_].GetCompletionEventPtr(); }
            int GetHostWriteStream() { return device_map_[ndevs_].GetWriteStream(); }
            void SetHostWriteStream(int stream) { device_map_[ndevs_].SetWriteStream(stream); }
            int GetHostWriteDevice() { return device_map_[ndevs_].devno(); }
            void SetHostWriteDevice(int dev) { device_map_[ndevs_].set_devno(dev); }
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
        void RecordEvent(int devno, int stream);
        void WaitForEvent(int devno, int stream, int dep_devno);
        void DestroyEvent(int devno, void *event);

        protected:
            int recommended_stream_[IRIS_MAX_NDEVS];
            MemHandlerType handler_type_;
            void* archs_[IRIS_MAX_NDEVS];
            Device* archs_dev_[IRIS_MAX_NDEVS];
            void* archs_off_[IRIS_MAX_NDEVS];
            size_t size_;
            int ndevs_;
            bool  reset_;
            BaseMemDevice *device_map_;
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
