#ifndef __IRIS_SRC_BASE_MEM_H__
#define __IRIS_SRC_BASE_MEM_H__

#include "Config.h"
#include "Retainable.h"
namespace iris {
namespace rt {
    class Device;
    enum MemHandlerType{
        IRIS_MEM = 0x1,
        IRIS_DMEM = 0x2,
        IRIS_DMEM_REGION = 0x4,
    };
    //This is an abstract class
    class BaseMem : public Retainable<struct _iris_mem, BaseMem> {
        public:
            BaseMem(MemHandlerType type, int ndevs) {
                handler_type_ = type;
                reset_ = false;
                size_ = 0;
                ndevs_ = ndevs;
                for (int i = 0; i < ndevs_; i++) {
                  archs_[i] = NULL;
                  archs_off_[i] = NULL;
                  archs_dev_[i] = NULL;
                }
                set_object_track(Platform::GetPlatform()->mem_track_ptr());
                track()->TrackObject(this, uid());
                _trace("Memory object is Created :%lu:%p", uid(), this);
            }
            MemHandlerType GetMemHandlerType() { return handler_type_; }
            virtual ~BaseMem() { 
                _trace("Memory object is deleted:%lu:%p", uid(), this);
                track()->UntrackObject(this, uid());
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
            inline int ndevs() { return ndevs_; }
            virtual inline void clear() { }
        protected:
            MemHandlerType handler_type_;
            void* archs_[IRIS_MAX_NDEVS];
            Device* archs_dev_[IRIS_MAX_NDEVS];
            void* archs_off_[IRIS_MAX_NDEVS];
            size_t size_;
            int ndevs_;
            bool  reset_;

    };
}
}

#endif // __IRIS_SRC_BASE_MEM_H__
