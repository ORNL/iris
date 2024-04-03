#pragma once
#include <pthread.h>
#include "Debug.h"
#include <condition_variable>
#include <mutex>

namespace iris {
    namespace rt {
        class StdThreadEventExchange;
        class PthreadEventExchange;
        class DestCentricGPUEventExchange;
        class SrcCentricGPUEventExchange;
        class Device;
        //Play with different event exchange mechanisms here
        //using EventExchange = PthreadEventExchange;
        //using EventExchange = StdThreadEventExchange;
        //using EventExchange = DestCentricGPUEventExchange;
        using EventExchange = SrcCentricGPUEventExchange;
        class BaseEventExchange {
            public:
                unsigned long uid_;
                int src_mem_stream_;
                int src_devno_;
                int devno_;
                int dest_mem_stream_;
                bool fired;
                Device *src_dev_;
                Device *dest_dev_;
                void *src_event_;
                void *dest_event_;
                void *src_event() { return src_event_; }
                void *dest_event() { return dest_event_; }
                BaseEventExchange() {
                    Clear();
                }
                void Clear() { 
                    fired = false;
                    src_mem_stream_ = -1;
                    src_devno_ = -1;
                    devno_ = -1;
                    dest_mem_stream_ = -1;
                    uid_ = 0;
                    dest_dev_ = NULL;
                    src_dev_ = NULL;
                    dest_dev_ = NULL;
                    dest_event_ = NULL;
                    src_event_ = NULL;
                }
                virtual void Wait()=0;
                virtual void Fire()=0;
                void set_mem(unsigned long uid, int src_mem_stream, int src_devno, int mem_stream, int devno, Device *src_dev, Device *dest_dev, void *src_event, void *dest_event) {
                    uid_ = uid; src_mem_stream_ = src_mem_stream; src_devno_ = src_devno; dest_mem_stream_ = mem_stream; devno_ = devno;
                    src_dev_ = src_dev;
                    dest_dev_ = dest_dev;
                    dest_event_ = dest_event;
                    src_event_ = src_event;
                }
                virtual ~BaseEventExchange() {}
                static void Wait(void *stream, int status, void *data) {
                    BaseEventExchange *exchange = (BaseEventExchange *)data;
                    //_info("Waiting for stream:%p fired:%d object:%p", stream, exchange->fired, exchange);
                    exchange->Wait();
                    //_info("Wait completed for stream:%p fired:%d object:%p", stream, exchange->fired, exchange);
                }
                static void Fire(void *stream, int status, void *data) {
                    BaseEventExchange *exchange = (BaseEventExchange *)data;
                    //_info("Fire for event:%p fired:%d object:%p", stream, exchange->fired, exchange);
                    exchange->Fire();
                    //_info("Fire complete for event:%p fired:%d object:%p", stream, exchange->fired, exchange);
                }
        };
        class DestCentricGPUEventExchange : public BaseEventExchange {
            public:
                DestCentricGPUEventExchange() : BaseEventExchange() { }
                ~DestCentricGPUEventExchange() { }
                void Wait();
                void Fire();
            private:
        };
        class SrcCentricGPUEventExchange : public BaseEventExchange {
            public:
                SrcCentricGPUEventExchange() : BaseEventExchange() { }
                ~SrcCentricGPUEventExchange() { }
                void Wait();
                void Fire();
            private:
        };
        //This class support event mechanism between heterorgeneous devices
        //There could be multiple waits for the single fire
        //Broadcast mechanism will be used for fire
        //It works based on pthreads
        class StdThreadEventExchange : public BaseEventExchange {
            public:
                StdThreadEventExchange() : BaseEventExchange() { }
                ~StdThreadEventExchange() { }
                void Wait()
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    _info("Waiting for fire:%d mem:%lu src_mem_stream:%d src_dev:%d dev:%d mem_stream:%d object:%p", fired, uid_, src_mem_stream_, src_devno_, devno_, dest_mem_stream_, this);
                    cond_.wait(lock, [this]{ return fired; });
                    _info("Completed Waiting for fired:%d mem:%lu src_mem_stream:%d src_dev:%d dev:%d mem_stream:%d object:%p", fired, uid_, src_mem_stream_, src_devno_, devno_, dest_mem_stream_, this);
                }
                void Fire()
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    fired = true;
                    _info("Broadcast fire:%d mem:%lu src_mem_stream:%d src_dev:%d dev:%d mem_stream:%d object:%p", fired, uid_, src_mem_stream_, src_devno_, devno_, dest_mem_stream_, this);
                    cond_.notify_all();
                    _info("Broadcast completed fire:%d mem:%lu src_mem_stream:%d src_dev:%d dev:%d mem_stream:%d object:%p", fired, uid_, src_mem_stream_, src_devno_, devno_, dest_mem_stream_, this);
                }
            private:
                std::mutex mutex_;
                std::condition_variable cond_;
        };
        class PthreadEventExchange : public BaseEventExchange {
            public:
                PthreadEventExchange() : BaseEventExchange() {
                    pthread_mutex_init(&mutex_, NULL);
                    pthread_cond_init(&cond_, NULL);
                }
                ~PthreadEventExchange() {
                    pthread_mutex_destroy(&mutex_);
                    pthread_cond_destroy(&cond_);
                }
                void Wait()
                {
                    Lock();
                    while (!fired) {
                        _info("Waiting for fire:%d mem:%lu src_mem_stream:%d src_dev:%d dev:%d mem_stream:%d object:%p", fired, uid_, src_mem_stream_, src_devno_, devno_, dest_mem_stream_, this);
                        pthread_cond_wait(&cond_, &mutex_);
                        _info("Completed Waiting for fired:%d mem:%lu src_mem_stream:%d src_dev:%d dev:%d mem_stream:%d object:%p", fired, uid_, src_mem_stream_, src_devno_, devno_, dest_mem_stream_, this);
                    }
                    Unlock();
                }
                void Fire()
                {
                    Lock();
                    fired = true;
                    _info("Broadcast fire:%d mem:%lu src_mem_stream:%d src_dev:%d dev:%d mem_stream:%d object:%p", fired, uid_, src_mem_stream_, src_devno_, devno_, dest_mem_stream_, this);
                    pthread_cond_broadcast(&cond_);
                    _info("Broadcast completed fire:%d mem:%lu src_mem_stream:%d src_dev:%d dev:%d mem_stream:%d object:%p", fired, uid_, src_mem_stream_, src_devno_, devno_, dest_mem_stream_, this);
                    Unlock();
                }
            private:
                void Unlock() {
                    pthread_mutex_unlock(&mutex_);
                }
                void Lock() {
                    pthread_mutex_lock(&mutex_);
                }
                pthread_mutex_t mutex_;
                pthread_cond_t  cond_;
        };
    }
}
