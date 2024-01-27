#pragma once
#include <pthread.h>
#include "Debug.h"

namespace iris {
    namespace rt {
        //This class support event mechanism between heterorgeneous devices
        //There could be multiple waits for the single fire
        //Broadcast mechanism will be used for fire
        //It works based on pthreads
        class EventExchange {
            public:
                unsigned long uid_;
                int written_stream_;
                int src_dev_;
                int devno_;
                int mem_stream_;
                EventExchange() {
                    pthread_mutex_init(&mutex_, NULL);
                    pthread_cond_init(&cond_, NULL);
                    fired = false;
                }
                void Clear() { 
                    fired = false;
                }
                ~EventExchange() {
                    pthread_mutex_destroy(&mutex_);
                    pthread_cond_destroy(&cond_);
                }
                void Wait()
                {
                    Lock();
                    while (!fired) {
                        _info("Waiting for fire:%d mem:%lu written_stream:%d src_dev:%d dev:%d mem_stream:%d", fired, uid_, written_stream_, src_dev_, devno_, mem_stream_);
                        pthread_cond_wait(&cond_, &mutex_);
                        _info("Completed Waiting for fired:%d mem:%lu written_stream:%d src_dev:%d dev:%d mem_stream:%d", fired, uid_, written_stream_, src_dev_, devno_, mem_stream_);
                    }
                    Unlock();
                }
                void Fire()
                {
                    Lock();
                    fired = true;
                    _info("Broadcast fire:%d mem:%lu written_stream:%d src_dev:%d dev:%d mem_stream:%d", fired, uid_, written_stream_, src_dev_, devno_, mem_stream_);
                    pthread_cond_broadcast(&cond_);
                    _info("Broadcast completed fire:%d mem:%lu written_stream:%d src_dev:%d dev:%d mem_stream:%d", fired, uid_, written_stream_, src_dev_, devno_, mem_stream_);
                    Unlock();
                }
                void set_mem(unsigned long uid, int written_stream, int src_dev, int mem_stream, int devno) {
                    uid_ = uid; written_stream_ = written_stream; src_dev_ = src_dev; mem_stream_ = mem_stream; devno_ = devno;
                }
                static void Wait(void *stream, int status, void *data) {
                    EventExchange *exchange = (EventExchange *)data;
                    _info("Waiting for stream:%p fired:%d", stream, exchange->fired);
                    exchange->Wait();
                    _info("Wait completed for stream:%p fired:%d", stream, exchange->fired);
                }
                static void Fire(void *stream, int status, void *data) {
                    EventExchange *exchange = (EventExchange *)data;
                    _info("Fire for event:%p fired:%d", stream, exchange->fired);
                    exchange->Fire();
                    _info("Fire complete for event:%p fired:%d", stream, exchange->fired);
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
                bool fired;
        };
    }
}
