#pragma once
#include <pthread.h>

namespace iris {
    namespace rt {
        class CPUEvent {
            public:
                CPUEvent() {
                    pthread_mutex_init(&mutex_, NULL);
                    pthread_cond_init(&cond_, NULL);
                }
                ~CPUEvent() {
                    pthread_mutex_destroy(&mutex_);
                    pthread_cond_destroy(&cond_);
                }
                void Wait()
                {
                    Lock();
                    pthread_cond_wait(&cond_, &mutex_);
                    Unlock();
                }
                void Record()
                {
                    pthread_cond_broadcast(&cond_);
                    //pthread_cond_signal(&cond_);
                }
                static void Record(CPUEvent *data) 
                {
                    data->Record();
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
