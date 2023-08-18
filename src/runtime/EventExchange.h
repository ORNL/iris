#pragma once
#include <pthread.h>

namespace iris {
    namespace rt {
        class EventExchange {
            public:
                EventExchange() {
                    pthread_mutex_init(&mutex_, NULL);
                    pthread_cond_init(&cond_, NULL);
                }
                ~EventExchange() {
                    pthread_mutex_destroy(&mutex_);
                    pthread_cond_destroy(&cond_);
                }
                void Wait()
                {
                    Lock();
                    pthread_cond_wait(&cond_, &mutex_);
                    Unlock();
                }
                void Fire()
                {
                    pthread_cond_broadcast(&cond_);
                }
                static void Wait(void *stream, int status, void *data) {
                    EventExchange *exchange = (EventExchange *)data;
                    exchange->Wait();
                }
                static void Fire(void *stream, int status, void *data) {
                    EventExchange *exchange = (EventExchange *)data;
                    exchange->Fire();
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
