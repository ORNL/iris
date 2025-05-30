#pragma once
#include <pthread.h>
#include "EventExchange.h"
#include <stack>

namespace iris {
    namespace rt {
        template <typename CoreClass>
            class AsyncData {
                public:
                    AsyncData() { 
                        core_=NULL; devno_ = 0; 
                        exchange_ = NULL;
                        completed_flag_ = false;
                        //completion_event_ = NULL;
                        proactive_transfer_ = false;
                        write_streams_ = -1;
                    }
                    ~AsyncData() {
                        pthread_mutex_destroy(&mutex_);
                        if (exchange_ != NULL) delete exchange_;
                    }
                    void Init(CoreClass *core, int devno) {
                        devno_ = devno; core_ = core;
                        //completion_event_ = NULL;
                        completed_flag_ = false;
                        exchange_ = new EventExchange();
                        write_streams_ = -1;
                        proactive_transfer_ = false;
                        pthread_mutex_init(&mutex_, NULL);
                    }
                    EventExchange *exchange() { 
                        return exchange_; 
                    }
                    CoreClass *core() { return core_; }
                    int devno() { return devno_; }
                    void set_devno(int dev) { devno_ = dev; }
                    //bool IsCompleted() { return completed_flag_; }
                    //void DisableCompleted() { completed_flag_ = false; }
                    //void EnableCompleted() { completed_flag_ = true; }
                    void DisableProactive() { proactive_transfer_ = false; }
                    void EnableProactive()  { proactive_transfer_ = true;  }
                    bool IsProactive() { return proactive_transfer_; }
                    stack<void *> & GetCompletionStack() {  return completion_event_; }
                    void *GetCompletionEvent() { 
                        if (completion_event_.size()==0) 
                            return NULL; 
                        return completion_event_.top(); 
                    }
                    void **GetCompletionEventPtr(bool new_entry=false) 
                    { 
                        if (new_entry || completion_event_.size() == 0)
                            completion_event_.push(NULL);
                        return &completion_event_.top(); 
                    }
                    void SetWriteStream(int stream) { write_streams_ = stream; }
                    int  GetWriteStream()   { return write_streams_; }
                    vector<void *> & GetWaitEvents() { return waiting_events_; }
                    void ClearAndAddWaitEvent(void *event) {
                        ClearWaitEvents();
                        AddWaitEvent(event);
                    }
                    void AddWaitEvent(void *event) {
                        waiting_events_.push_back(event);
                    }
                    void ClearWaitEvents() {
                        waiting_events_.clear();
                    }
                    void ClearDevice() {
                         devno_ = -1;
                    }
                    void Clear() {
                         if (exchange_ != NULL) exchange_->Clear();
                         write_streams_ = -1;
                    }
                    void Unlock() {
                        pthread_mutex_unlock(&mutex_);
                    }
                    void Lock() {
                        pthread_mutex_lock(&mutex_);
                    }
                private:
                    CoreClass *core_;
                    EventExchange *exchange_;
                    stack<void *> completion_event_;
                    pthread_mutex_t mutex_;
                    vector<void *> waiting_events_;
                    int devno_;
                    int  write_streams_;
                    bool proactive_transfer_;
                    bool completed_flag_;
            };
    }
}
