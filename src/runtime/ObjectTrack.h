#pragma once

#include <map>
#include <pthread.h>
#include "Debug.h"

using namespace std;
typedef void (*ObjectTrackCallBack)(void *data);
typedef void (*SafeAssignmentCallBack)(void *data, void *lhs, void *rhs);
namespace iris {
    namespace rt {
        class ObjectTrack
        {
            public:
                ObjectTrack( ) {
                    //freed_objects = 0;
                    pthread_mutex_init(&track_lock_, NULL);
                    _debug2("Initialized track_lock:%p", track_lock_);
                }
                virtual ~ObjectTrack() {
                    pthread_mutex_destroy(&track_lock_);
                }
                virtual bool IsObjectExists(unsigned long uid) { 
                    bool flag = false;
                    void *obj = NULL;
                    if (allocated_objects_.find(uid) != allocated_objects_.end())  {
                        obj = allocated_objects_[uid];
                        flag = (obj != NULL);
                    }
                    //printf("object:%lu: %p exists flag:%d\n", uid, obj, flag);
                    _debug2("object:%lu: %p exists flag:%d", uid, obj, flag);
                    return flag;
                }
                virtual bool CallBackIfObjectExists(unsigned long uid, SafeAssignmentCallBack callbackfn, void *lhs, void *rhs) {
                    bool object_exists = true;
                    _debug2("Waiting for lock uid:%lu track_lock:%p", uid, &track_lock_);
                    pthread_mutex_lock(&track_lock_);
                    _debug2("Acquired lock uid:%lu track_lock:%p", uid, &track_lock_);
                    if (allocated_objects_.find(uid) != allocated_objects_.end() &&
                            allocated_objects_[uid] != NULL)
                        callbackfn(allocated_objects_[uid], lhs, rhs);
                    else
                        object_exists = false;
                    pthread_mutex_unlock(&track_lock_);
                    _debug2("Released lock uid:%lu track_lock:%p", uid, &track_lock_);
                    return object_exists;
                }
                virtual bool CallBackIfObjectExists(unsigned long uid, ObjectTrackCallBack callbackfn) {
                    bool object_exists = true;
                    _debug2("Waiting for lock uid:%lu track_lock:%p", uid, &track_lock_);
                    pthread_mutex_lock(&track_lock_);
                    _debug2("Acquired lock uid:%lu track_lock:%p", uid, &track_lock_);
                    if (allocated_objects_.find(uid) != allocated_objects_.end() &&
                            allocated_objects_[uid] != NULL)
                        callbackfn(allocated_objects_[uid]);
                    else
                        object_exists = false;
                    pthread_mutex_unlock(&track_lock_);
                    _debug2("Released lock uid:%lu track_lock:%p", uid, &track_lock_);
                    return object_exists;
                }
                virtual void *GetObject(unsigned long uid) {
                    void *obj = NULL;
                    //_debug2("Waiting for lock uid:%lu", uid);
                    _debug2("Waiting for lock uid:%lu track_lock:%p", uid, &track_lock_);
                    pthread_mutex_lock(&track_lock_);
                    //_debug2("Acquired lock uid:%lu", uid);
                    _debug2("Acquired lock uid:%lu track_lock:%p", uid, &track_lock_);
                    if (allocated_objects_.find(uid) != allocated_objects_.end())
                        obj = allocated_objects_[uid];
                    pthread_mutex_unlock(&track_lock_);
                    _debug2("Released lock uid:%lu track_lock:%p", uid, &track_lock_);
                    //_debug2("Released lock uid:%lu", uid);
                    return obj;
                }
                virtual void UntrackObjectNoLock(void *p, unsigned long uid) {
                    //_debug2("Waiting for lock uid:%lu", uid);
                    //pthread_mutex_lock(&track_lock_);
                    //_debug2("Acquired lock uid:%lu", uid);
                    allocated_objects_[uid] = NULL;
                    //_debug2("Untracking object: %lu: %p", uid, p);
                    //printf("Untracking object: %lu: %p\n", uid, p);
                    //pthread_mutex_unlock(&track_lock_);
                    //_debug2("Released lock uid:%lu", uid);
                    //freed_objects+=1; 
                }
                virtual void UntrackObject(void *p, unsigned long uid) {
                    _debug2("Waiting for lock uid:%lu track_lock:%p", uid, &track_lock_);
                    pthread_mutex_lock(&track_lock_);
                    _debug2("Acquired lock uid:%lu track_lock:%p", uid, &track_lock_);
                    allocated_objects_[uid] = NULL;
                    _debug2("Untracking object: %lu: %p track_lock:%p", uid, p, &track_lock_);
                    //printf("Untracking object: %lu: %p\n", uid, p);
                    pthread_mutex_unlock(&track_lock_);
                    _debug2("Released lock uid:%lu track_lock:%p", uid, &track_lock_);
                    //freed_objects+=1; 
                }
                virtual void TrackObject(void *p, unsigned long uid) {
                    _debug2("Waiting for lock uid:%lu track_lock:%p", uid, &track_lock_);
                    pthread_mutex_lock(&track_lock_);
                    _debug2("Acquired lock uid:%lu track_lock:%p", uid, &track_lock_);
                    if (allocated_objects_.find(uid) != allocated_objects_.end()) 
                        allocated_objects_[uid] = p;
                    else
                        allocated_objects_.insert(pair<unsigned long, void *>(uid, p));
                    pthread_mutex_unlock(&track_lock_);
                    _debug2("Released lock uid:%lu track_lock:%p", uid, &track_lock_);
#if 0
                    if (freed_objects > 2048) {
                        Clear();
                        freed_objects = 0;
                    }
#endif
                }
                virtual void Clear() {
                    //allocated_objects_.clear();
                    _debug2("Waiting for lock");
                    pthread_mutex_lock(&track_lock_);
                    _debug2("Acquired lock ");
                    for (auto i = allocated_objects_.begin(), 
                            last = allocated_objects_.end(); i != last; ) {
                        if ((*i).second == NULL) i = allocated_objects_.erase(i);
                        else ++i;
                    }
                    pthread_mutex_unlock(&track_lock_);
                    _debug2("Released lock ");
                }
                void Print(const char *data="Task track") {
                    printf("%s\n", data);
                    for ( auto & z : allocated_objects_) {
                        if (z.second != NULL)
                            printf("%lu:%p ", z.first, z.second);
                    }
                    printf("\n");
                }

            private:
                //int freed_objects;
                std::map<unsigned long, void *> allocated_objects_;
                pthread_mutex_t track_lock_;
            protected:
        };
    }
}


