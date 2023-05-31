#pragma once

#include <map>
#include <pthread.h>
#include "Debug.h"

using namespace std;

namespace iris {
    namespace rt {
        class ObjectTrack
        {
            public:
                ObjectTrack( ) {
                    //freed_objects = 0;
                    pthread_mutex_init(&track_lock_, NULL);
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
                    _trace("object:%lu: %p exists flag:%d", uid, obj, flag);
                    return flag;
                }
                virtual void *GetObject(unsigned long uid) {
                    if (allocated_objects_.find(uid) != allocated_objects_.end())
                        return allocated_objects_[uid];
                    return NULL;
                }
                virtual void UntrackObject(void *p, unsigned long uid) {
                    pthread_mutex_lock(&track_lock_);
                    allocated_objects_[uid] = 0;
                    _trace("Untracking object: %lu: %p", uid, p);
                    //printf("Untracking object: %lu: %p\n", uid, p);
                    pthread_mutex_unlock(&track_lock_);
                    //freed_objects+=1; 
                }
                virtual void TrackObject(void *p, unsigned long uid) {
                    pthread_mutex_lock(&track_lock_);
                    if (allocated_objects_.find(uid) != allocated_objects_.end()) 
                        allocated_objects_[uid] = p;
                    else
                        allocated_objects_.insert(pair<unsigned long, void *>(uid, p));
                    pthread_mutex_unlock(&track_lock_);
#if 0
                    if (freed_objects > 2048) {
                        _trace("Freeing uncleared objects size before: %lu", allocated_objects_.size());
                        for (auto i = allocated_objects_.begin(), last = allocated_objects_.end(); i != last; ) {
                            if (! (*i).second) i = allocated_objects_.erase(i);
                            else ++i;
                        }
                        _trace("Size after free: %lu", allocated_objects_.size());
                        freed_objects = 0;
                    }
#endif
                }
                virtual void Clear() {
                    allocated_objects_.clear();
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
            protected:
                pthread_mutex_t track_lock_;
        };
    }
}


