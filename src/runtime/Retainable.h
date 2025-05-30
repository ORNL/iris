#ifndef IRIS_SRC_RT_RETAINABLE_H
#define IRIS_SRC_RT_RETAINABLE_H

#include <iris/iris.h>
#include "Structs.h"
#include "Debug.h"
#include <stddef.h>
#include "Platform.h"
#include "pthread.h"
#include "ObjectTrack.h"

extern unsigned long iris_create_new_uid();

namespace iris {
namespace rt {

enum RetainMode { CREATE_MODE=0, PROCESS_MODE=1 };
template <typename struct_type, class class_type>
class Retainable {
public:
  Retainable() {
    uid_ = iris_create_new_uid();
    ref_cnt_ = 1;
    struct_obj_.uid = uid_;
    struct_obj_.class_obj = (class_type*) this;
    is_release_ = true;
    track_ = NULL;
    mode_ = CREATE_MODE;
    pthread_mutex_init(&delete_lock_, NULL);
  }
  Retainable(struct_type obj) {
    uid_ = iris_create_new_uid();
    struct_obj_.uid = uid_;
    struct_obj_.class_obj = (class_type*) this;
    obj.uid = uid_;
    obj.class_obj = (class_type*) this;
    track_ = NULL;
    obj->uid = uid_;
    ref_cnt_ = 1;
    is_release_ = true;
    mode_ = CREATE_MODE;
    pthread_mutex_init(&delete_lock_, NULL);
  }
  void SetStructObject(struct_type *obj)
  {
      obj->uid = uid_;
      obj->class_obj = (class_type*) this;
  }
  void ChangeToProcessMode() {
    mode_ = PROCESS_MODE;
  }
  bool IsRelease() { return is_release_; }
  void DisableRelease() { is_release_ = false; }
  void EnableRelease() { is_release_ = true; }
  virtual ~Retainable() { pthread_mutex_destroy(&delete_lock_); }

  unsigned long uid() { return uid_; }
  struct_type *struct_obj() { return &struct_obj_; }

  static void StaticRetain(void *data) {
    Retainable *obj = (Retainable *)data;
    obj->Retain();
  }

  void Retain() {
    int i;
    do i = ref_cnt_;
    while (!__sync_bool_compare_and_swap(&ref_cnt_, i, i + 1));
    //printf("Retain: id:%lu ref_cnt:%d\n", uid_, i+1);
  }
  void ForceRelease(bool check=false) {
    Platform *platform = Platform::GetPlatform();
    pthread_mutex_lock(&delete_lock_);
    if (track_!=NULL && !track_->IsObjectExists(uid_)) {
        pthread_mutex_unlock(&delete_lock_);
        return;
    }
    bool ref_cnt_bound_check = (mode_ == CREATE_MODE) ? (ref_cnt_ == 0) : (ref_cnt_ == 1);
    //printf("force release called id:%lu ref_cnt:%d\n", uid_, ref_cnt_);
    //if (!struct_obj()) return;
    pthread_mutex_unlock(&delete_lock_);
    //void *obj = track_->GetObject(uid_);
    if (!check || (ref_cnt_bound_check && is_release_)) {
        track_->UntrackObjectNoLock(this, uid());
        delete this;
    }
  }
  static void StaticForceRelease(void *data) {
    Retainable *obj = (Retainable *)data;
    obj->ForceRelease(true);   
  }

  ObjectTrack *track() { return track_; }
  int retain_mode() { return mode_; }
  int Release() {
    int i;
    //printf("from id:%lu ref_cnt_:%d\n", uid_, ref_cnt_);
    do i = ref_cnt_;
    while (!__sync_bool_compare_and_swap(&ref_cnt_, i, i - 1));
    //printf("Release id:%lu ref_cnt_:%d %d mode:%d\n", uid_, ref_cnt_, i-1, mode_);
    // ref_cnt should be derived from local variable i
    int ref_cnt = i-1;
    //if (ref_cnt < 1) printf("problem from id:%lu ref_cnt_:%d\n", uid_, ref_cnt_);
    //assert(ref_cnt >= 1 && "ref_cnt should be more than 1");
    bool ref_cnt_bound_check = (mode_ == CREATE_MODE) ? (ref_cnt == 0) : (ref_cnt == 1);
    if (ref_cnt_bound_check && is_release_) 
        track_->CallBackIfObjectExists(uid_, Retainable::StaticForceRelease);
        //ForceRelease(true);
    return ref_cnt;
  }
  void set_object_track(ObjectTrack *track) { track_ = track; }
  int ref_cnt() { return ref_cnt_; }

private:
  unsigned long uid_;
  int ref_cnt_;
  RetainMode mode_;
  bool is_release_;
  struct_type struct_obj_;
  ObjectTrack *track_;
  pthread_mutex_t delete_lock_;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_RETAINABLE_H */

