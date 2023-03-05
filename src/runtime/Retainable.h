#ifndef IRIS_SRC_RT_RETAINABLE_H
#define IRIS_SRC_RT_RETAINABLE_H

#include <iris/iris.h>
#include "Structs.h"
#include "Debug.h"
#include <stddef.h>
#include "Platform.h"
#include "pthread.h"

extern unsigned long iris_create_new_uid();

namespace iris {
namespace rt {

template <typename struct_type, class class_type>
class Retainable {
public:
  Retainable() {
    uid_ = iris_create_new_uid();
    struct_obj_.class_obj = (class_type*) this;
    ref_cnt_ = 1;
    is_release_ = true;
    pthread_mutex_init(&delete_lock_, NULL);
  }
  bool IsRelease() { return is_release_; }
  void DisableRelease() { is_release_ = false; }
  void EnableRelease() { is_release_ = true; }
  virtual ~Retainable() { pthread_mutex_destroy(&delete_lock_); }

  unsigned long uid() { return uid_; }
  struct_type* struct_obj() { return &struct_obj_; }

  void Retain() {
    int i;
    do i = ref_cnt_;
    while (!__sync_bool_compare_and_swap(&ref_cnt_, i, i + 1));
  }

  void ForceRelease() {
    Platform *platform = Platform::GetPlatform();
    pthread_mutex_lock(&delete_lock_);
    if (!platform->track().IsObjectExists(this)) return;
    if (!platform->track().IsObjectExists(struct_obj())) return;
    platform->track().UntrackObject(this);
    platform->track().UntrackObject(struct_obj());
    pthread_mutex_unlock(&delete_lock_);
    delete this;
  }

  void Release() {
    int i;
    do i = ref_cnt_;
    while (!__sync_bool_compare_and_swap(&ref_cnt_, i, i - 1));
    if (ref_cnt_ <= 1 && is_release_) ForceRelease();
  }
  int ref_cnt() { return ref_cnt_; }

private:
  unsigned long uid_;
  int ref_cnt_;
  bool is_release_;
  struct_type struct_obj_;
  pthread_mutex_t delete_lock_;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_RETAINABLE_H */

