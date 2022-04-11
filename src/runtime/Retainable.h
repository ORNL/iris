#ifndef IRIS_SRC_RT_RETAINABLE_H
#define IRIS_SRC_RT_RETAINABLE_H

#include <iris/iris.h>
#include "Structs.h"
#include "Debug.h"
#include <stddef.h>

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
  }
  virtual ~Retainable() {}

  unsigned long uid() { return uid_; }
  struct_type* struct_obj() { return &struct_obj_; }

  void Retain() {
    int i;
    do i = ref_cnt_;
    while (!__sync_bool_compare_and_swap(&ref_cnt_, i, i + 1));
  }

  void Release() {
    int i;
    do i = ref_cnt_;
    while (!__sync_bool_compare_and_swap(&ref_cnt_, i, i - 1));
    if (i == 1) delete this;
  }

private:
  unsigned long uid_;
  int ref_cnt_;
  struct_type struct_obj_;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_RETAINABLE_H */

