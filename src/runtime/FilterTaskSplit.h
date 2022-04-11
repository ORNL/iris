#ifndef IRIS_SRC_RT_FILTER_TASK_SPLIT_H
#define IRIS_SRC_RT_FILTER_TASK_SPLIT_H

#include "Filter.h"

namespace iris {
namespace rt {

class Polyhedral;
class Platform;

class FilterTaskSplit : public Filter {
public:
  FilterTaskSplit(Polyhedral* polyhedral, Platform* platform);
  virtual ~FilterTaskSplit();

  virtual int Execute(Task* task);

private:
  Polyhedral* polyhedral_;
  Platform* platform_;
};

} /* namespace rt */
} /* namespace iris */


#endif /* IRIS_SRC_RT_FILTER_TASK_SPLIT_H */

