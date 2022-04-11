#ifndef IRIS_SRC_RT_FILTER_H
#define IRIS_SRC_RT_FILTER_H

namespace iris {
namespace rt {

class Task;

class Filter {
public:
  Filter() {}
  virtual ~Filter() {}

  virtual int Execute(Task* task) = 0;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_FILTER_H */

