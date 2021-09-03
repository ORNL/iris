#ifndef BRISBANE_SRC_RT_FILTER_H
#define BRISBANE_SRC_RT_FILTER_H

namespace brisbane {
namespace rt {

class Task;

class Filter {
public:
  Filter() {}
  virtual ~Filter() {}

  virtual int Execute(Task* task) = 0;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_SRC_RT_FILTER_H */

