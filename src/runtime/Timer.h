#ifndef IRIS_SRC_RT_TIMER_H
#define IRIS_SRC_RT_TIMER_H

#define IRIS_TIMER_MAX        128
#define IRIS_TIMER_APP        1
#define IRIS_TIMER_PLATFORM   2
#define IRIS_TIMER_INIT       3
#define IRIS_TIMER_KERNEL     4
#define IRIS_TIMER_H2D        5
#define IRIS_TIMER_D2H        6

#include <stddef.h>

namespace iris {
namespace rt {

class Timer {
public:
  Timer();
  ~Timer();

  double Now();
  static double GetCurrentTime();
  size_t NowNS();
  double Start(int i);
  double Stop(int i);
  double Total(int i);

  size_t Inc(int i);
  size_t Inc(int i, size_t s);

private:
  double start_[IRIS_TIMER_MAX];
  double total_[IRIS_TIMER_MAX];
  size_t total_ull_[IRIS_TIMER_MAX];

  static double boot_;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_TIMER_H */

