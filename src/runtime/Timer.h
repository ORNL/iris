#ifndef BRISBANE_SRC_RT_TIMER_H
#define BRISBANE_SRC_RT_TIMER_H

#define BRISBANE_TIMER_MAX        128
#define BRISBANE_TIMER_APP        1
#define BRISBANE_TIMER_PLATFORM   2
#define BRISBANE_TIMER_INIT       3
#define BRISBANE_TIMER_KERNEL     4
#define BRISBANE_TIMER_H2D        5
#define BRISBANE_TIMER_D2H        6

#include <stddef.h>

namespace brisbane {
namespace rt {

class Timer {
public:
  Timer();
  ~Timer();

  double Now();
  size_t NowNS();
  double Start(int i);
  double Stop(int i);
  double Total(int i);

  size_t Inc(int i);
  size_t Inc(int i, size_t s);

private:
  double start_[BRISBANE_TIMER_MAX];
  double total_[BRISBANE_TIMER_MAX];
  size_t total_ull_[BRISBANE_TIMER_MAX];

  double boot_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_SRC_RT_TIMER_H */

