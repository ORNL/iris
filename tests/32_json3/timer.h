#ifndef DAGGER_TIMER_H
#define DAGGER_TIMER_H

#include <time.h>

double now() {
  static double boot = 0.0;
  struct timespec t;
  clock_gettime(CLOCK_REALTIME, &t);
  if (boot == 0.0) boot = t.tv_sec + 1.e-9 * t.tv_nsec;
  return t.tv_sec + 1.e-9 * t.tv_nsec - boot;
}

#endif /* end of DAGGER_TIMER_H */

