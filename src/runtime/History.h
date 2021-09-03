#ifndef BRISBANE_SRC_RT_HISTORY_H
#define BRISBANE_SRC_RT_HISTORY_H

#include "Config.h"
#include <map>
#include <string>

namespace brisbane {
namespace rt {

class Command;
class Device;
class Kernel;
class Platform;
class Task;

class History {
public:
  History(Kernel* kernel);
  ~History();

  void AddKernel(Command* cmd, Device* dev, double time);
  void AddH2D(Command* cmd, Device* dev, double time);
  void AddD2H(Command* cmd, Device* dev, double time);
  Device* OptimalDevice(Task* task);

  double t_kernel();
  double t_h2d();
  double t_d2h();
  size_t c_kernel();
  size_t c_h2d();
  size_t c_d2h();

private:
  void Add(Command* cmd, Device* dev, double time, double* t, double* ta, size_t* c);
  double total(double* d);
  size_t total(size_t* s);

private:
  Kernel* kernel_;
  Platform* platform_;
  int ndevs_;

  double t_kernel_[BRISBANE_MAX_NDEVS];
  double t_h2d_[BRISBANE_MAX_NDEVS];
  double t_d2h_[BRISBANE_MAX_NDEVS];

  double ta_kernel_[BRISBANE_MAX_NDEVS];
  double ta_h2d_[BRISBANE_MAX_NDEVS];
  double ta_d2h_[BRISBANE_MAX_NDEVS];

  size_t c_kernel_[BRISBANE_MAX_NDEVS];
  size_t c_h2d_[BRISBANE_MAX_NDEVS];
  size_t c_d2h_[BRISBANE_MAX_NDEVS];
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_SRC_RT_HISTORY_H */
