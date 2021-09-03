#include "History.h"
#include "Debug.h"
#include "Command.h"
#include "Device.h"
#include "Platform.h"
#include "Kernel.h"

namespace brisbane {
namespace rt {

History::History(Kernel* kernel) {
  kernel_ = kernel;
  platform_ = kernel_->platform();
  ndevs_ = platform_->ndevs();
  for (int i = 0; i < ndevs_; i++) {
    t_kernel_[i] = 0.0;
    t_h2d_[i] = 0.0;
    t_d2h_[i] = 0.0;
    ta_kernel_[i] = 0.0;
    ta_h2d_[i] = 0.0;
    ta_d2h_[i] = 0.0;
    c_kernel_[i] = 0;
    c_h2d_[i] = 0;
    c_d2h_[i] = 0;
  }
}

History::~History() {
}

void History::AddKernel(Command* cmd, Device* dev, double time) {
  return Add(cmd, dev, time, t_kernel_, ta_kernel_, c_kernel_);
}

void History::AddH2D(Command* cmd, Device* dev, double time) {
  return Add(cmd, dev, time, t_h2d_, ta_h2d_, c_h2d_);
}

void History::AddD2H(Command* cmd, Device* dev, double time) {
  return Add(cmd, dev, time, t_d2h_, ta_d2h_, c_d2h_);
}

void History::Add(Command* cmd, Device* dev, double time, double* t, double *ta, size_t* c) {
  Kernel* kernel = cmd->kernel();
  int devno = dev->devno();
  size_t cnt = c[devno];
  t[devno] += time;
  c[devno]++;
  ta[devno] = (ta[devno] * cnt + time) / (cnt + 1);
}

Device* History::OptimalDevice(Task* task) {
  for (int i = 0; i < ndevs_; i++) {
    if (c_kernel_[i] == 0) return platform_->device(i);
  }

  double min_time = ta_kernel_[0];
  int min_dev = 0;
  for (int i = 0; i < ndevs_; i++) {
    if (ta_kernel_[i] < min_time) {
      min_time = ta_kernel_[i];
      min_dev = i;
    }
  }
  return platform_->device(min_dev);
}

double History::t_kernel() {
  return total(t_kernel_);
}

double History::t_h2d() {
  return total(t_h2d_);
}

double History::t_d2h() {
  return total(t_d2h_);
}


size_t History::c_kernel() {
  return total(c_kernel_);
}

size_t History::c_h2d() {
  return total(c_h2d_);
}

size_t History::c_d2h() {
  return total(c_d2h_);
}

double History::total(double* d) {
  double t = 0.0;
  for (int i = 0; i < ndevs_; i++) t += d[i];
  return t;
}

size_t History::total(size_t* s) {
  size_t t = 0.0;
  for (int i = 0; i < ndevs_; i++) t += s[i];
  return t;
}

} /* namespace rt */
} /* namespace brisbane */
