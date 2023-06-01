#include "History.h"
#include "Debug.h"
#include "Command.h"
#include "Device.h"
#include "Platform.h"
#include "Kernel.h"
#include "Task.h"

namespace iris {
namespace rt {

History::History(Platform* platform) {
  platform_ = platform;
  ndevs_ = platform_->ndevs();
  for (int i = 0; i < ndevs_; i++) {
    t_kernel_[i] = 0.0;
    t_h2d_[i] = 0.0;
    t_d2d_[i] = 0.0;
    t_d2h_h2d_[i] = 0.0;
    t_d2o_[i] = 0.0;
    t_o2d_[i] = 0.0;
    t_d2h_[i] = 0.0;
    ta_kernel_[i] = 0.0;
    ta_h2d_[i] = 0.0;
    ta_d2d_[i] = 0.0;
    ta_d2h_h2d_[i] = 0.0;
    ta_d2o_[i] = 0.0;
    ta_o2d_[i] = 0.0;
    ta_d2h_[i] = 0.0;
    c_kernel_[i] = 0;
    c_h2d_[i] = 0;
    c_d2d_[i] = 0;
    c_d2h_h2d_[i] = 0;
    c_d2o_[i] = 0;
    c_o2d_[i] = 0;
    c_d2h_[i] = 0;
    size_h2d_[i] = 0;
    size_d2d_[i] = 0;
    size_d2h_h2d_[i] = 0;
    size_d2o_[i] = 0;
    size_o2d_[i] = 0;
    size_d2h_[i] = 0;
  }
}

History::~History() {
    _trace("Deleted history object");
}

void History::AddKernel(Command* cmd, Device* dev, double time) {
  //printf("Adding history for %s time:%f acc:%f\n", cmd->task()->name(), time, t_kernel_[dev->devno()]);
  return Add(cmd, dev, time, t_kernel_, ta_kernel_, c_kernel_);
}

void History::AddH2D(Command* cmd, Device* dev, double time, size_t s) {
  return Add(cmd, dev, time, t_h2d_, ta_h2d_, c_h2d_, s, size_h2d_);
}

void History::AddD2D(Command* cmd, Device* dev, double time, size_t s) {
  return Add(cmd, dev, time, t_d2d_, ta_d2d_, c_d2d_, s, size_d2d_);
}

void History::AddD2H_H2D(Command* cmd, Device* dev, double time, size_t s, bool count_incr) {
  return Add(cmd, dev, time, t_d2h_h2d_, ta_d2h_h2d_, c_d2h_h2d_, s, size_d2h_h2d_, count_incr);
}

void History::AddO2D(Command* cmd, Device* dev, double time, size_t s) {
  return Add(cmd, dev, time, t_o2d_, ta_o2d_, c_o2d_, s, size_o2d_);
}

void History::AddD2O(Command* cmd, Device* dev, double time, size_t s) {
  return Add(cmd, dev, time, t_d2o_, ta_d2o_, c_d2o_, s, size_d2o_);
}

void History::AddD2H(Command* cmd, Device* dev, double time, size_t s) {
  return Add(cmd, dev, time, t_d2h_, ta_d2h_, c_d2h_, s, size_d2h_);
}

void History::Add(Command* cmd, Device* dev, double time, double* t, double *ta, size_t* c) {
  //Kernel* kernel = cmd->kernel();
  int devno = dev->devno();
  size_t cnt = c[devno];
  t[devno] += time;
  c[devno]++;
  ta[devno] = (ta[devno] * cnt + time) / (cnt + 1);
}

void History::Add(Command* cmd, Device* dev, double time, double* t, double *ta, size_t* c, size_t s, size_t *sp, bool count_incr) {
  //Kernel* kernel = cmd->kernel();
  int devno = dev->devno();
  size_t cnt = c[devno];
  t[devno] += time;
  if (count_incr) c[devno]++;
  sp[devno] += s;
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

double History::t_d2d() {
  return total(t_d2d_);
}

double History::t_d2h_h2d() {
  return total(t_d2h_h2d_);
}

double History::t_d2o() {
  return total(t_d2o_);
}

double History::t_o2d() {
  return total(t_o2d_);
}


size_t History::c_kernel() {
  return total(c_kernel_);
}

size_t History::c_h2d() {
  return total(c_h2d_);
}

size_t History::c_d2d() {
  return total(c_d2d_);
}

size_t History::c_d2h_h2d() {
  return total(c_d2h_h2d_);
}

size_t History::c_d2o() {
  return total(c_d2o_);
}

size_t History::c_o2d() {
  return total(c_o2d_);
}

size_t History::c_d2h() {
  return total(c_d2h_);
}

size_t History::size_h2d() {
  return total(size_h2d_);
}

size_t History::size_d2h() {
  return total(size_d2h_);
}

size_t History::size_d2h_h2d() {
  return total(size_d2h_h2d_);
}

size_t History::size_d2d() {
  return total(size_d2d_);
}

size_t History::size_d2o() {
  return total(size_d2o_);
}

size_t History::size_o2d() {
  return total(size_o2d_);
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
} /* namespace iris */
