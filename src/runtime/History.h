#ifndef IRIS_SRC_RT_HISTORY_H
#define IRIS_SRC_RT_HISTORY_H

#include "Config.h"
#include <map>
#include <string>

namespace iris {
namespace rt {

class Command;
class Device;
class Kernel;
class Platform;
class Task;

struct TaskProfile
{
    uint32_t task_id;  
    uint32_t device_id;  
    double start;
    double end;
};
struct DataObjectProfile {
  uint32_t  task_id_;
  uint32_t  mem_id_;
  uint32_t datatransfer_type_;
  uint32_t from_dev_id_;
  uint32_t dev_id_;
  double start_;
  double end_;
};
class History {
public:
  History(Platform* platform);
  ~History();

  void AddKernel(Command* cmd, Device* dev, double time);
  void AddH2D(Command* cmd, Device* dev, double time, size_t s);
  void AddD2D(Command* cmd, Device* dev, double time, size_t s);
  void AddD2H_H2D(Command* cmd, Device* dev, double time, size_t s, bool count_incr=true);
  void AddD2O(Command* cmd, Device* dev, double time, size_t s);
  void AddO2D(Command* cmd, Device* dev, double time, size_t s);
  void AddD2H(Command* cmd, Device* dev, double time, size_t s);
  Device* OptimalDevice(Task* task);

  double t_kernel();
  double t_h2d();
  double t_d2h();
  double t_d2d();
  double t_d2h_h2d();
  double t_d2o();
  double t_o2d();
  size_t c_kernel();
  size_t c_h2d();
  size_t c_d2h();
  size_t c_d2d();
  size_t c_d2h_h2d();
  size_t c_d2o();
  size_t c_o2d();
  size_t size_h2d();
  size_t size_d2h();
  size_t size_d2d();
  size_t size_d2h_h2d();
  size_t size_d2o();
  size_t size_o2d();

private:
  void Add(Command* cmd, Device* dev, double time, double* t, double* ta, size_t* c);
  void Add(Command* cmd, Device* dev, double time, double* t, double* ta, size_t* c, size_t s, size_t *sp, bool count_incr=true);
  double total(double* d);
  size_t total(size_t* s);

private:
  Platform* platform_;
  int ndevs_;

  double t_kernel_[IRIS_MAX_NDEVS];
  double t_h2d_[IRIS_MAX_NDEVS];
  double t_d2d_[IRIS_MAX_NDEVS];
  double t_d2h_h2d_[IRIS_MAX_NDEVS];
  double t_d2o_[IRIS_MAX_NDEVS];
  double t_o2d_[IRIS_MAX_NDEVS];
  double t_d2h_[IRIS_MAX_NDEVS];

  double ta_kernel_[IRIS_MAX_NDEVS];
  double ta_h2d_[IRIS_MAX_NDEVS];
  double ta_d2d_[IRIS_MAX_NDEVS];
  double ta_d2h_h2d_[IRIS_MAX_NDEVS];
  double ta_d2o_[IRIS_MAX_NDEVS];
  double ta_o2d_[IRIS_MAX_NDEVS];
  double ta_d2h_[IRIS_MAX_NDEVS];

  size_t c_kernel_[IRIS_MAX_NDEVS];
  size_t c_h2d_[IRIS_MAX_NDEVS];
  size_t c_d2d_[IRIS_MAX_NDEVS];
  size_t c_d2h_h2d_[IRIS_MAX_NDEVS];
  size_t c_d2o_[IRIS_MAX_NDEVS];
  size_t c_o2d_[IRIS_MAX_NDEVS];
  size_t c_d2h_[IRIS_MAX_NDEVS];
  
  size_t size_h2d_[IRIS_MAX_NDEVS];
  size_t size_d2d_[IRIS_MAX_NDEVS];
  size_t size_d2h_h2d_[IRIS_MAX_NDEVS];
  size_t size_d2o_[IRIS_MAX_NDEVS];
  size_t size_o2d_[IRIS_MAX_NDEVS];
  size_t size_d2h_[IRIS_MAX_NDEVS];
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_HISTORY_H */
