#ifndef IRIS_SRC_RT_POLICY_H
#define IRIS_SRC_RT_POLICY_H

#define REGISTER_CUSTOM_POLICY(class_name, name)              \
  iris::rt::class_name name;                              \
  extern "C" void* name ## _instance() { return (void*) &name; }

namespace iris {
namespace rt {

class Device;
class Scheduler;
class Task;

class Policy {
public:
  Policy();
  virtual ~Policy();

  virtual void Init(void* arg) {}
  virtual bool IsKernelSupported(Task *task, Device *dev);
  virtual void GetDevices(Task* task, Device** devs, int* ndevs) = 0;
  void SetScheduler(Scheduler* scheduler);

protected:
  Device** devices() const { return  devs_; }
  Device* device(int i) const { return  devs_[i]; }
  int ndevices() const { return ndevs_; }

protected:
  Scheduler* scheduler_;
  Device** devs_;
  int ndevs_;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_POLICY_H */

