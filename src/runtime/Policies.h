#ifndef IRIS_SRC_RT_POLICIES_H
#define IRIS_SRC_RT_POLICIES_H

#include <map>
#include <string>

namespace iris {
namespace rt {

class Policy;
class LoaderPolicy;
class Scheduler;

class Policies {
public:
  Policies(Scheduler* scheduler);
  ~Policies();

  Policy* GetPolicy(int brs_policy, char* opt);

  int Register(const char* lib, const char* name, void* params);

private:
  Scheduler* scheduler_;

  Policy* policy_all_;
  Policy* policy_any_;
  Policy* policy_data_;
  Policy* policy_default_;
  Policy* policy_depend_;
  Policy* policy_device_;
  Policy* policy_profile_;
  Policy* policy_random_;
  Policy* policy_roundrobin_;

  std::map<std::string, LoaderPolicy*> policy_customs_;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_POLICIES_H */

