#include <iris/iris.h>
#include <iris/rt/Policy.h>
#include <iris/rt/Debug.h>
#include <iris/rt/Device.h>
#include <iris/rt/Command.h>
#include <iris/rt/Device.h>
#include <iris/rt/Task.h>

#include "aiwc_utils.h"
#include <csignal>//TODO: delete when done
#include <iostream>

namespace iris {
namespace rt {

int task_hook_pre(void* task) {
  Task* t = (Task*) task;
  printf("[%s:%d] enter task[%lu]\n", __FILE__, __LINE__, t->uid());
  return 0;
}

int task_hook_post(void* task) {
  Task* t = (Task*) task;
  printf("[%s:%d] exit task[%lu] time[%lf]\n", __FILE__, __LINE__, t->uid(), t->time());
  return 0;
}

int cmd_hook_pre(void* cmd) {
  Command* c = (Command*) cmd;
  printf("[%s:%d] enter cmd[%x]\n", __FILE__, __LINE__, c->type());
  return 0;
}

int cmd_hook_post(void* cmd) {
  Command* c = (Command*) cmd;
  if (c->type() == IRIS_CMD_KERNEL)
    printf("[%s:%d] exit cmd[%x] policy[%s] kernel[%s] gws[%lu][%lu][%lu] dev[%d][%s] time[%lf]\n", __FILE__, __LINE__, c->type(), c->task()->opt(), c->kernel()->name(), c->gws(0), c->gws(1), c->gws(2), c->task()->dev()->devno(), c->task()->dev()->name(), c->time());
  else printf("[%s:%d] exit cmd[%x] time[%lf]\n", __FILE__, __LINE__, c->type(), c->time());
  return 0;
}

class AIWCPolicy: public Policy {
public:
  AIWCPolicy() {
    iris_register_hooks_task(task_hook_pre, task_hook_post);
    iris_register_hooks_command(cmd_hook_pre, cmd_hook_post);
  }
  virtual ~AIWCPolicy() {}
  virtual void Init(void* params) {
    //threshold_ = (size_t) params;
  }
  virtual void GetDevices(Task* task, Device** devs, int* ndevs) {
    /*
    Command* cmd = task->cmd_kernel();
    size_t* gws = cmd->gws();
    size_t total_work_items = gws[0] * gws[1] * gws[2];
    int target_dev = total_work_items > threshold_ ? iris_gpu : iris_cpu;
    int devid = 0;
    for (int i = 0; i < ndevices(); i++)
      if (device(i)->type() & target_dev) devs[devid++] = device(i);
    *ndevs = devid;
  }

  size_t threshold_;
  */
      //auto blah = task->kernel();
      using namespace plugin;
      //first locate the AIWC device -- it has the kernel digest!
      Device* aiwc_dev = NULL;
      for (int i = 0; i < ndevs_; i++){
          if(AIWC_Utils::IsAIWCDevice(devs_[i]->name(),devs_[i]->vendor())){
              aiwc_dev = devs_[i];
          }
      }
      if (aiwc_dev == NULL){
        std::cerr << "No AIWC device found in policy " << __FILE__ << std::endl;
        devs[0] = devs_[rand() % ndevs_];
        *ndevs = 1;
        return; //TODO: fail gracefully, perhaps swap to another policy
      }

      //load the kernel source file (again)
      char* src = NULL;
      size_t srclen = 0;
      AIWC_Utils::ReadFile((char*)"./kernel.cl", &src, &srclen);
      if (srclen == 0){
          std::cerr << "No kernel file to perform prediction on." << __FILE__ << std::endl;
          return;
      }

      //compute digest/hash of the source
      char* aiwc_metric_name = AIWC_Utils::ComputeDigest(src);

      if(AIWC_Utils::HaveMetrics(aiwc_metric_name)==false) {
          const char* storage_path = AIWC_Utils::MetricLocation(aiwc_metric_name);
          AIWC_Utils::SetEnvironment("OCLGRIND_WORKLOAD_CHARACTERISATION_OUTPUT_PATH", storage_path);
          AIWC_Utils::SetEnvironment("OCLGRIND_WORKLOAD_CHARACTERISATION", "1");
          delete storage_path;
          //NOTE: we must recreate the OpenCL Compute Context because that's where Oclgrind loads/interprets these environment variables.
          aiwc_dev->RecreateContext();

          //run AIWC and store the metrics
          devs[0] = aiwc_dev;
          *ndevs = 1;
      } else {
          //load and query predictions
          const char* metric_url = AIWC_Utils::MetricLocation(aiwc_metric_name);
          /*
          Rtime["aiwc_url"] = metric_url;
          Rtime.parseEvalQ("source('./query-model/query_model.R',chdir=TRUE)");//TODO: make this less hacky -- avoid absolute paths (both here and in the scripts).
          Rcpp::NumericVector predictions = Rtime.parseEval("query(aiwc_url)");
          std::vector<std::string> names = predictions.names();
          std::vector<double> predicted_us;
          for(Rcpp::NumericVector::iterator i = predictions.begin(); i != predictions.end(); ++i){
              predicted_us.push_back(*i);
              std::cout << names[i-predictions.begin()] << " predicted: " << *i << " us" << std::endl;
          }
          */
          std::cout << "TODO: query model here (and don't run AIWC again)!!!" << std::endl;

          // for starters let's just select the device with the shortest predicted time.
          //int index = std::min_element(predicted_us.begin(),predicted_us.end()) - predicted_us.begin();
          //for (int i = 0; i < ndevs_; i++){
          //    if (devs_[i]->name() == names[index]){
          //        devs[0] = devs_[i];
          //        *ndevs = 1;
          //    }
          //}
          //int index = 0;
          //devs[0] = devs_[index];
          //*ndevs = 1;
          //assert(*ndevs == 1);

          aiwc_dev->RecreateContext();
          devs[0] = aiwc_dev;
          *ndevs = 1;

          std::cout << "Selected " << devs[0]->name() << " for this kernel." << std::endl;

          //TODO: we could add some logic to inform scheduling policy -- if the device is free only consider it, or back-fill it in the scheduler queue.
    }

      //raise(SIGINT);

    }
    //RInside Rtime;
};

} /* namespace rt */
} /* namespace iris*/

REGISTER_CUSTOM_POLICY(AIWCPolicy, aiwc_policy)

