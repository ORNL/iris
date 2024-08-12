#include <iris/iris.h>
#include <iris/rt/Policy.h>
#include <iris/rt/Policies.h>
#include <iris/rt/Scheduler.h>
#include <iris/rt/Command.h>
#include <iris/rt/Device.h>
#include <iris/rt/Task.h>
#include <iris/rt/Graph.h>
#include <mutex>

#ifdef RECORD_GNN_OVERHEAD
#include "timer.h"
#endif

namespace iris {
namespace rt {

std::string exec(const char* cmd) {
    std::shared_ptr<FILE> pipe(popen(cmd, "r"), pclose);
    if (!pipe) return "ERROR";
    char buffer[128];
    std::string result = "";
    while (!feof(pipe.get())) {
        if (fgets(buffer, 128, pipe.get()) != NULL)
            result += buffer;
    }
    return result;
}

class PolicyGNN: public Policy {
public:
  PolicyGNN() {}
  virtual ~PolicyGNN() {}
  virtual void Init(void* params) {
    ideal_dynamic_policy = NULL;
#ifdef RECORD_GNN_OVERHEAD
    outfile.open("gnn_overhead.csv", std::ios_base::app); // append instead of overwrite
#endif
  }
  //this policy assumes only one task-graph per submission, or at least if multiple graphs are submitted in the same IRIS instance then 
  //they have a similar shape and have the same predicted schedule.
  virtual void GetDevices(Task* task, Device** devs, int* ndevs) {
    //if the ideal policy is unknown at this point, only allow the first task to run the **expensive** GNN python predictor on the graph
    if (ideal_dynamic_policy == NULL){
      std::lock_guard<std::mutex> guard(gnn_predictor_mutex);
    }
    //all subsequent tasks run through the scheduler should now be able to use the set ideal dynamic policy
    if (ideal_dynamic_policy != NULL) return(ideal_dynamic_policy->GetDevices(task,devs,ndevs));
#ifdef RECORD_GNN_OVERHEAD
    t0 = now();
#endif

    //otherwise the ideal dynamic policy for this task graph hasn't been set yet---so go through the GNN to classify it
    const char* json_url = task->get_graph()->get_metadata()->json_url();

    if (json_url == NULL){
      _error("The task graph was not built from a JSON. TODO: construct it as a JSON to continue. For now just abort");
       return;
    }
    std::string working_dir = std::getenv("WORKING_DIRECTORY");
    std::string dagger_dir = std::getenv("DAGGER_DIRECTORY");
    std::string command = std::string("python3 "+working_dir+"/iris-gnn.py "+dagger_dir+"/"+json_url);
    printf("Running the expensive IRIS-GNN inference: %s ...\n",command.c_str());
    std::string predicted_schedule = exec(command.c_str());
    //last string value is the prediction
    std::string str = predicted_schedule;
    while( !str.empty() && std::isspace( str.back() ) ) str.pop_back() ; // remove trailing white space
    const auto pos = str.find_last_of( "\n" ) ; // locate the last white space
    // if not found, return the entire string else return the tail after the space
    str = str.substr(pos+1);
    std::string prediction = str;
    //select appropriate scheduling policy instead
    if(prediction == "locality"){
      printf("Predicted: \"locality\" using the IRIS dynamic policy: \"depend\"\n");
      ideal_dynamic_policy = scheduler_->policies()->GetPolicy(iris_depend,NULL);
    }
    else if(prediction == "concurrency"){
      printf("Predicted: \"concurrency\" using the IRIS dynamic policy \"roundrobin\"\n");
      ideal_dynamic_policy = scheduler_->policies()->GetPolicy(iris_roundrobin,NULL);
    }
    else if(prediction == "mixed"){
      printf("Predicted: \"mixed\" using the IRIS dynamic policy \"ftf\"\n");
      ideal_dynamic_policy = scheduler_->policies()->GetPolicy(iris_ftf,NULL);
    }
    else { std::cout << "Unknown prediction! " << prediction << " Aborting..." << std::endl, std::abort(); }
#ifdef RECORD_GNN_OVERHEAD
    t1 = now();
    outfile << t1-t0 << std::endl;
#endif
    return(ideal_dynamic_policy->GetDevices(task,devs,ndevs));
  }

#ifdef RECORD_GNN_OVERHEAD
    std::ofstream outfile;
    double t0, t1;
#endif

  Policy* ideal_dynamic_policy;
  std::mutex gnn_predictor_mutex;
};

} /* namespace runtime */
} /* namespace iris */

REGISTER_CUSTOM_POLICY(PolicyGNN, gnn)

