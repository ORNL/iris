#include "timer.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <iris/iris.h>
#include <assert.h>
#include <getopt.h>
#include <map>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <limits>
#include <ctype.h>

double t0, t1;

void ShowUsage(){
printf("This runner will run the generated DAG from DAGGER on IRIS. It can be immediately after DAGGER has been run and accepts the same arguments as those supplied to DAGGER.\nThe required additional arguments include:\n\tthe size of the memory buffers to use in this IRIS test (\"--size\"),\n\tthe number of repeats (\"--repeats\"),\n\tand the location to log the timing results (\"--logfile\")\n");
printf("For instance:\n");
printf("./dagger_runner\t --kernels=\"process,ijk\"\n");
printf("\t\t --graph=\"linear10.json\"\n");
printf("\t\t --buffers-per-kernel=\"process: rw,ijk: r r w rw\"\n");
printf("\t\t --concurrent-kernels=\"process:2,ijk:4\"\n");
printf("\t\t --kernel-dimensions=\"ijk:2,process:1\"\n");
printf("\t\t --size=\"1024\"\n");
printf("\t\t --repeats=\"100\"\n");
printf("\t\t --logfile=\"log.csv\"\n");
printf("\t\t --scheduling-policy=\"roundrobin\"\t all options include (roundrobin, depend, profile, random, sdq, ftf, custom) or any integer [0-9] denoting the device id to run all tasks on\n");
//printf("\t\t --num-tasks=\"6\"\t (optional) only used for throughput computation\n");
//printf("\t\t --sandwich\t\t (optional) determines if there are the beginning and terminating nodes\n");
//printf("\t\t --kernel-split=\"70,30\"\t (optional) list of probabilities of each kernel being used\n");
//printf("\t\t --num-memory-objects=\"5\"\t (optional) the total number of memory objects to be passed around in the DAG between tasks (allows greater task interactions).\n");
printf("\t\t --use-data-memory=1 (optional) Enables the graph to use memory instead of the default explicit memory buffers. This results in final explicit flush events of buffers that are written.");
}

int main(int argc, char** argv) {

size_t SIZE;
int REPEATS;
char* POLICY;
char* LOGFILE = NULL;
char* GRAPHFILE = NULL;
char LOG_BUFFER[32];
  LOG_BUFFER[0] = '\0';
  int retcode;
  int task_target = -1;//this is set in the scheduling-policy e.g. --scheduling-policy="roundrobin"
  int memory_task_target = iris_pending;
  int duplicates = 1;
  bool use_data_memory = false;
  char* custom_policy_name = NULL;
  char* custom_policy_liburl = NULL;

  std::map<std::string,bool> required_arguments_set = {
    {"kernels", false},
    {"graph",false},
    {"buffers-per-kernel", false},
    {"size", false},
    {"repeats", false},
    {"kernel-dimensions", false},
    {"scheduling-policy", false}
  };

  struct kernel_parameters {
    char* name;
    std::vector<char*> buffers;
    int concurrency;
    int dimensions;
    std::vector<size_t> lws;
  };

  std::map<std::string,int> scheduling_policy_lookup = {
    {"roundrobin", iris_roundrobin},
    {"depend", iris_depend},
    {"data", iris_data},
    {"profile", iris_profile},
    {"random", iris_random},
    {"sdq",iris_sdq},
    {"ftf", iris_ftf},
    {"custom", iris_custom}
  };

  std::vector<kernel_parameters> kernels;
  int num_kernels = 0;
  int num_tasks = 0;
  int opt_char;
  int option_index;
  static struct option long_options[] = {
    {"graph", required_argument, 0, 'g'},
    {"kernels", required_argument, 0, 'k'},
    {"buffers-per-kernel", required_argument, 0, 'b'},
    {"duplicates", optional_argument, 0, 'z'},
    {"concurrent-kernels", optional_argument, 0, 'c'},
    {"size", required_argument, 0, 's'},
    {"repeats", required_argument, 0, 'r'},
    {"logfile", required_argument, 0, 'l'},
    {"kernel-dimensions", required_argument, 0, 'd'},
    {"scheduling-policy", required_argument, 0, 't'},
    //not critical---but we do use the number of tasks in the dag to determine throughput
    {"num-tasks", required_argument, 0, 'n'},
    //superfluous arguments (but are accepted by DAGGER)
    {"kernel-split", required_argument, 0, 'p'},
    {"depth", required_argument, 0, 'e'},
    {"min-width", required_argument, 0, 'i'},
    {"max-width", required_argument, 0, 'x'},
    {"sandwich", no_argument, 0, 'y'},
    {"num-memory-objects",required_argument, 0,'a'},
    {"skips",required_argument,0,'q'},
    {"use-data-memory",no_argument, 0,'f'},
    {"local-sizes",required_argument, 0, 'w'}
  };

  while((opt_char = getopt_long(argc, argv, "s=", long_options, &option_index)) != -1) {
    switch(opt_char){

      case (int)'g':{//graph_file
          GRAPHFILE = optarg;
          if (GRAPHFILE != NULL){
            required_arguments_set["graph"] = true;
          }
        } break;

      case (int)'k':{//kernels
          //split all kernels by ,
          char* x = strtok(optarg,",");
          while(x){
            struct kernel_parameters kp;
            kp.name = x;
            kp.concurrency = 0;
            kernels.push_back(kp);
            x = strtok(NULL, ",");
            num_kernels++;
          }
          if(num_kernels > 0) {
            required_arguments_set["kernels"] = true;
          }
        } break;

      case (int)'b':{//buffers-per-kernel
          //sample: --buffers-per-kernel="process: rw,ijk: r r w rw"
          int num_kernels_with_buffers_set = 0;
          //split all kernels by ,
          char* kernel_str = strtok(optarg,",");
          std::vector<char*> kernel_strings;
          while(kernel_str){
            kernel_strings.push_back(kernel_str);
            kernel_str = strtok(NULL, ",");
          }
          //for each kernel (key-value) string
          for (auto & kernel_str : kernel_strings){
            char* key_str = strtok(kernel_str,":");
            char* val_str = strtok(NULL,":");
            //find the matching kernel on which to add the buffer arguments
            for (auto & kernel : kernels){
              if (strcmp(kernel.name,key_str) == 0){//found it!
                //now split up the string of buffers...
                char* buffer_type = strtok(val_str," ");
                while(buffer_type){
                  //if the buffer type isn't an empty string, then add it!
                  if (strcmp(buffer_type," ") != 0 and strlen(buffer_type) != 0){
                    kernel.buffers.push_back(buffer_type);
                  }
                  buffer_type = strtok(NULL," ");
                }
                //check to make sure at least one buffer was specified for this kernel
                if (kernel.buffers.size() != 0){
                  num_kernels_with_buffers_set ++;
                }
              }
            }
          }
          //final check of goodness
          if(num_kernels_with_buffers_set == num_kernels){
            required_arguments_set["buffers-per-kernel"] = true;
          }else{
            printf("\033[41mError: Incorrect --buffers-per-kernel supplied.\n%s\033[0m",optarg);
          }
        } break;
      case (int)'z':{//duplicates
          duplicates = strtol(optarg,NULL,10);//convert the provided value to an (base-10) integer
          //check to ensure a suitable value was provided for this test
          required_arguments_set["duplicates"] = true;
      } break;
      case (int)'c':{//concurrent-kernels
          int num_kernels_with_concurrency_set = 0;
          //split all kernels by ,
          char* kernel_str = strtok(optarg,",");
          std::vector<char*> kernel_strings;
          while(kernel_str){
            kernel_strings.push_back(kernel_str);
            kernel_str = strtok(NULL, ",");
          }
          //for each kernel (key-value) string
          for (auto & kernel_str : kernel_strings){
            char* key_str = strtok(kernel_str,":");
            char* val_str = strtok(NULL,":");
            //find the matching kernel on which to add the buffer arguments
            for (auto & kernel : kernels){
              if (strcmp(kernel.name,key_str) == 0){//found it!
                kernel.concurrency = strtol(val_str,NULL,10);//convert the provided value to an (base-10) integer
                //check to ensure a good value was set for this kernel
                if (kernel.concurrency != 0){
                  num_kernels_with_concurrency_set ++;
                }
              }
            }
          }
          //final check of goodness
          if(num_kernels_with_concurrency_set == num_kernels){
            required_arguments_set["concurrent-kernels"] = true;
          }else{
            printf("\033[41mError: Incorrect --concurrent-kernels supplied.\n%s\033[0m",optarg);
          }
        } break;

      case (int)'d':{//kernel-dimensions
          int num_kernels_with_dimensions_set = 0;
          //split all kernels by ,
          char* kernel_str = strtok(optarg,",");
          std::vector<char*> kernel_strings;
          while(kernel_str){
            kernel_strings.push_back(kernel_str);
            kernel_str = strtok(NULL, ",");
          }
          //for each kernel (key-value) string
          for (auto & kernel_str : kernel_strings){
            char* key_str = strtok(kernel_str,":");
            char* val_str = strtok(NULL,":");
            //find the matching kernel on which to add the buffer arguments
            for (auto & kernel : kernels){
              if (strcmp(kernel.name,key_str) == 0){//found it!
                kernel.dimensions = strtol(val_str,NULL,10);//convert the provided value to an (base-10) integer
                //check to ensure a good value was set for this kernel
                if (kernel.dimensions != 0){
                  num_kernels_with_dimensions_set ++;
                }
              }
            }
          }
          //final check of goodness
          if(num_kernels_with_dimensions_set == num_kernels){
            required_arguments_set["kernel-dimensions"] = true;
          }else{
            printf("\033[41mError: Incorrect --kernel-dimensions supplied.\n%s\033[0m",optarg);
          }
        } break;

      case (int)'s':{//size
          SIZE = strtol(optarg,NULL,10);//convert the provided value to an (base-10) integer
          //check to ensure a good size was provided for this test
          if (SIZE != 0){
            required_arguments_set["size"] = true;
          }
        } break;

      case (int)'r':{//repeats
          REPEATS = strtol(optarg,NULL,10);//convert the provided value to an (base-10) integer
          //check to ensure a suitable value was provided for this test
          if (REPEATS != 0){
            required_arguments_set["repeats"] = true;
          }
         } break;

      case (int)'l':{//logfile
          LOGFILE = optarg;
          setenv("IRIS_HISTORY","1",true);
          setenv("IRIS_HISTORY_FILE",LOGFILE,true);
        } break;

      case (int)'t':{//scheduling-policy
          //if the policy requested by the user doesn't exist, fail.
          char* policy_name = strtok(optarg,":");
          if (scheduling_policy_lookup.find(policy_name) != scheduling_policy_lookup.end()){
            task_target = scheduling_policy_lookup[policy_name]; //using only device #1 and 2
            POLICY = policy_name;
            if (task_target == iris_custom){
              custom_policy_name = new char[128];
              custom_policy_liburl = new char[128];
              //This is our only custom policy loader
              char* cpn = strtok(NULL, ":");
              char* cpl = strtok(NULL, ":");
              if (cpn == NULL) task_target = -1; //fail if no policy name provided
              strcpy(custom_policy_name,cpn);
              //we need to record the name of the policy and lib.so filename
              //but can't evaluate it until *after* IRIS has been initialized
              if (cpl == NULL) task_target = -1;//fail if no libCustomPolicyName.so provided
              strcpy(custom_policy_liburl,cpl);
              //give it a shiny printed name
              strcpy(POLICY,"custom:");
              strcat(POLICY,custom_policy_name);
            }
          } else if (isdigit(optarg[0])){//we also support assigning individual device ids.
            task_target = atoi(optarg);
          }
          if (task_target != -1){
            required_arguments_set["scheduling-policy"] = true;
          }
        } break;

      case (int)'n':{//num-tasks
          num_tasks = strtol(optarg,NULL,10);
        } break;

      case (int)'p'://kernel-split
        break;

      case (int)'a'://num-memory-objects
        break;

      case (int)'w':{//local-sizes
          printf("The --local-sizes argument provided to the runner currently doesn't override the DAG, instead whatever is passed to the generator is used.\n");
          //sample: --local-sizes="ijk:256 256,process:128"
          int num_kernels_with_lws_set = 0;

          //split all kernels by ,
          char* kernel_str = strtok(optarg,",");
          std::vector<char*> kernel_strings;
          while(kernel_str){
            kernel_strings.push_back(kernel_str);
            kernel_str = strtok(NULL, ",");
          }
          //for each kernel (key-value) string
          for (auto & kernel_str : kernel_strings){
            char* key_str = strtok(kernel_str,":");
            char* val_str = strtok(NULL,":");
            //find the matching kernel on which to add the lws arguments
            for (auto & kernel : kernels){
              if (strcmp(kernel.name,key_str) == 0){//found it!
                //now split up the string of ints...
                char* wgsize_buffer = strtok(val_str," ");
                while(wgsize_buffer != NULL){
                  size_t lws_val = strtol(wgsize_buffer,NULL,10);
                  kernel.lws.push_back(lws_val);
                  wgsize_buffer = strtok(NULL ," ");
                }
                //check to make sure at least one buffer was specified for this kernel
                if (kernel.lws.size() != kernel.dimensions){
                  printf("\033[41mError: Incorrect --local-sizes don't match the kernel dimensions.\n%s\033[0m",optarg);
                }
                else{
                  num_kernels_with_lws_set ++;
                }
              }
            }
          }
          //final check of goodness
          if(num_kernels_with_lws_set == num_kernels){
            printf("\033[41mError: Incorrect --local-sizes supplied.\n%s\033[0m",optarg);
          }
        } break;
      //TODO: copy the previous chunk for gws and these could be passed as inputs for the graph

      case (int)'f':{//use-data-memory
        use_data_memory = true; 
        } break;
    };
  }

  //ensure that all required arguments have been set
  bool all_good = true;
  for( std::map<std::string,bool>::iterator it = required_arguments_set.begin(); it != required_arguments_set.end(); it++){
    all_good = all_good && it->second;
    if (!all_good){
      printf("\033[41mArgument --%s was not set.\n\033[0m",it->first.c_str());
      break;
    }
  }

  if(!all_good){
    printf("\033[43mNot all arguments were properly provided to the runner!\n\033[0m");
    ShowUsage();
    return(EXIT_FAILURE);
  }

  //finally, ensure at least 1 copy of each set of memory buffers are used
  if (duplicates < 1){
    duplicates = 1;
  }

  iris_init(&argc, &argv, true);

  if (custom_policy_name != NULL){
    int errnum = iris_register_policy(custom_policy_liburl, custom_policy_name, (void*) 16);
  }

  iris_overview();

  printf("REPEATS:%d LOGFILE:%s POLICY:%s\n",REPEATS,LOGFILE,POLICY);
  for (int i = 0; i < num_kernels; i ++){
    if (kernels[i].concurrency < 1) kernels[i].concurrency = 1;
    printf("KERNEL:%s available has %d instances\n",kernels[i].name,kernels[i].concurrency);
  }

  for (int t = 0; t < REPEATS; t++) {

    int num_buffers_used = 0;
    std::vector<double*> host_mem;
    std::vector<iris_mem> dev_mem;
    std::vector<size_t> sizecb;

    for(auto & kernel : kernels){
      for(auto dup = 1; dup < duplicates+1; dup++){
        for(auto concurrent_kernel_instance = 1; concurrent_kernel_instance < kernel.concurrency+1; concurrent_kernel_instance++){
          int argument_index = 0;
          for(auto & buffer : kernel.buffers){
            //create and add the host-side buffer based on it's type
            std::string buf = std::string(buffer);
            if(buf == "r"){
              //create and populate memory
              double* tmp = new double[(int)pow(SIZE,kernel.dimensions)];
              for(int i = 0; i < pow(SIZE,kernel.dimensions); i++){
                tmp[i] = i;
              }
              host_mem.push_back(tmp);
              num_buffers_used++;
            } else if(buf == "w"){
              double* tmp = new double[(int)pow(SIZE,kernel.dimensions)];
              host_mem.push_back(tmp);
              num_buffers_used++;
            } else if(buf == "rw"){
              //create and populate memory
              double* tmp = new double[(int)pow(SIZE,kernel.dimensions)];
              for(int i = 0; i < pow(SIZE,kernel.dimensions); i++){
                tmp[i] = i;
              }
              host_mem.push_back(tmp);
              num_buffers_used++;
            } else {
              printf("\033[41mInvalid memory argument! Kernel %s has a buffer of memory type %s but only r,w or rw are allowed.\n\033[0m",kernel.name,buf.c_str());
              return(EXIT_FAILURE);
            }
            //and create device memory of the same size
            iris_mem x;
            char buffer_name[80];
            sprintf(buffer_name,"%s-%s-%d",kernel.name,buf.c_str(),argument_index);
            if (use_data_memory){
              retcode = iris_data_mem_create(&x,host_mem[0], (int)pow(SIZE,kernel.dimensions)*sizeof(double));
              memory_task_target = iris_pending;
            }
           else
              retcode = iris_mem_create( (int)pow(SIZE,kernel.dimensions)*sizeof(double), &x);//, (char*)buffer_name);
           assert (retcode == IRIS_SUCCESS && "Failed to create IRIS memory buffer");
           dev_mem.push_back(x);
           //and update the count of buffers used
           num_buffers_used++;
           argument_index++;
          }
        }
      }
      sizecb.push_back(pow(SIZE,kernel.dimensions) * sizeof(double));
      printf("SIZE[%zu] MATRIX_SIZE[%lu]MB\n", SIZE, sizecb.back() / 1024 / 1024);
    }

    //variable number of memory buffers can be provided into IRIS
    void* json_inputs[5+num_buffers_used];
    int indexer = 0;
    json_inputs[indexer] = &SIZE; indexer++;
    for(auto & bytes : sizecb){
      json_inputs[indexer] = &bytes; indexer++;
    }
    for(auto & mem : host_mem){
      json_inputs[indexer] = mem; indexer++;
    }
    for(auto & mem : dev_mem){
      json_inputs[indexer] = &mem; indexer++;
    }
    json_inputs[indexer] = &memory_task_target; indexer++;
    json_inputs[indexer] = &task_target; indexer++;
    if (task_target == iris_custom){json_inputs[indexer] = custom_policy_name; indexer++;}

    iris_graph graph;
    retcode = iris_graph_create_json(GRAPHFILE, json_inputs, &graph);
    assert(retcode == IRIS_SUCCESS && "Failed to create IRIS graph");
    for(auto i = 0; i < indexer; i++){
      json_inputs[i] = NULL;
    }
    t0 = now();

    iris_graph_submit(graph, iris_gpu, true);//iris_default in task-graph target is the only case that doesn't override the submission policy--so leave it as iris_gpu.

    retcode = iris_synchronize();
    assert(retcode == IRIS_SUCCESS && "Failed to synchronize IRIS");

    t1 = now();

    //TODO: currently we only present FLOPs for ijk kernel but not DEVICE_CONCURRENCY! IT SHOULD BE MULTIPLIED BY THE TOTAL NUMBER OF TASKS!
/*
    if (num_tasks != 0){
      //compute FLOPs
      printf("iteration took %i secs: %lf\n", t, t1 - t0);
      long i=SIZE; long j=SIZE; long k=SIZE;
      long long op_count, iop_count, fop_count;
      op_count = i*j*k*6*num_tasks;
      iop_count= i*j*k*4*num_tasks;
      fop_count= i*j*k*2*num_tasks;
      double gops, giops, gflops;
      gops   = 1.e-9 * ((double)op_count /(t1 - t0));
      giops  = 1.e-9 * ((double)iop_count/(t1 - t0));
      gflops = 1.e-9 * ((double)fop_count/(t1 - t0));
      //append the result to file
      if (LOGFILE != NULL) {
        LF_HANDLE = fopen(LOGFILE, "a");
        assert(LF_HANDLE != NULL);
        if(i == 0){
          sprintf(LOG_BUFFER, "gops,giops,gflops\n");
        }
        sprintf(LOG_BUFFER, "%f,%f,%f\n", gops,giops,gflops);
        fputs(LOG_BUFFER,LF_HANDLE);
        fclose(LF_HANDLE);
      }
      else{
        printf("GOPs = %f, GIOPs = %f, GFLOPs = %f\n",gops,giops,gflops);
      }
    }
*/
    //clean up device memory
    for (auto& this_mem : dev_mem){
      iris_mem_release(this_mem);
    }
    //clean up host memory
    for (auto & this_mem : host_mem){
      delete [] this_mem;
    }
  }

  int num_errors = iris_error_count();
  printf("dagger_runner finished with %i errors.\n", num_errors);
  iris_finalize();
  return(num_errors);
}

