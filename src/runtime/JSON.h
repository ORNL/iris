#ifndef IRIS_SRC_RT_JSON_H
#define IRIS_SRC_RT_JSON_H

#include <string>
#include <vector>

namespace iris {
namespace rt {

class Command;
class Graph;
class Mem;
class Platform;
class Task;
class Timer;

class JSON {
public:
  JSON(Platform* platform);
  ~JSON();

  int Load(Graph* graph, const char* path, void** params);

  int RecordTask(Task* task);
  int RecordFlush();

private:
  int LoadInputs(char* src, void* tok, int i, int r);
  int LoadTasks(Graph* graph, void** params, char* src, void* tok, int i, int r);
  int LoadTask(Graph* graph, void** params, char* src, void* tok, int j, int r);

private:
  void* GetParameterInput(void** params, const char* string_to_lookup);
  int UniqueUIDFromHostPointer(void*host_ptr);
  int UniqueUIDFromDevicePointer(Mem* dev_ptr);
  const std::string NameFromHostPointer(void*host_ptr);
  const std::string NameFromDeviceMem(Mem* dev_mem);

  Platform* platform_;
  std::vector<const char*> inputs_;
  std::vector<Task*> tasks_;
  Timer* timer_;
  std::string str_;
  std::vector<Mem*> mems_;
  std::vector<void*> host_ptrs_;
};

} /* namespace rt */
} /* namespace iris */


#endif /*IRIS_SRC_RT_JSON_H */
