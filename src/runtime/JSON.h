#ifndef IRIS_SRC_RT_JSON_H
#define IRIS_SRC_RT_JSON_H

#include <string>
#include <vector>
#include <set>

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
  int RecordH2D(Command* cmd, char* buf);
  int RecordD2H(Command* cmd, char* buf);
  int RecordKernel(Command* cmd, char* buf);
  int RecordFlush();

private:
  int LoadInputs(char* src, void* tok, int i, int r);
  int LoadTasks(Graph* graph, void** params, char* src, void* tok, int i, int r);
  int LoadTask(Graph* graph, void** params, char* src, void* tok, int j, int r);

  int STR(const char* json, void* tok, char *s);
  int EQ(const char* json, void* tok, const char *s);

  void* GetParameterInput(void** params, const char* string_to_lookup);
  int InputPointer(void* p);

private:
  Platform* platform_;
  std::vector<const char*> inputs_;
  std::vector<Task*> tasks_;
  Timer* timer_;
  std::string str_;
  std::set<Mem*> mems_;
};

} /* namespace rt */
} /* namespace iris */


#endif /*IRIS_SRC_RT_JSON_H */
