#ifndef BRISBANE_SRC_RT_JSON_H
#define BRISBANE_SRC_RT_JSON_H

#include <string>
#include <set>

#define BRISBANE_JSON_MAX_TOK (1024 * 1024)

namespace brisbane {
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

  void* GetInput(void** params, char* c);
  int InputPointer(void* p);

private:
  Platform* platform_;
  char inputs_[32][64];
  Task* tasks_[8192];
  int ninputs_;
  int ntasks_;
  Timer* timer_;
  std::string str_;
  void* ptrs_[128];
  int nptrs_;
  std::set<Mem*> mems_;
};

} /* namespace rt */
} /* namespace brisbane */


#endif /*BRISBANE_SRC_RT_JSON_H */
