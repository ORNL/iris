#include "JSON.h"
#include "Debug.h"
#include "Command.h"
#include "Graph.h"
#include "Kernel.h"
#include "Mem.h"
#include "Platform.h"
#include "Task.h"
#include "Timer.h"
#include "Utils.h"
#include "jsmn.h"
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

namespace iris {
namespace rt {

JSON::JSON(Platform* platform) {
  platform_ = platform;
  ninputs_ = 0;
  ntasks_ = 0;
  nptrs_ = 0;
  timer_ = new Timer();
}

JSON::~JSON() {
  delete timer_;
}

int JSON::STR(const char* json, void* tok, char* s) {
  jsmntok_t* t = (jsmntok_t*) tok;
  if (t->type != JSMN_STRING) return 0;
  strncpy(s, json + t->start, t->end - t->start);
  s[t->end - t->start] = 0;
  return 1;
}

int JSON::EQ(const char* json, void* tok, const char *s) {
  jsmntok_t* t = (jsmntok_t*) tok;
  return t->type == JSMN_STRING && (int)strlen(s) == t->end - t->start &&
      strncmp(json + t->start, s, t->end - t->start) == 0;
}

void* JSON::GetInput(void** params, char* c) {
  for (int i = 0; i < ninputs_; i++) {
    if (strcmp(c, inputs_[i]) == 0) return params[i];
  }
  return NULL;
}

int JSON::Load(Graph* graph, const char* path, void** params) {
  int r;
  char* src = NULL;
  size_t srclen = 0;
  if (Utils::ReadFile((char*) path, &src, &srclen) == IRIS_ERR) {
    _error("no JSON file[%s]", path);
    return IRIS_ERR;
  }

  jsmn_parser p;
  jsmntok_t *t = (jsmntok_t*) malloc(IRIS_JSON_MAX_TOK * sizeof(jsmntok_t));
  jsmn_init(&p);

  r = jsmn_parse(&p, src, srclen, t, IRIS_JSON_MAX_TOK);

  if (r < 0) return IRIS_ERR;

  for (int i = 1; i < r; i++) {
    if (EQ(src, t + i, "inputs")) {
      i += LoadInputs(src, t, i + 1, r);
    } else if (EQ(src, t + i, "graph")) {
      i += LoadTasks(graph, params, src, t, i + 1, r);
    }
  }
  free(t);
  return IRIS_OK;
}

int JSON::LoadInputs(char* src, void* tok, int i, int r) {
  jsmntok_t* t = (jsmntok_t*) tok;
  int j = 0;
  ninputs_ = t[i].size;
  for (j = 1; j < ninputs_ + 1; j++) {
    STR(src, t + i + j, inputs_[j - 1]);
  }
  return j;
}

int JSON::LoadTasks(Graph* graph, void** params, char* src, void* tok, int i, int r) {
  jsmntok_t* t = (jsmntok_t*) tok;
  int j = 0;
  for (j = 1; j < r; j++) {
    if (EQ(src, t + i + j, "tasks")) {
      int ntasks = t[i + j + 1].size;
      for (int l = 0; l < ntasks; l++)
        j += LoadTask(graph, params, src, tok, i + j, r);
    }
  }
  return j;
}

int JSON::LoadTask(Graph* graph, void** params, char* src, void* tok, int j, int r) {
  jsmntok_t* t = (jsmntok_t*) tok;
  char buf[256];
  Task* task = Task::Create(platform_, IRIS_TASK_PERM, NULL);
  tasks_[ntasks_++] = task;
  int l, k = 0;
  for (k = 1; k < r; k++) {
    if (EQ(src, t + j + k, "name")) {
      k++;
      STR(src, t + j + k, buf);
      task->set_name(buf);
    } else if (EQ(src, t + j + k, "h2d")) {
      k++;
      STR(src, t + j + (++k), buf);
      iris_mem mem = (iris_mem) GetInput(params, buf);
      STR(src, t + j + (++k), buf);
      void* host = GetInput(params, buf);
      STR(src, t + j + (++k), buf);
      void* p_off = GetInput(params, buf);
      size_t off = p_off ? (*(size_t*) p_off) : atol(buf);
      STR(src, t + j + (++k), buf);
      void* p_size = GetInput(params, buf);
      size_t size = p_size ? (*(size_t*) GetInput(params, buf)) : atol(buf);
      Command* cmd = Command::CreateH2D(task, mem->class_obj, off, size, host);
      task->AddCommand(cmd);
    } else if (EQ(src, t + j + k, "d2h")) {
      k++;
      STR(src, t + j + (++k), buf);
      iris_mem mem = (iris_mem) GetInput(params, buf);
      STR(src, t + j + (++k), buf);
      void* host = GetInput(params, buf);
      STR(src, t + j + (++k), buf);
      void* p_off = GetInput(params, buf);
      size_t off = p_off ? (*(size_t*) p_off) : atol(buf);
      STR(src, t + j + (++k), buf);
      void* p_size = GetInput(params, buf);
      size_t size = p_size ? (*(size_t*) GetInput(params, buf)) : atol(buf);
      Command* cmd = Command::CreateD2H(task, mem->class_obj, off, size, host);
      task->AddCommand(cmd);
    } else if (EQ(src, t + j + k, "kernel")) {
      k++;
      char name[256];
      STR(src, t + j + (++k), name);
      int dim = t[j + k + 1].size;
      size_t* gws = (size_t*) malloc(sizeof(size_t) * dim);
      for (l = 0; l < dim; l++) {
        STR(src, t + j + k + 1 + l + 1, buf);
        void* p_gws = GetInput(params, buf);
        size_t s = p_gws ? (*(size_t*) GetInput(params, buf)) : atol(buf);
        gws[l] = s;
      }
      k += l + 1;
      int nkparams = t[j + k + 1].size;
      void** kparams = (void**) malloc(sizeof(void*) * t[j + k + 1].size);
      for (l = 0; l < t[j + k + 1].size; l++) {
        STR(src, t + j + k + 1 + l + 1, buf);
        kparams[l] = GetInput(params, buf);
      }
      k += l + 1;
      int* kparams_info = (int*) malloc(sizeof(int) * t[j + k + 1].size);
      for (l = 0; l < t[j + k + 1].size; l++) {
        STR(src, t + j + k + 1 + l + 1, buf);
        if (strcmp("rw", buf) == 0) kparams_info[l] = iris_rw;
        else if (strcmp("r", buf) == 0) kparams_info[l] = iris_r;
        else if (strcmp("w", buf) == 0) kparams_info[l] = iris_w;
        else _check();
      }
      k += l + 1;
      Kernel* kernel = platform_->GetKernel(name);
      Command* cmd = Command::CreateKernel(task, kernel, dim, NULL, gws, NULL, nkparams, kparams, NULL, kparams_info, NULL);
      task->AddCommand(cmd);
    } else if (EQ(src, t + j + k, "depends")) {
      k++;
      for (l = 0; l < t[j + k].size; l++) {
        STR(src, t + j + k + l + 1, buf);
        for (int m = 0; m < ntasks_; m++) {
          if (strcmp(buf, tasks_[m]->name()) == 0) {
            task->AddDepend(tasks_[m]);
            break;
          }
        }
      }
      k += l;
    } else if (EQ(src, t + j + k, "target")) {
      STR(src, t + j + (++k), buf);
      void* p_target = GetInput(params, buf);
      int target = iris_default;
      if (p_target) target = (*(int*) p_target);
      else if (strcmp(buf, "cpu") == 0) target = iris_cpu;
      else if (strcmp(buf, "gpu") == 0) target = iris_gpu;
      else target = iris_default;
      task->set_target_perm(target, NULL);  
      graph->AddTask(task);
      break;
    }
  }
  return k;
}

int JSON::RecordTask(Task* task) {
  char buf[256];
  memset(buf, 0, 256);
  sprintf(buf, "{\n  \"name\": \"task%lu\",\n", task->uid());
  str_.append(buf);
  for (int i = 0; i < task->ncmds(); i++) {
    memset(buf, 0, 256);
    Command* cmd = task->cmd(i);
    if (cmd->type() == IRIS_CMD_H2D) RecordH2D(cmd, buf);
    else if (cmd->type() == IRIS_CMD_D2H) RecordD2H(cmd, buf);
    else if (cmd->type() == IRIS_CMD_KERNEL) RecordKernel(cmd, buf);
    else _todo("cmd[%d]", cmd->type());
    str_.append(buf);
  }
  str_.append("  \"depends\": [");
  int ndeps = task->ndepends();
  for (int i = 0; i < ndeps; i++) {
    str_.append("\"task");
    str_.append(std::to_string(task->depend(i)->uid()));
    if (ndeps > 1 && i < ndeps - 1) str_.append("\", ");
    else str_.append("\"");
  }
  str_.append("],\n");
  memset(buf, 0, 256);
  sprintf(buf, "  \"target\": \"0x%x\"\n},\n", task->brs_policy());
  str_.append(buf);
  return IRIS_OK;
}

int JSON::RecordH2D(Command* cmd, char* buf) {
  mems_.insert(cmd->mem());
  sprintf(buf, "  \"h2d\": [\"mem-%lu\", \"user-%d\", \"%zu\", \"%zu\"],\n", cmd->mem()->uid(), InputPointer(cmd->host()), cmd->off(0), cmd->size());
  return IRIS_OK;
}

int JSON::RecordD2H(Command* cmd, char* buf) {
  mems_.insert(cmd->mem());
  sprintf(buf, "  \"d2h\": [\"mem-%lu\", \"user-%d\", \"%zu\", \"%zu\"],\n", cmd->mem()->uid(), InputPointer(cmd->host()), cmd->off(0), cmd->size());
  return IRIS_OK;
}

int JSON::RecordKernel(Command* cmd, char* buf) {
  std::string str;
  char c[256];
  sprintf(c, "  \"kernel\": [\"%s\", \"%zu\", [", cmd->kernel()->name(), cmd->gws(0));
  str.append(c);
  memset(c, 0, 256);

  int nargs = cmd->kernel_nargs();

  for (int i = 0; i < nargs; i++) {
    KernelArg* arg = cmd->kernel_arg(i);
    if (arg->mem) {
      mems_.insert(arg->mem);
      str.append("\"mem-");
      str.append(std::to_string(arg->mem->uid()));
      str.append("\"");
    } else {
    }
    if (nargs > 1 && i < nargs - 1) str.append(", ");
  }
  str.append("], [");

  for (int i = 0; i < nargs; i++) {
    KernelArg* arg = cmd->kernel_arg(i);
    if (arg->mem) {
      if (arg->mode == iris_r) str.append("\"r\"");
      else if (arg->mode == iris_w) str.append("\"w\"");
      else if (arg->mode == iris_rw) str.append("\"rw\"");
      else _error("not valid mode[%d]", arg->mode);
    } else {
    }
    if (nargs > 1 && i < nargs - 1) str.append(", ");
  }

  str.append("] ],\n");
  sprintf(buf, "%s", str.c_str());
  return IRIS_OK;
}

int JSON::RecordFlush() {
  char path[256];
  char buf[128];
  memset(buf, 0, 128);
  Utils::Datetime(buf);
  sprintf(path, "graph-%s.json", buf);
  int fd = open((const char*) path, O_CREAT | O_WRONLY, 0644);
  if (fd == -1) return IRIS_ERR;
  memset(buf, 0, 128);
  sprintf(buf, "{\"iris-graph\": {\n\"inputs\": [");
  write(fd, buf, strlen(buf));
  memset(buf, 0, 128);
  for (std::set<Mem*>::iterator I = mems_.begin(); I != mems_.end(); ++I) {
    sprintf(buf, "\"mem-%lu\", ", (*I)->uid());
    write(fd, buf, strlen(buf));
    memset(buf, 0, 128);
  }

  for (int i = 0; i < nptrs_; i++) {
    if (nptrs_ > 1 && i < nptrs_ - 1) sprintf(buf, "\"user-%d\", ", i);
    else if (i == nptrs_ - 1) sprintf(buf, "\"user-%d\"", i);
    write(fd, buf, strlen(buf));
    memset(buf, 0, 128);
  }
  sprintf(buf, "],\n\"graph\": {\n\"tasks\": [\n");
  write(fd, buf, strlen(buf));
  write(fd, str_.c_str(), strlen(str_.c_str()));
  memset(buf, 0, 128);
  sprintf(buf, "]\n}}}\n");
  write(fd, buf, strlen(buf));
  close(fd);
  nptrs_ = 0;
  return IRIS_OK;
}

int JSON::InputPointer(void* p) {
  for (int i = 0; i < nptrs_; i++) {
    if (ptrs_[i] == p) return i;
  }
  ptrs_[nptrs_++] = p;
  return nptrs_ - 1;
}

} /* namespace rt */
} /* namespace iris */

