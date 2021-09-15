#ifndef IRIS_INCLUDE_IRIS_IRIS_HPP
#define IRIS_INCLUDE_IRIS_IRIS_HPP

#include <iris/iris_runtime.h>

namespace iris {

class Platform {
public:
Platform() {
  finalized_ = false;
}

~Platform() {
  if (!finalized_) finalize();
}

int init(int* argc, char*** argv, bool sync) {
  return iris_init(argc, argv, sync ? 1 : 0);
}

int finalize() {
  if (finalized_) return IRIS_ERROR;
  int ret = iris_finalize();
  finalized_ = true;
  return ret;
}

private:
bool finalized_;

};

class Mem {
public:
Mem(size_t size) {
  iris_mem_create(size, &mem_);
}

~Mem() {
  iris_mem_release(mem_);
}

iris_mem mem() { return mem_; }

private:
  iris_mem mem_;
};

class Task {
public:
Task() {
  iris_task_create(&task_);
}

~Task() {
  iris_task_release(task_);
}

int h2d(Mem* mem, size_t off, size_t size, void* host) {
  return iris_task_h2d(task_, mem->mem(), off, size, host);
}

int h2d_full(Mem* mem, void* host) {
  return iris_task_h2d_full(task_, mem->mem(), host);
}

int d2h(Mem* mem, size_t off, size_t size, void* host) {
  return iris_task_d2h(task_, mem->mem(), off, size, host);
}

int d2h_full(Mem* mem, void* host) {
  return iris_task_d2h_full(task_, mem->mem(), host);
}

int kernel(const char* kernel, int dim, size_t* off, size_t* gws, size_t* lws, int nparams, void** params, int* params_info) {
  void** new_params = new void*[nparams];
  for (int i = 0; i < nparams; i++) {
    if (params_info[i] == iris_w ||
        params_info[i] == iris_r ||
        params_info[i] == iris_rw) {
      new_params[i] = ((Mem*) params[i])->mem();
    } else new_params[i] = params[i];
  }
  int ret = iris_task_kernel(task_, kernel, dim, off, gws, lws, nparams, new_params, params_info);
  delete[] new_params;
  return ret;
}

int submit(int device, const char* opt, bool sync) {
  return iris_task_submit(task_, device, opt, sync ? 1 : 0);
}

private:
  iris_task task_;

};

} // namespace iris

#endif /* IRIS_INCLUDE_IRIS_IRIS_HPP */

