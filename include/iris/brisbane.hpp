#ifndef BRINSBANE_INCLUDE_BRISBANE_BRISBANE_HPP
#define BRINSBANE_INCLUDE_BRISBANE_BRISBANE_HPP

#include <iris/brisbane_runtime.h>

namespace brisbane {

class Platform {
public:
Platform() {
  finalized_ = false;
}

~Platform() {
  if (!finalized_) finalize();
}

int init(int* argc, char*** argv, bool sync) {
  return brisbane_init(argc, argv, (int) sync);
}

int finalize() {
  if (finalized_) return BRISBANE_ERR;
  int ret = brisbane_finalize();
  finalized_ = true;
  return ret;
}

private:
bool finalized_;

};

class Mem {
public:
Mem(size_t size) {
  brisbane_mem_create(size, &mem_);
}

~Mem() {
  brisbane_mem_release(mem_);
}

brisbane_mem mem() { return mem_; }

private:
  brisbane_mem mem_;
};

class Task {
public:
Task() {
  brisbane_task_create(&task_);
}

~Task() {
  brisbane_task_release(task_);
}

int h2d_full(Mem* mem, void* host) {
  return brisbane_task_h2d_full(task_, mem->mem(), host);
}

int d2h_full(Mem* mem, void* host) {
  return brisbane_task_d2h_full(task_, mem->mem(), host);
}

int kernel(const char* kernel, int dim, size_t* off, size_t* gws, size_t* lws, int nparams, void** params, int* params_info) {
  void** new_params = new void*[nparams];
  for (int i = 0; i < nparams; i++) {
    if (params_info[i] == brisbane_w ||
        params_info[i] == brisbane_r ||
        params_info[i] == brisbane_rw) {
      new_params[i] = ((Mem*) params[i])->mem();
    } else new_params[i] = params[i];
  }
  int ret = brisbane_task_kernel(task_, kernel, dim, off, gws, lws, nparams, new_params, params_info);
  delete[] new_params;
  return ret;
}

int submit(int device, const char* opt, int sync) {
  return brisbane_task_submit(task_, device, opt, sync);
}

private:
  brisbane_task task_;

};

} // namespace brisbane

#endif /* BRINSBANE_INCLUDE_BRISBANE_BRISBANE_HPP */

