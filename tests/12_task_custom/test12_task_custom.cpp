#include <iris/iris.h>
#include <iris/rt/DeviceCUDA.h>
#include <iris/rt/LoaderCUDA.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int cmd_handler_nv(void* params, void* device) {
  int off = 0;
  size_t name_len = *((size_t*) params);
  off += sizeof(name_len);
  char* name = (char*) malloc(name_len);
  memcpy(name, params + off, name_len);
  off += name_len;
  int type = *((int*) (params + off));
  off += sizeof(type);
  void* dptr = *((void**) (params + off));
  off += sizeof(dptr);
  size_t size = *((size_t*) (params + off));
  printf("[%s:%d] namelen[%lu] name[%s] type[%d] dptr[%p] size[%lu]\n", __FILE__, __LINE__, name_len, name, type, dptr, size);

  CUtexref tex;
  CUresult err;
  iris::rt::DeviceCUDA* dev = (iris::rt::DeviceCUDA*) device;
  iris::rt::LoaderCUDA* ld = dev->ld();
  err = ld->cuModuleGetTexRef(&tex, *(dev->module()), name);
  err = ld->cuTexRefSetAddress(0, tex, (CUdeviceptr) dptr, size);
  err = ld->cuTexRefSetAddressMode(tex, 0, CU_TR_ADDRESS_MODE_WRAP);
  err = ld->cuTexRefSetFilterMode(tex, CU_TR_FILTER_MODE_LINEAR);
  err = ld->cuTexRefSetFlags(tex, CU_TRSF_NORMALIZED_COORDINATES);
  err = ld->cuTexRefSetFormat(tex, type == iris_int ? CU_AD_FORMAT_SIGNED_INT32 : CU_AD_FORMAT_FLOAT, 1);
  return IRIS_SUCCESS;
}

int main(int argc, char** argv) {
  setenv("IRIS_ARCHS", "cuda", 1);
  iris_init(&argc, &argv, true);

  iris_register_command(0xdeadcafe, iris_cuda, cmd_handler_nv);

  const char* name = "Yahoo";
  size_t name_len = strlen(name) + 1;
  int type = iris_int;
  const void* dptr = (void*) 0xbeefbeef;
  size_t size = 1010;

  size_t params_size = sizeof(name_len) + name_len + sizeof(type) + sizeof(dptr) + sizeof(size);
  char* params = (char*) malloc(params_size);
  int off = 0;
  memcpy(params + off, &name_len, sizeof(name_len));
  off += sizeof(name_len);
  memcpy(params + off, name, name_len);
  off += name_len;
  memcpy(params + off, &type, sizeof(type));
  off += sizeof(type);
  memcpy(params + off, &dptr, sizeof(dptr));
  off += sizeof(dptr);
  memcpy(params + off, &size, sizeof(size));
  off += sizeof(size);

  iris_task task;
  iris_task_create(&task);
  iris_task_custom(task, 0xdeadcafe, params, params_size);
  iris_task_submit(task, iris_nvidia, NULL, true);

  iris_finalize();

  return iris_error_count();
}

