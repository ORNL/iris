#include <iris/iris.h>
#include <iris/rt/DeviceCUDA.h>
#include <iris/rt/LoaderCUDA.h>
#include <iris/rt/Command.h>
#include <iris/rt/Task.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SIZE 128

int task_hook_pre(void* task) {
  iris::rt::Task* t = (iris::rt::Task*) task;
  printf("[%s:%d] enter task[%lu]\n", __FILE__, __LINE__, t->uid());
  return 1;
}

int task_hook_post(void* task) {
  iris::rt::Task* t = (iris::rt::Task*) task;
  printf("[%s:%d] exit task[%lu]\n", __FILE__, __LINE__, t->uid());
  return 1;
}

int cmd_hook_pre(void* cmd) {
  iris::rt::Command* c = (iris::rt::Command*) cmd;
  printf("[%s:%d] enter cmd[%x]\n", __FILE__, __LINE__, c->type());
  return 1;
}

int cmd_hook_post(void* cmd) {
  iris::rt::Command* c = (iris::rt::Command*) cmd;
  printf("[%s:%d] exit cmd[%x]\n", __FILE__, __LINE__, c->type());
  return 1;
}

int main(int argc, char** argv) {
  iris_init(&argc, &argv, true);

  iris_register_hooks_task(task_hook_pre, task_hook_post);
  iris_register_hooks_command(cmd_hook_pre, cmd_hook_post);

  void* host = malloc(SIZE);
  iris_mem mem;
  iris_mem_create(SIZE, &mem);
  iris_task task;
  iris_task_create(&task);
  iris_task_h2d_full(task, mem, host);
  iris_task_submit(task, iris_random, NULL, true);

  free(host);

  iris_finalize();

  int errors = iris_error_count();
  printf("Errors: %d\n", errors);
  return errors;
}

