#ifndef BRISBANE_SRC_RT_STRUCTS_H
#define BRISBANE_SRC_RT_STRUCTS_H

namespace brisbane {
namespace rt {
class Kernel;
class Mem;
class Task;
class Graph;
} /* namespace rt */
} /* namespace brisbane */

struct _brisbane_task {
  brisbane::rt::Task* class_obj;
};

struct _brisbane_kernel {
  brisbane::rt::Kernel* class_obj;
};

struct _brisbane_mem {
  brisbane::rt::Mem* class_obj;
};

struct _brisbane_graph {
  brisbane::rt::Graph* class_obj;
};

#endif /* BRISBANE_SRC_RT_STRUCTS_H */
