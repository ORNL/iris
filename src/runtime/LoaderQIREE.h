#ifndef IRIS_SRC_RT_LOADER_QIREE_H
#define IRIS_SRC_RT_LOADER_QIREE_H

#include "Loader.h"
#include "HostInterface.h"
//#include <iris/qiree/qiree.h>

typedef struct CQiree_s CQiree;

enum QireeReturnCode
{
    QIREE_SUCCESS = 0,
    QIREE_FAIL_LOAD = 1,
};

namespace iris {
namespace rt {

class LoaderQIREE : public HostInterfaceClass {
public:
  LoaderQIREE();
  ~LoaderQIREE();

  //const char* library_precheck() { return "cuInit"; }
  const char* library();// { return "libqiree.so"; }
  int LoadFunctions();
  CQiree *(*qiree_create)();
  void (*qiree_destroy)(CQiree *manager);
  int (*qiree_load_module_from_memory)(CQiree *manager, char const *data_contents, size_t length);
  int (*qiree_load_module_from_file)(CQiree *manager, char const *filename);
  int (*qiree_num_quantum_reg)(CQiree *manager, int *result);
  int (*qiree_num_classical_reg)(CQiree *manager, int *result);
  int (*qiree_max_result_items)(CQiree *manager, int num_shots, int *result);
  int (*qiree_setup_executor)(CQiree *manager, char const *backend, char const *config_json);
  int (*qiree_execute)(CQiree *manager, int num_shots);

  int (*qiree_save_result_items)(CQiree *manager, size_t max, uint64_t *encoded);

};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_LOADER_QIREE_H */

