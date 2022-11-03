#ifndef IRIS_SRC_RT_PRESENT_TABLE_H
#define IRIS_SRC_RT_PRESENT_TABLE_H

#include "Config.h"
#include <map>

namespace iris {
namespace rt {

class BaseMem;

typedef struct _PresentTableEntity {
  size_t size;
  BaseMem* mem;
} PresentTableEntity;

class PresentTable {
public:
  PresentTable();
  ~PresentTable();

  int Add(void* host, size_t size, BaseMem* mem);
  BaseMem* Get(void* host, size_t* off);
  BaseMem* Remove(void* host);

private:
  std::map<void*, PresentTableEntity*> entities_;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_PRESENT_TABLE_H */
