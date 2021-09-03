#ifndef BRISBANE_SRC_RT_PRESENT_TABLE_H
#define BRISBANE_SRC_RT_PRESENT_TABLE_H

#include <map>

namespace brisbane {
namespace rt {

class Mem;

typedef struct _PresentTableEntity {
  size_t size;
  Mem* mem;
} PresentTableEntity;

class PresentTable {
public:
  PresentTable();
  ~PresentTable();

  int Add(void* host, size_t size, Mem* mem);
  Mem* Get(void* host, size_t* off);
  Mem* Remove(void* host);

private:
  std::map<void*, PresentTableEntity*> entities_;
};

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_SRC_RT_PRESENT_TABLE_H */
