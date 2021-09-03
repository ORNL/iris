#include "PresentTable.h"
#include "Config.h"
#include "Debug.h"

namespace brisbane {
namespace rt {

PresentTable::PresentTable() {

}

PresentTable::~PresentTable() {

}

int PresentTable::Add(void* host, size_t size, Mem* mem) {
  if (entities_.find(host) != entities_.end()) {
    _error("%p", host);
    return BRISBANE_ERR;
  }
  PresentTableEntity* entity = new PresentTableEntity;
  entity->size = size;
  entity->mem = mem;
  entities_[host] = entity;
  return BRISBANE_OK;
}

Mem* PresentTable::Get(void* host, size_t* off) {
  char* input = (char*) host;
  for (std::map<void*, PresentTableEntity*>::iterator I = entities_.begin(), E = entities_.end(); I != E; ++I) {
    char* s0 = (char*) I->first;
    char* s1 = (char*) s0 + I->second->size;
    if (input >= s0 && input < s1) {
      if (off) *off = input - s0;
      return I->second->mem;
    }
  }
  return NULL;
}

Mem* PresentTable::Remove(void* host) {
  char* input = (char*) host;
  for (std::map<void*, PresentTableEntity*>::iterator I = entities_.begin(), E = entities_.end(); I != E; ++I) {
    char* s0 = (char*) I->first;
    char* s1 = (char*) s0 + I->second->size;
    if (input >= s0 && input < s1) {
      Mem* ret = I->second->mem;
      entities_.erase(I);
      return ret;
    }
  }
  return NULL;
}

} /* namespace rt */
} /* namespace brisbane */

