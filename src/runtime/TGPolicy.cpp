#include "TGPolicy.h"
#include "Command.h"
#include "Task.h"
#include "Debug.h"
#include "Scheduler.h"

namespace iris {
namespace rt {

TGPolicy::TGPolicy() {
}

bool TGPolicy::IsKernelSupported(Task *task, Device *dev) {
    return task->IsKernelSupported(dev);
}

TGPolicy::~TGPolicy() {
}


} /* namespace rt */
} /* namespace iris */


