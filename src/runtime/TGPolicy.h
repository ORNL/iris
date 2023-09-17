#ifndef IRIS_SRC_RT_TGPOLICY_H
#define IRIS_SRC_RT_TGPOLICY_H

#define REGISTER_CUSTOM_TGPOLICY(class_name, name)              \
iris::rt::class_name name;                              \
extern "C" void* name ## _instance() { return (void*) &name; }

namespace iris {
    namespace rt {

        class Device;
        class Scheduler;
        class Task;
        class Graph;

        class TGPolicy {
            public:
                TGPolicy();
                virtual ~TGPolicy();

                virtual void Init(void* arg) {}
                virtual bool IsKernelSupported(Task *task, Device *dev);
                virtual void Schedule(Graph *graph, Device** devs, int* ndevs) = 0;

            protected:
        };

    } /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_TGPOLICY_H */


