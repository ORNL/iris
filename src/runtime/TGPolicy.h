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
                virtual void Schedule(Graph *graph, Device** devs, int* ndevs) = 0;
                void SetScheduler(Scheduler* scheduler);

            protected:
                Device** devices() const { return  devs_; }
                Device* device(int i) const { return  devs_[i]; }
                int ndevices() const { return ndevs_; }

            protected:
                Scheduler* scheduler_;
                Device** devs_;
                int ndevs_;
        };

    } /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_TGPOLICY_H */


