#ifndef IRIS_INCLUDE_IRIS_IRIS_HPP
#define IRIS_INCLUDE_IRIS_IRIS_HPP

#include <iris/iris_runtime.h>
#include <assert.h>
#include <vector>
#ifdef ENABLE_SMART_PTR
#include <iris/RetainableBase.hpp>
#include <memory>
using namespace std;
#define ENABLE_SMART_PTR_MEM
#define ENABLE_SMART_PTR_TASK
typedef iris::rt::RetainableBase RetainableBase;
typedef shared_ptr<RetainableBase> iris_task_sp; 
typedef shared_ptr<RetainableBase> iris_mem_sp; 
#define iris_mem_type iris_mem_sp
#define iris_task_type iris_task_sp
#else // ENABLE_SMART_PTR
#define iris_mem_type iris_mem
#define iris_task_type iris_task
#endif //ENABLE_SMART_PTR
namespace iris {
    class Platform {
        public:
            Platform() {
                finalized_ = false;
            }

            ~Platform() {
                if (!finalized_) finalize();
            }

            int init(int* argc, char*** argv, bool sync) {
                return iris_init(argc, argv, sync ? 1 : 0);
            }

            int finalize() {
                if (finalized_) return IRIS_ERROR;
                int ret = iris_finalize();
                finalized_ = true;
                return ret;
            }

            int error_count() {
                return iris_error_count();
            }

        private:
            bool finalized_;

    };
    class BaseMem {
        public:
            BaseMem() {} 
            virtual ~BaseMem() {} 
            iris_mem_type mem() { return mem_; }
            iris_mem_type *mem_ptr() { return &mem_; }
        protected:
            iris_mem_type mem_;
    };

    class Mem : public BaseMem {
        public:
            Mem(size_t size);
            virtual ~Mem();
    };
    class DMem : public BaseMem {
        public:
            DMem(void *host, size_t size);
            DMem(void *host, size_t *off, size_t *host_size, size_t *dev_size, size_t elem_size, int dim);
            virtual ~DMem();
            int update(void *host);
            int reset(bool reset);
            int enable_outer_dim_regions();
    };
    class DMemRegion : public BaseMem {
        public:
            DMemRegion(iris_mem_type root_mem, int region);
            DMemRegion(DMem *root_mem, int region);
            virtual ~DMemRegion() {}
            DMem *parent() { return dmem_; }

        private:
            DMem *dmem_;
    };

    class Task {
        public:
            Task(const char *name=NULL, bool perm=false, bool retainable=false);
            virtual ~Task() { }
            int set_order(int *order);
            int h2d(Mem* mem, size_t off, size_t size, void* host);
            int h2d_full(Mem* mem, void* host);
            int h2broadcast(Mem* mem, size_t off, size_t size, void* host);
            int h2broadcast_full(Mem* mem, void* host);
            int d2h(Mem* mem, size_t off, size_t size, void* host);
            int d2h_full(Mem* mem, void* host);
            int flush_out(DMem & mem);
            int kernel(const char* kernel, int dim, size_t* off, size_t* gws, size_t* lws, int nparams, void** params, int* params_info);
            int submit(int device, const char* opt, bool sync);
            void depends_on(int ntasks, Task **tasks);
            void depends_on(std::vector<Task *> tasks);
            void depends_on(Task & task);
            void disable_launch() { iris_task_kernel_launch_disabled(task_, true); }
            iris_task_type task() { return task_; }
        private:
            iris_task_type task_;
            bool retainable_;
    };

    class Graph {
        public:
            Graph(bool retainable=false);
            virtual ~Graph();
            void retainable();
            int add_task(Task & task, int device, const char *opt=NULL);
            int set_order(int *order);
            int submit(int device, int sync=false);
            int wait();
            int release();
            iris_graph graph() { return graph_; }
        private:
            bool retainable_;
            bool is_released_;
            iris_graph graph_;
    };
} // namespace iris

#endif /* IRIS_INCLUDE_IRIS_IRIS_HPP */

