#ifndef IRIS_INCLUDE_IRIS_IRIS_HPP
#define IRIS_INCLUDE_IRIS_IRIS_HPP

#ifndef DOXYGEN_SHOULD_SKIP_THIS
#include <iris/iris_runtime.h>
#endif
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
    /**
      * IRIS Platform Class for application
      */
    class Platform {
        public:
            /**@brief Platform Classs constructor.
             *
             */
            Platform() {
                finalized_ = false;
            }

            /**@brief Platform Classs destructor
             *
             */
            ~Platform() {
                if (!finalized_) finalize();
            }

            /**@brief Initialize IRIS platform 
              *
              * @param argc pointer to the number of arguments
              * @param argv argument array
              * @param sync 0: non-blocking, 1: blocking
              * @return This functions return an error value. IRIS_SUCCESS, IRIS_ERROR
              */
            int init(int* argc, char*** argv, bool sync=false) {
                return iris_init(argc, argv, sync ? 1 : 0);
            }

            /**@brief Terminates the IRIS platform .
             *
             * this funciton put end to IRIS execution environment.
             *
             * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
             */
            int finalize() {
                if (finalized_) return IRIS_ERROR;
                int ret = iris_finalize();
                finalized_ = true;
                return ret;
            }

            /**@brief Return number of errors occurred in IRIS
             *
             * @return This function returns the number of errors
             */
            int error_count() {
                return iris_error_count();
            }

        private:
            bool finalized_;

    };
    /**
      * BaseMem Class for IRIS memory objects. It is a base class IRIS memory objects 
      */
    class BaseMem {
        public:
            /**@brief BaseMem Classs constructor.
             *
             */
            BaseMem() {} 
            /**@brief BaseMem Classs destructor.
             *
             */
            virtual ~BaseMem() {} 
            /**@brief get C structure IRIS memory object 
             *
             * @return C structure IRIS memory object
             */
            iris_mem_type mem() { return mem_; }
            /**@brief get C structure IRIS memory object pointer
             *
             * @return C structure IRIS memory object pointer
             */
            iris_mem_type *mem_ptr() { return &mem_; }
        protected:
            iris_mem_type mem_;
    };

    /**
      * Mem class for IRIS classic memory objects. It is a derived from the BaseMem class
      */
    class Mem : public BaseMem {
        public:
            /**@brief Mem Classs constructor.
             *
             */
            Mem(size_t size);
            /**@brief Mem Classs destructor.
             *
             */
            virtual ~Mem();
    };
    /**
      * DMem class for IRIS (Novel data memory) memory objects. It is a derived from the BaseMem class
      */
    class DMem : public BaseMem {
        public:
            /**@brief DMem Classs constructor.
             *
             * Creates IRIS data memory object for a given size
             *
             * @param host host pointer of the data structure
             * @param size size of the memory
             */
            DMem(void *host, size_t size);
            /**@brief DMem Classs constructor.
             *
             * @param host host memory pointer
             * @param off host memory pointer
             * @param host_size indexes to specify sizes from host memory
             * @param dev_size indexes to specify sizes from device memory
             * @param elem_size element size
             * @param dim dimension
             */
            DMem(void *host, size_t *off, size_t *host_size, size_t *dev_size, size_t elem_size, int dim);
            /**@brief DMem Classs destructor.
             *
             */
            virtual ~DMem();
            /**@brief Update DMem object with new host memory pointer
             *
             * @param host host memory pointer
             * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
             */
            int update(void *host);
            /**@brief Update DMem object with new host memory pointer
             *
             * @param reset the data memory object and all its internal dirty bit flags
             * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
             */
            int reset(bool reset);
            /**@brief enable decomposition along the outer dimension
             *
             * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
             */
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

