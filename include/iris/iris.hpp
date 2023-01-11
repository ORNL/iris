#ifndef IRIS_INCLUDE_IRIS_IRIS_HPP
#define IRIS_INCLUDE_IRIS_IRIS_HPP

#include <iris/iris_runtime.h>
#include <assert.h>

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

        private:
            bool finalized_;

    };
    class BaseMem {
        public:
            BaseMem() {} 
            iris_mem mem() { return mem_; }
        protected:
            iris_mem mem_;
    };

    class Mem : public BaseMem {
        public:
            Mem(size_t size) {
                iris_mem_create(size, &mem_);
            }

            ~Mem() {
                iris_mem_release(mem_);
            }
    };
    class DMem : public BaseMem {
        public:
            DMem(void *host, size_t size) {
                assert(iris_data_mem_create(&mem_, host, size) == IRIS_SUCCESS);
            }
            DMem(void *host, size_t *off, size_t *host_size, size_t *dev_size, size_t elem_size, int dim) {
                assert(iris_data_mem_create_tile(&mem_, host, off, host_size, dev_size, elem_size, dim) == IRIS_SUCCESS);
            }
            int update(void *host) {
                return iris_data_mem_update(mem_, host);
            }
            int reset(bool reset) { 
                return iris_data_mem_init_reset(mem_, reset); 
            }
            int enable_outer_dim_regions() {
                return iris_data_mem_enable_outer_dim_regions(mem_);
            }
            ~DMem() {
                //iris_mem_release(mem_);
            }
    };
    class DMemRegion {
        public:
            DMemRegion(iris_mem root_mem, int region) {
                iris_data_mem_create_region(&mem_, root_mem, region);
            }
            DMemRegion(DMem *root_mem, int region) {
                dmem_ = root_mem;
                iris_data_mem_create_region(&mem_, root_mem->mem(), region);
            }
            ~DMemRegion() {
                //iris_mem_release(mem_);
            }
            DMem *parent() { return dmem_; }
            iris_mem dmem() { return iris_get_dmem_for_region(mem_); }
            iris_mem mem() { return mem_; }

        private:
            DMem *dmem_;
            iris_mem mem_;
    };

    class Task {
        public:
            Task() {
                iris_task_create(&task_);
            }

            ~Task() {
                iris_task_release(task_);
            }

            int h2d(Mem* mem, size_t off, size_t size, void* host) {
                return iris_task_h2d(task_, mem->mem(), off, size, host);
            }

            int h2d_full(Mem* mem, void* host) {
                return iris_task_h2d_full(task_, mem->mem(), host);
            }

            int d2h(Mem* mem, size_t off, size_t size, void* host) {
                return iris_task_d2h(task_, mem->mem(), off, size, host);
            }

            int d2h_full(Mem* mem, void* host) {
                return iris_task_d2h_full(task_, mem->mem(), host);
            }

            int flush_out(iris_mem mem) {
                return iris_task_dmem_flush_out(task_, mem);
            }

            int flush_out(DMem & mem) {
                return iris_task_dmem_flush_out(task_, mem.mem());
            }

            int flush_out(DMem * mem) {
                return iris_task_dmem_flush_out(task_, mem->mem());
            }

            int kernel(const char* kernel, int dim, size_t* off, size_t* gws, size_t* lws, int nparams, void** params, int* params_info) {
                void** new_params = new void*[nparams];
                for (int i = 0; i < nparams; i++) {
                    if (params_info[i] == iris_w ||
                            params_info[i] == iris_r ||
                            params_info[i] == iris_rw) {
                        new_params[i] = ((BaseMem*) params[i])->mem();
                    } else new_params[i] = params[i];
                }
                int ret = iris_task_kernel(task_, kernel, dim, off, gws, lws, nparams, new_params, params_info);
                delete[] new_params;
                return ret;
            }

            int submit(int device, const char* opt, bool sync) {
                return iris_task_submit(task_, device, opt, sync ? 1 : 0);
            }

        private:
            iris_task task_;

    };

} // namespace iris

#endif /* IRIS_INCLUDE_IRIS_IRIS_HPP */

