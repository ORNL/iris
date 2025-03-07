#ifndef IRIS_INCLUDE_IRIS_IRIS_HPP
#define IRIS_INCLUDE_IRIS_IRIS_HPP

#ifndef DOXYGEN_SHOULD_SKIP_THIS
#include <iris/iris_runtime.h>
#endif
#include <assert.h>
#include <vector>
#include <iostream>
#include <iomanip>
using namespace std;
#ifdef ENABLE_SMART_PTR
#include <iris/RetainableBase.hpp>
#include <memory>
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
             * this function put end to IRIS execution environment.
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

            /**@brief Return number of devices
             * @return This function returns the number of devices
             */
            int ndevs() {
                return iris_ndevices();
            }

            /**@brief Return number of streams
             * @return This function returns the number of streams
             */
            int nstreams() {
                return iris_nstreams();
            }

            /**@brief Return number of streams
             * @param This function sets the number of streams
             */
            void set_nstreams(int n) {
                iris_set_nstreams(n);
            }

            /**@brief Return number of copy streams
             * @param This function sets the number of copy streams
             */
            void set_ncopy_streams(int n) {
                iris_set_ncopy_streams(n);
            }

            /**@brief Return number of copy streams
             * @return This function returns the number of copy streams
             */
            int ncopy_streams() {
                return iris_ncopy_streams();
            }

            /**@brief Prints an overview of the system available to IRIS; specifically
             * platforms, devices and their corresponding backends.
             * It is logged to standard output.
             */
            void overview() {
                 iris_overview();
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
            /**@brief get size of memory object
             *
             * @return size of memory object for the given mem
             */
            size_t size() {
                return iris_mem_get_size(mem_); 
            }
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
        bool is_usm_set_;
        public:
            /**@brief DMem Classs constructor.
             *
             * Creates IRIS data memory object for a given size
             *
             * @param host host pointer of the data structure
             * @param size size of the memory
             */
            DMem(size_t size, bool usm_flag=false);
            /**@brief DMem Classs constructor.
             *
             * Creates IRIS data memory object for a given size
             *
             * @param host host pointer of the data structure
             * @param size size of the memory
             */
            DMem(void *host, size_t size, bool usm_flag=false);
            /**@brief DMem Classs constructor.
             *
             * Creates IRIS data memory object for a given size
             *
             * @param host host pointer of the data structure
             * @param size size of the memory
             * @param device symbol name (global scope) used for kernel
             */
            DMem(void *host, size_t size, const char *symbol);
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

            template<typename T>
            static DMem *create(bool usm_flag=false) { return DMem::create((void *)NULL, sizeof(T), usm_flag); }
            template<typename T>
            static DMem *create(T & host, bool usm_flag=false) { return DMem::create((void *)&host, sizeof(T), usm_flag); }
            template<typename T>
            static DMem *create(T *host, bool usm_flag=false) { return DMem::create((void *)host, sizeof(T), usm_flag); }
            static DMem *create(void *host, size_t size, bool usm_flag=false) {
                DMem *dmem = new DMem(host, size, usm_flag);
                return dmem;
            }
            static void release(DMem *dmem) {
                delete dmem;
            }
            unsigned long uid() {
                return mem_.uid;
            }
            /**@brief DMem Classs destructor.
             *
             */
            virtual ~DMem();
            /**@brief Fetch DMem object content and copy it to host pointer 
             *
             * @param host host memory pointer
             * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
             */
            int fetch(void *host);
            /**@brief Fetch DMem object content and copy it to host pointer 
             *
             * @param host host memory pointer
             * @return This function returns an integer indicating IRIS_SUCCESS or IRIS_ERROR .
             */
            int fetch(void *host, size_t size);
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
            /**@brief get host pointer for the given DMEM object
              *
              * @return Returns host pointer of void *
              */
            void *host(bool valid=false);
            void *fetch_host();
            void *fetch_host(size_t size);
            void add_child(DMem **tmem) {
                // This is required for DMEM of structure where structure 
                // members have arrays which are DMEM objects.
                void *base = host(); 
                DMem **dmem = (DMem **)tmem;
                uintptr_t address1 = reinterpret_cast<uintptr_t>(base);
                uintptr_t address2 = reinterpret_cast<uintptr_t>(dmem);
                size_t offset = static_cast<size_t>(address2 - address1);
                iris_dmem_add_child(mem(), (*dmem)->mem(), offset);
            }

    };
    template<typename T> 
    class DMemStruct : public DMem {
        public:
            DMemStruct(bool usm_flag=false) : DMem(sizeof(T), usm_flag) { }
            DMemStruct(T & data, bool usm_flag=false) : DMem((void *)&data, sizeof(T), usm_flag) { }
            DMemStruct(T *data, bool usm_flag=false) : DMem((void *)data, sizeof(T), usm_flag) { }
            DMemStruct(T & data, const char *symbol) : DMem((void *)&data, sizeof(T), symbol) { }
            DMemStruct(T *data, const char *symbol) : DMem((void *)data, sizeof(T), symbol) { }
            virtual ~DMemStruct() { }
            static DMemStruct *create(bool usm_flag=false) { return new DMemStruct<T>(usm_flag); }
            static DMemStruct *create(T & host, bool usm_flag=false) { return new DMemStruct<T>(host, usm_flag); }
            static DMemStruct *create(T * host, bool usm_flag=false) { return new DMemStruct<T>(host, usm_flag); }
            int update(T *host) { return DMem::update(host); }
            int update(T & host) { return DMem::update(&host); }
            T *host(bool valid=false) { return (T *) DMem::host(valid); }
            T *fetch_host() { return (T *) DMem::fetch_host(); }
            T *fetch_host(size_t size) { return (T *) DMem::fetch_host(size); }
    };
    template<typename T> 
    class DMemArray : public DMem {
        public:
            DMemArray(size_t count, bool usm_flag=false) : DMem(count*sizeof(T), usm_flag) { }
            DMemArray(int count, bool usm_flag=false) : DMem(count*sizeof(T), usm_flag) { }
            DMemArray(T *data, size_t count, bool usm_flag=false) : DMem((void *)data, sizeof(T)*count, usm_flag) { }
            DMemArray(T *data, int count, bool usm_flag=false) : DMem((void *)data, sizeof(T)*count, usm_flag) { }
            DMemArray(T data[], bool usm_flag=false) : DMem((void *)data, sizeof(data), usm_flag) { }
            //DMemArray(T data[], size_t count) : DMem((void *)data, sizeof(T)*count) { }
            virtual ~DMemArray() { }
            static DMemArray *create(int count, bool usm_flag=false) { return new DMemArray(count, usm_flag); }
            static DMemArray *create(size_t count, bool usm_flag=false) { return new DMemArray(count, usm_flag); }
            static DMemArray *create(T host[], bool usm_flag=false) { return new DMemArray(host, usm_flag); }
            static DMemArray *create(T *host, size_t count, bool usm_flag=false) { return new DMemArray(host, count, usm_flag); }
            static DMemArray *create(T *host, int count, bool usm_flag=false) { return new DMemArray(host, count, usm_flag); }
            int update(T *host) { return DMem::update(host); }
            T *host(bool valid=false) { return (T *) DMem::host(valid); }
            T *fetch_host() { return (T *) DMem::fetch_host(); }
            T *fetch_host(size_t size) { return (T *) DMem::fetch_host(size); }
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
            void disable_async();
            int set_order(int *order);
            int h2d(DMem* mem);
            int d2h(DMem* mem);
            int dmem2dmem(DMem* src_mem, DMem *dst_mem);
            int h2d(Mem* mem, size_t off, size_t size, void* host);
            int h2d_full(Mem* mem, void* host);
            int h2broadcast(Mem* mem, size_t off, size_t size, void* host);
            int h2broadcast_full(Mem* mem, void* host);
            int d2h(Mem* mem, size_t off, size_t size, void* host);
            int d2h_full(Mem* mem, void* host);
            int flush_out(DMem & mem);
            int kernel(const char* kernel, int dim, size_t* off, size_t* gws, size_t* lws, int nparams, void** params, int* params_info);
            int submit(int device=iris_default, const char* opt=NULL, bool sync=false);
            void depends_on(int ntasks, Task **tasks);
            void depends_on(std::vector<Task *> tasks);
            void depends_on(Task & task);
            void disable_launch() { iris_task_kernel_launch_disabled(task_, true); }
            iris_task_type task() { return task_; }
            int wait();
            int add_hidden_dmem(DMem & dmem, int mode);
            int add_hidden_dmem(DMem *dmem, int mode);
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
    template<typename DType>
    static DType *AllocateArray(size_t SIZE, DType init=0)
    {
        DType *A = (DType *)malloc ( SIZE *  sizeof(DType));
        for (size_t i = 0; i < SIZE; i++ )
            A[i] = init;
        return A;
    }
    template<typename DType>
    static DType *AllocateRandomArray(size_t SIZE)
    {
        DType *A = (DType *)malloc ( SIZE *  sizeof(DType));
        for (size_t i = 0; i < SIZE; i++ )
        {
            int a = rand() % 5+1;
            A[i] = (DType)a;
        }
        return A;
    }
    template<class DType>
    static void PrintVectorLimited(vector<DType> & A, const char *description, size_t limit=8)
        {
            printf("%s (%d)\n", description, A.size());
            size_t print_cols = A.size()>limit?limit:A.size();
            for (size_t j = 0; j < print_cols; j++) {
                cout << setw(6) << A[j] << " ";
            }
            cout << endl;
        }
    template<class DType>
    static void PrintVectorFull(vector<DType> &A, const char *description)
        {
            printf("%s (%d)\n", description, A.size());
            for (size_t j = 0; j < A.size(); j++) {
                cout << setw(6) << A[j] << " ";
            }
            cout << endl;
        }

    template<class DType>
    static void PrintArrayLimited(DType *A, size_t N, const char *description, size_t limit=8)
        {
            printf("%s data:%p (%ld)\n", description, A, N);
            size_t print_cols = N>limit?limit:N;
            for (size_t j = 0; j < print_cols; j++) {
                cout << setw(6) << A[j] << " ";
            }
            cout << endl;
        }
    template<class DType>
    static void PrintAarrayFull(DType *A, size_t N, const char *description)
        {
            printf("%s data:%p (%ld)\n", description, A, N);
            for (size_t j = 0; j < N; j++) {
                cout << setw(6) << A[j] << " ";
            }
            cout << endl;
        }

    template<class DType>
    static void PrintMatrixLimited(DType *A, size_t M, size_t N, const char *description, size_t limit=8)
        {
            printf("%s data:%p (%ld, %ld)\n", description, A, M, N);
            size_t print_rows = M>limit?limit:M;
            size_t print_cols = N>limit?limit:N;
            for(size_t i=0; i<print_rows; i++) {
                for (size_t j = 0; j < print_cols; j++) {
                    cout << setw(6) << A[i*N+j] << " ";
                }
                cout << endl;
            }
        }
    template<class DType>
    static void PrintMatrixFull(DType *A, size_t M, size_t N, const char *description)
        {
            printf("%s data:%p (%ld, %ld)\n", description, A, M, N);
            for(size_t i=0; i<M; i++) {
                for (size_t j = 0; j < N; j++) {
                    cout << setw(6) << A[i*N+j] << " ";
                }
                cout << endl;
            }
        }
    template<class DType>
        static void MemSet(DType *A, size_t M, DType value)
        {
            for(size_t i=0; i<M; i++) {
                A[i] = value;
            }
        }
    template<class DType>
        static size_t CompareMatrix(DType *A, DType *ref, size_t M, size_t N, DType & error_data)
        {
            size_t ERROR = 0;
            error_data = (DType) 0;
            printf("Checking errors\n");
            for(size_t i=0; i < M*N; i++) {
                if (A[i] != ref[i]) {
                    ERROR++;
                    DType diff = A[i] - ref[i];
                    error_data += (diff < (DType)0) ? -diff : diff;
                }
            }
            printf("Total errors:%ld\n", ERROR);
            if (ERROR>0)
                printf("Mean error:%f\n", error_data/ERROR);
            return ERROR;
        }
    template<class DType>
        static size_t CompareArray(DType *A, DType *ref, size_t M, DType & error_data)
        {
            size_t ERROR = 0;
            error_data = (DType) 0;
            printf("Checking errors\n");
            for(size_t i=0; i < M; i++) {
                if (A[i] != ref[i]) {
                    ERROR++;
                    DType diff = A[i] - ref[i];
                    error_data += (diff < (DType)0) ? -diff : diff;
                }
            }
            printf("Total errors:%ld\n", ERROR);
            if (ERROR>0)
                printf("Mean error:%f\n", error_data/ERROR);
            return ERROR;
        }

} // namespace iris

#endif /* IRIS_INCLUDE_IRIS_IRIS_HPP */

