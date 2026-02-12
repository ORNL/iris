#pragma once
#include <iostream>
#include <vector>
#include <atomic>
#include <unistd.h>
#include <sys/mman.h>
#include <cstring>
#include <thread>
#include <unordered_map>
#include <memory>
#include <cassert>
#include <signal.h>
#include <random>
#include <sstream>
#include <sys/mman.h>
#include <sys/stat.h>        /* For mode constants */
#include <fcntl.h>           /* For O_* constants */

#include "DeviceQIREE.h"
#include "Debug.h"

#define QMGR_RUN_QIREE_OUT(OUT, DEFAULT, X, ...) \
    if (qmgr->ld_->X == NULL) { \
        _error(#X " not found in QIREE library"); \
        OUT = DEFAULT; \
    } else { \
        OUT = qmgr->ld_->X(__VA_ARGS__); \
    }
#define QMGR_RUN_QIREE(X, ...) \
    if (qmgr->ld_->X == NULL) { \
        _error(#X " not found in QIREE library"); \
    } else { \
        ASSERT(qmgr->ld_->X(__VA_ARGS__) == IRIS_SUCCESS); \
    }
#define QMGR_RUN_QIREE_NO_RETURN(X, ...) \
    if (qmgr->ld_->X == NULL) { \
        _error(#X " not found in QIREE library"); \
    } else { \
        qmgr->ld_->X(__VA_ARGS__); \
    }

namespace iris {
    namespace rt {

        enum QireeCmdType {
            QireeCreate = 0,
            QireeDestroy = 1,
            QireeLoad = 2,
            QireeMaxResults = 3,
            QireeSetupExecutor = 4,
            QireeExecute = 5
        };

        enum CmdStatus {
            Idle        = 0,
            InputReady  = 1,
            Processing  = 2,
            OutputReady = 3,
            Exit        = 4
        };

        struct SharedHeader {
            alignas(64) std::atomic<uint32_t> status;
            uint32_t cmdId;

            uint32_t numInputs;
            uint32_t inputSizes[4];

            uint32_t numOutputs;
            uint32_t outputSizes[4];
            size_t outputSizes64[4];

            int fd;
            char cmd_shm_name[128];
            CmdStatus loadStatus() {
                return static_cast<CmdStatus>(status.load());
            }
            void storeStatus(CmdStatus st) {
                status.store(static_cast<uint32_t>(st));
            }
            const char *status_str() {
                switch(status.load()) {
                    case Idle: return "Idle";
                    case InputReady: return "InputReady";
                    case Processing: return "Processing";
                    case OutputReady: return "OutputReady";
                    case Exit: return "Exit";
                    default: return "Error";
                }
            }
            void set_shm_name(const char *shm_name) {
                if (!shm_name) {
                    cmd_shm_name[0] = '\0';
                    return;
                }
                strcpy(cmd_shm_name, shm_name);
            }
            const char *shm_name() { return cmd_shm_name; }
        };

        class CommandChannel {
            public:
                std::string random_name(const char* prefix, int devno) {
                    static std::random_device rd;
                    static std::mt19937_64 gen(rd());
                    static std::uniform_int_distribution<uint64_t> dis;

                    std::ostringstream os;
                    os << prefix << "_" << std::hex << dis(gen) << "_d" << devno;
                    _debug2("Tmp name:%s", os.str().c_str());
                    return os.str();
                }
                CommandChannel(size_t sz, int devno=-1, bool with_fd=false, const char *shm_name=NULL, bool is_parent=true) : size_(sz) {
                    addr_ = NULL;
                    fd_ = -1;
                    is_parent_ = is_parent;
                    shm_name_ = "na";
                    _debug2("Inside now size:%lu\n", size_);
                    if (size_ == 0) return;
                    void* addr;
                    _debug2("with_fd:%d size:%lu devno:%d shm_name:%s\n", with_fd, sz, devno, shm_name);
                    if (with_fd) {
                        if (shm_name == NULL) {
                            shm_name_ = random_name("/qiris_memfd", devno);
                            //fd_ = memfd_create(shm_name_.c_str(), MFD_CLOEXEC);
                            fd_ = shm_open(shm_name_.c_str(), O_CREAT | O_RDWR, 0600);
                            if (fd_ < 0) {
                                perror("shm_open");
                                throw std::runtime_error("shm_open failed");
                            }
                            assert(ftruncate(fd_, size_) == 0);
                            _debug2("Creating parent shmem: %s fd_:%d", shm_name_.c_str(), fd_);
                        }
                        else {
                            //fd_ = fd;
                            fd_ = shm_open(shm_name, O_RDWR, 0600);
                            if (fd_ < 0) {
                                perror("shm_open");
                                throw std::runtime_error("shm_open failed");
                            }
                            _debug2("Creating child shmem: %s fd_:%d", shm_name, fd_);
                            shm_name_ = shm_name;
                        }
                        addr = mmap(nullptr, size_,
                                PROT_READ | PROT_WRITE,
                                MAP_SHARED,
                                fd_, 0);
                    }
                    else {
                        addr = mmap(nullptr, size_,
                                PROT_READ | PROT_WRITE,
                                MAP_SHARED | MAP_ANONYMOUS,
                                -1, 0);
                    }
                    if (addr == MAP_FAILED) {
                        perror("mmap failed");
                        throw std::runtime_error("mmap failed");
                    }
                    addr_ = addr;
                    if (is_parent) 
                        memset(addr_, 0, size_);
                    _debug2("Created mmap size:%lu addr:%p\n", size_, addr_);
                }

                ~CommandChannel() {
                    _debug2(" Now releasing\n");
                    if (addr_) munmap(addr_, size_);
                    if (fd_ != -1) close(fd_);
                    if (is_parent_ && fd_ != -1) shm_unlink(shm_name());
                }
                int fd() { return fd_; }

                void* base() { return addr_; }
                void set_shm_name(std::string shm_name) {
                    if (fd_ == -1) 
                        shm_name_ = "nothing";
                    else
                        shm_name_ = shm_name;
                }
                const char *shm_name() { 
                    return shm_name_.c_str();
                }
            private:
                int fd_;
                bool is_parent_;
                void* addr_;
                size_t size_;
                std::string shm_name_;
        };

        class QireeManager {
            public:
                CQiree *manager_;
                LoaderQIREE *ld_;
        };

        class ProcessQireeBase {
            protected:
                pid_t pid_;
                int devno_;
                bool started_;
                std::unique_ptr<CommandChannel> shared_;
                QireeManager qmgr_;
            public:
                ProcessQireeBase() { 
                    started_ = false; 
                    shared_ = std::make_unique<CommandChannel>(sizeof(SharedHeader));
                    memset((uint8_t*)base(), 0, sizeof(SharedHeader));
                    qmgr_.manager_ = NULL;
                    qmgr_.ld_ = NULL;
                }
                virtual ~ProcessQireeBase() { 

                }
                pid_t pid() { return pid_; }
                int devno() { return devno_; }
                bool started() { return started_; }
                virtual SharedHeader *base() const {
                    void* base = shared_->base();
                    SharedHeader* h = reinterpret_cast<SharedHeader*>(base);
                    return h;
                }
                CmdStatus status() { return base()->loadStatus(); }
                void set_status(CmdStatus s) { base()->storeStatus(s); }
                const char *status_str() { return base()->status_str(); }
        };

        class ICommand {
            private:
            ProcessQireeBase *proc_;
            public:
                ICommand(ProcessQireeBase *proc) { proc_ = proc; mmap_ = NULL; }
                virtual ~ICommand() {
                    if (mmap_ != NULL) delete mmap_;   
                }

                virtual QireeCmdType commandId() const = 0;

                virtual size_t requiredSize() const { return 0; }
                virtual void packInputs() const { }
                virtual void unpackInputs() { }
                virtual void unpackOutputs() { }

                virtual void execute(QireeManager *qmgr) = 0;
                virtual void create(const char *shm_name=NULL) {
                    _debug2("Hello\n\n");
                    _debug2("Calling for shm:%s \n", shm_name);
                    //print("Show");
                    if (requiredSize() == 0) {
                        mmap_ = nullptr;
                        return;
                    }
                    _debug2("Calling for shm:%s\n", shm_name);
                    mmap_ = new CommandChannel(requiredSize(), proc_->devno(), true, shm_name, false);
                    print("After child create");
                }
                virtual void release() {
                    print("Deleting mmap");
                    if (mmap_ != NULL) delete mmap_;
                    mmap_ = NULL;
                }
                virtual void *base() const { 
                    if (mmap_ == NULL) 
                        return NULL; 
                    return mmap_->base(); 
                }
                virtual SharedHeader *hdr_base() const {
                    return proc_->base();
                }
                virtual void print(const char *tag) { 
                    _debug2("Dev:%d Cmd:%s base:%p hdr_base:%p requiredSize:%lu %s state:%s fd:%d", proc_->devno(), name(), base(), hdr_base(), requiredSize(), tag, proc_->status_str(), fd());
                }
                const char *name() {
                    switch(commandId()) {
                        case QireeCreate:           return "QireeCreateCommand";
                        case QireeDestroy:          return "QireeDestroyCommand";
                        case QireeLoad:             return "QireeLoadCommand";
                        case QireeMaxResults:       return "QireeMaxResultsCommand";
                        case QireeSetupExecutor:    return "QireeSetupExecutorCommand";
                        case QireeExecute:          return "QireeExecuteCommand";
                        default:                    return "ErrorCommand";
                    }
                }
                int fd() { 
                    if (mmap_ != NULL) return mmap_->fd(); 
                    else return -1;
                }
                const char *shm_name() { 
                    if (mmap_ != NULL) return mmap_->shm_name(); 
                    else return NULL;
                }
                CommandChannel *mmap_;
        };

        class QireeCreateCommand: public ICommand {
            public:
                QireeCreateCommand(ProcessQireeBase *proc) : ICommand(proc) {}
                QireeCmdType commandId() const { return QireeCreate; }
                void execute(QireeManager *qmgr) {
                    _debug2("Before call");
                    QMGR_RUN_QIREE_OUT(qmgr->manager_, NULL, qiree_create);
                    _debug2("After call");
                }
        };
        class QireeDestroyCommand: public ICommand {
            public:
                QireeDestroyCommand(ProcessQireeBase *proc) : ICommand(proc) {}
                QireeCmdType commandId() const { return QireeDestroy; }
                void execute(QireeManager *qmgr) {
                    _debug2("Before call");
                    QMGR_RUN_QIREE_NO_RETURN(qiree_destroy, qmgr->manager_);
                    _debug2("After call");
                }
        };
        class QireeMaxResultsCommand: public ICommand {
            public:
                QireeMaxResultsCommand(ProcessQireeBase *proc) : ICommand(proc) {}
                uint32_t num_shots_;
                size_t result_items_;

                QireeCmdType commandId() const { return QireeMaxResults; }
                void packInputs() const {
                    SharedHeader *h = hdr_base();
                    h->numInputs = 1;
                    h->inputSizes[0] = num_shots_;
                }
                void unpackInputs() {
                    SharedHeader *h = hdr_base();
                    num_shots_ = h->inputSizes[0];
                }

                void execute(QireeManager *qmgr) {
                    _debug2("Before call num_shots:%d", num_shots_);
                    auto* h = hdr_base();

                    size_t result_items = 0;
                    QMGR_RUN_QIREE(qiree_max_result_items, qmgr->manager_, num_shots_, &result_items);

                    h->numOutputs = 1;
                    h->outputSizes64[0] = result_items;
                    _debug2("After call");
                }

                void unpackOutputs() {
                    auto* h = hdr_base();
                    result_items_ = h->outputSizes64[0];
                }
        };
        class QireeLoadCommand: public ICommand {
            public:
                QireeLoadCommand(ProcessQireeBase *proc) : ICommand(proc) {}
                uint32_t nquant_results_;
                uint32_t nclass_results_;
                size_t   data_len_;
                const char *module_;

                QireeCmdType commandId() const { return QireeLoad; }

                size_t requiredSize() const {
                    return data_len_;
                }

                void print(const char *tag) {
                    ICommand::print(tag);
                    _debug2(" data len: %lu\n", data_len_);
                }

                void packInputs() const {
                    auto* h = hdr_base();
                    h->numInputs = 1;
                    h->inputSizes[0] = (uint32_t) data_len_;
                    uint8_t* ptr = (uint8_t*) base();
                    _debug2("Packing input ptr:%p length:%u\n", ptr, data_len_);
                    memcpy(ptr, module_, data_len_); 
                    _debug2("Copied LLVM module to ptr:%p\n", ptr);
                }
                void unpackInputs() {
                    SharedHeader *h = hdr_base();
                    data_len_ = h->inputSizes[0];
                }

                void execute(QireeManager *qmgr) {
                    //_debug2("Before call");
                    auto* h = hdr_base();
                    //_debug2("SharedHdr: %p", h);
                    uint8_t* ptr = (uint8_t*) base();
                    //_debug2("Base: %p", ptr);
                    const char *llvm_module = (const char *)ptr;
                    _debug2("LLVM: %p", llvm_module);
                    //_debug2("LLVM : %s", llvm_module);
                    uint32_t data_len = h->inputSizes[0];
                    uint32_t nquant_results = 0, nclass_results = 0;
                    _debug2("Params data_len:%u module:%p", data_len, llvm_module);
                    QMGR_RUN_QIREE(qiree_load_module_from_memory, qmgr->manager_, llvm_module, data_len);
                    _debug2("After load module");
                    QMGR_RUN_QIREE(qiree_num_quantum_reg, qmgr->manager_, (int*)&nquant_results);
                    _debug2("After num quant");
                    QMGR_RUN_QIREE(qiree_num_classical_reg, qmgr->manager_, (int*)&nclass_results);
                    _debug2("After num classical");

                    h->numOutputs = 2;
                    h->outputSizes[0] = nquant_results;
                    h->outputSizes[1] = nclass_results;
                    _debug2("After call");
                }

                void unpackOutputs() {
                    auto* h = hdr_base();
                    nquant_results_ = h->outputSizes[0];
                    nclass_results_ = h->outputSizes[1];
                }
        };
        class QireeSetupExecutorCommand : public ICommand {
            public:
                QireeSetupExecutorCommand(ProcessQireeBase *proc) : ICommand(proc) {}
                const char *backend_;
                const char *config_json_;
                uint32_t backend_length_;
                uint32_t config_json_length_;

                QireeCmdType commandId() const { return QireeSetupExecutor; }

                size_t requiredSize() const {
                    return backend_length_ + config_json_length_;
                }

                void packInputs() const {
                    auto* h = hdr_base();
                    h->inputSizes[0] = backend_length_;
                    h->inputSizes[1] = config_json_length_;
                    uint8_t* ptr = (uint8_t*) base();
                    memcpy(ptr, backend_, strlen(backend_)+1); ptr += strlen(backend_)+1;
                    memcpy(ptr, config_json_, strlen(config_json_)+1); 
                    _debug2(" backend:%s\n", backend_);
                    _debug2(" json:%s\n", config_json_);
                }
                void unpackInputs() {
                    SharedHeader *h = hdr_base();
                    backend_length_ = h->inputSizes[0];
                    config_json_length_ = h->inputSizes[1];
                }


                void execute(QireeManager *qmgr) {
                    _debug2("Before call");
                    auto* h = hdr_base();
                    uint8_t* ptr = (uint8_t*) base();
                    const char *backend = (const char *)ptr;
                    const char *config_json = (const char *)ptr + strlen(backend)+1;
                    if (strcasecmp(config_json, "null") == 0)
                        config_json = NULL;
                    _debug2(" backend:%s\n", backend);
                    _debug2(" json:%s\n", config_json);
                    QMGR_RUN_QIREE(qiree_setup_executor, qmgr->manager_, backend, config_json);
                    _debug2("After call");
                }
        };
        class QireeExecuteCommand : public ICommand {
            public:
                QireeExecuteCommand(ProcessQireeBase *proc) : ICommand(proc) {}
                uint32_t num_shots_;
                CQireeResultRecord *encoded_;

                QireeCmdType commandId() const { return QireeExecute; }

                size_t requiredSize() const {
                    return sizeof(CQireeResultRecord) * (num_shots_+1);
                }

                void packInputs() const {
                    auto* h = hdr_base();
                    h->numInputs = 1;
                    h->inputSizes[0] = (uint32_t) num_shots_;
                }
                void unpackInputs() {
                    SharedHeader *h = hdr_base();
                    num_shots_ = h->inputSizes[0];
                }

                void execute(QireeManager *qmgr) {
                    _debug2("Before call");
                    auto* h = hdr_base();
                    uint8_t* ptr = (uint8_t*) base();
                    CQireeResultRecord *encoded = (CQireeResultRecord*)ptr;
                    uint32_t num_shots = h->inputSizes[0];
                    QMGR_RUN_QIREE(qiree_execute, qmgr->manager_, num_shots);
                    QMGR_RUN_QIREE(qiree_save_result_items, qmgr->manager_, encoded, num_shots);
                    _debug2("After call");
                }

                void unpackOutputs() {
                    auto* h = hdr_base();
                    uint8_t* ptr = (uint8_t*) base();
                    memcpy((uint8_t *)encoded_, ptr, sizeof(CQireeResultRecord)*(num_shots_+1)); 
                }
        };

        //  ProcessQiree (ENCAPSULATED WORKER)
        class ProcessQiree : public ProcessQireeBase {
            private:
                // Both client and server -side command table
                std::unordered_map<int, std::unique_ptr<ICommand>> worker_table_;

            private:
                // Server loop is now INSIDE the class
                void serverLoop() {
                    auto* h = base();

                    while (true) {
                        CmdStatus st = status();

                        if (st == CmdStatus::InputReady) {
                            set_status(CmdStatus::Processing);

                            int cmdId = h->cmdId;
                            auto it = worker_table_.find(cmdId);
                            if (it != worker_table_.end()) {
                                const char *shm_name = h->shm_name();
                                _debug2("Child Received shm_name:%s\n\n", shm_name);
                                it->second->unpackInputs();
                                _debug2("Child unpack shm_name:%p : %s\n", shm_name, shm_name);
                                it->second->create(shm_name);
                                _debug2("Child after create\n");
                                it->second->execute(&qmgr_);
                                _debug2("Now trying to release it\n");
                                it->second->release();
                                _debug2("Ended release it\n");
                            }
                            else
                                std::cerr << "Unknown command " << cmdId << "\n";

                            set_status(CmdStatus::OutputReady);
                        }
                        else if (st == CmdStatus::Exit) {
                            break;
                        }
                    }
                }

            public:
                ProcessQiree() : ProcessQireeBase() {
                    worker_table_[QireeCreate] = std::make_unique<QireeCreateCommand>(this);
                    worker_table_[QireeDestroy] = std::make_unique<QireeDestroyCommand>(this);
                    worker_table_[QireeLoad] = std::make_unique<QireeLoadCommand>(this); // stubs if needed
                    worker_table_[QireeMaxResults] = std::make_unique<QireeMaxResultsCommand>(this);
                    worker_table_[QireeSetupExecutor] = std::make_unique<QireeSetupExecutorCommand>(this);
                    worker_table_[QireeExecute] = std::make_unique<QireeExecuteCommand>(this);
                }
                void Init(int devno, LoaderQIREE *ld) { 
                    devno_ = devno;
                    qmgr_.manager_ = NULL ; qmgr_.ld_ = ld; 
                }
                void Stop() {
                    set_status(Exit);
                }

                void qiree_create() {
                    _debug2("Inside ");
                    invoke(worker_table_[QireeCreate].get());
                    _debug2("Exited");
                }
                void qiree_destroy() {
                    _debug2("Inside ");
                    invoke(worker_table_[QireeDestroy].get());
                    _debug2("Exited");
                }
                void qiree_max_result_items(uint32_t num_shots, size_t *result_items_) {
                    _debug2("Inside ");
                    QireeMaxResultsCommand *cmd = 
                        static_cast<QireeMaxResultsCommand*>(worker_table_[QireeMaxResults].get());
                    cmd->num_shots_ = num_shots;
                    invoke(cmd);
                    *result_items_ = cmd->result_items_;
                    _debug2("Exited");
                }
                void qiree_load(const char *llvm_module, size_t data_len, int *nquant_results, int *nclass_results) {
                    _debug2("Inside ");
                    QireeLoadCommand *cmd = 
                        static_cast<QireeLoadCommand*>(worker_table_[QireeLoad].get());
                    cmd->module_ = llvm_module;
                    cmd->data_len_ = data_len;
                    invoke(cmd);
                    *nquant_results = cmd->nquant_results_;
                    *nclass_results = cmd->nclass_results_;
                    _debug2("Exited");
                }
                void qiree_setup_executor(const char *backend, const char *config_json) {
                    _debug2("Inside ");
                    QireeSetupExecutorCommand *cmd = 
                        static_cast<QireeSetupExecutorCommand*>(worker_table_[QireeSetupExecutor].get());
                    cmd->backend_ = backend;
                    cmd->backend_length_ = strlen(backend)+1;
                    cmd->config_json_ = config_json;
                    cmd->config_json_length_ = strlen(config_json)+1;
                    invoke(cmd);
                    _debug2("Exited");
                }
                void qiree_execute(uint32_t num_shots, CQireeResultRecord *encoded) {
                    _debug2("Inside ");
                    QireeExecuteCommand *cmd = 
                        static_cast<QireeExecuteCommand*>(worker_table_[QireeExecute].get());
                    cmd->num_shots_ = num_shots;
                    cmd->encoded_ = encoded;
                    invoke(cmd);
                    _debug2("Exited");
                }


                void exit() { set_status(CmdStatus::Exit); }

                void invoke(ICommand *cmd) {
                    auto* h = base();

                    set_status(CmdStatus::Idle);

                    if (!started_) {
                        started_ = true;
                        pid_ = fork();

                        if (pid_ == 0) {
                            serverLoop();
                            //printf("Exiting server\n");
                            _exit(0);
                        }
                    }
                    cmd->print("Before Create");
                    cmd->create();
                    cmd->print("After Create");
                    h->set_shm_name(cmd->shm_name());
                    cmd->packInputs();
                    cmd->print("After pack Inputs");
                    h->cmdId = cmd->commandId();
                    cmd->print("Set the InputReady");
                    set_status(CmdStatus::InputReady);
                    cmd->print("After the InputReady");

                    while (status() != CmdStatus::OutputReady)
                        usleep(1000);

                    cmd->print("Unpack outputs");
                    cmd->unpackOutputs();
                    set_status(CmdStatus::Idle);
                    cmd->print("Setting idle");
                    cmd->release();
                    cmd->print("Releasing");
                }
        };

    } /* namespace rt */
} /* namespace iris */


