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
#include "DeviceQIREE.h"

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

        enum class CmdStatus : uint32_t {
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
            uint32_t inputSizes[16];

            uint32_t numOutputs;
            uint32_t outputSizes[16];

            void *cmd_base;
            CmdStatus loadStatus() {
                return static_cast<CmdStatus>(status.load());
            }
            void storeStatus(CmdStatus st) {
                status.store(static_cast<uint32_t>(st));
            }
        };

        class CommandChannel {
            public:
                CommandChannel(size_t sz) : size_(sz) {
                    addr_ = NULL;
                    if (size_ == 0) return;
                    void* addr = mmap(nullptr, size_,
                            PROT_READ | PROT_WRITE,
                            MAP_SHARED | MAP_ANONYMOUS,
                            -1, 0);
                    if (addr == MAP_FAILED) {
                        perror("mmap failed");
                        throw std::runtime_error("mmap failed");
                    }
                    addr_ = addr;
                    memset(addr_, 0, size_);
                }

                ~CommandChannel() {
                    if (addr_) munmap(addr_, size_);
                }

                void* base() { return addr_; }

            private:
                void* addr_;
                size_t size_;
        };

        class QireeManager {
            public:
                CQiree *manager_;
                LoaderQIREE *ld_;
        };

        class ICommand {
            public:
                ICommand() { mmap_ = NULL; }
                virtual ~ICommand() {
                    if (mmap_ != NULL) delete mmap_;   
                }

                virtual QireeCmdType commandId() const = 0;

                virtual size_t requiredSize() const { return 0; }
                virtual void packInputs(void* base) const { }
                virtual void unpackOutputs(void* base) { }

                virtual void execute(QireeManager *qmgr, void* base) = 0;
                virtual void create() {
                    mmap_ = new CommandChannel(requiredSize());
                }
                virtual void release() {
                    delete mmap_;
                    mmap_ = NULL;
                }
                void *base() { return mmap_->base(); }
                CommandChannel *mmap_;
        };

        class QireeCreateCommand: public ICommand {
            public:
                QireeCmdType commandId() const { return QireeCreate; }
                void execute(QireeManager *qmgr, void* base) {
                    QMGR_RUN_QIREE_OUT(qmgr->manager_, NULL, qiree_create);
                }
        };
        class QireeDestroyCommand: public ICommand {
            public:
                QireeCmdType commandId() const { return QireeDestroy; }
                void execute(QireeManager *qmgr, void* base) {
                    QMGR_RUN_QIREE_NO_RETURN(qiree_destroy, qmgr->manager_);
                }
        };
        class QireeMaxResultsCommand: public ICommand {
            public:
                uint32_t num_shots_;
                uint32_t result_items_;

                QireeCmdType commandId() const { return QireeMaxResults; }
                void packInputs(void* base) const {
                    auto* h = reinterpret_cast<SharedHeader*>(base);
                    h->numInputs = 1;
                    h->inputSizes[0] = num_shots_;
                }

                void execute(QireeManager *qmgr, void* base) {
                    auto* h = reinterpret_cast<SharedHeader*>(base);
                    uint8_t* ptr = (uint8_t*)(h + 1);

                    uint32_t num_shots = h->inputSizes[0];
                    uint32_t result_items = 0;
                    QMGR_RUN_QIREE(qiree_max_result_items, qmgr->manager_, num_shots, (int *)&result_items);

                    h->numOutputs = 1;
                    h->outputSizes[0] = result_items;
                }

                void unpackOutputs(void* base) {
                    auto* h = reinterpret_cast<SharedHeader*>(base);
                    result_items_ = h->outputSizes[0];
                }
        };
        class QireeLoadCommand: public ICommand {
            public:
                uint32_t nquant_results_;
                uint32_t nclass_results_;
                size_t   data_len_;
                const char *module_;

                QireeCmdType commandId() const { return QireeLoad; }

                size_t requiredSize() const {
                    return data_len_;
                }

                void packInputs(void* base) const {
                    auto* h = reinterpret_cast<SharedHeader*>(base);
                    h->numInputs = 1;
                    h->inputSizes[0] = (uint32_t) data_len_;
                    uint8_t* ptr = (uint8_t*) h->cmd_base;
                    memcpy(ptr, module_, data_len_); 
                }

                void execute(QireeManager *qmgr, void* base) {
                    auto* h = reinterpret_cast<SharedHeader*>(base);
                    uint8_t* ptr = (uint8_t*) h->cmd_base;
                    const char *llvm_module = (const char *)ptr;

                    uint32_t data_len = h->inputSizes[0];
                    uint32_t nquant_results = 0, nclass_results = 0;
                    QMGR_RUN_QIREE(qiree_load_module_from_memory, qmgr->manager_, llvm_module, data_len);
                    QMGR_RUN_QIREE(qiree_num_quantum_reg, qmgr->manager_, (int*)&nquant_results);
                    QMGR_RUN_QIREE(qiree_num_classical_reg, qmgr->manager_, (int*)&nclass_results);

                    h->numOutputs = 2;
                    h->outputSizes[0] = nquant_results;
                    h->outputSizes[1] = nclass_results;
                }

                void unpackOutputs(void* base) {
                    auto* h = reinterpret_cast<SharedHeader*>(base);
                    nquant_results_ = h->outputSizes[0];
                    nclass_results_ = h->outputSizes[1];
                }
        };
        class QireeSetupExecutorCommand : public ICommand {
            public:
                const char *backend_;
                const char *config_json_;

                QireeCmdType commandId() const { return QireeSetupExecutor; }

                size_t requiredSize() const {
                    return strlen(backend_) + strlen(config_json_) + 2;
                }

                void packInputs(void* base) const {
                    auto* h = reinterpret_cast<SharedHeader*>(base);
                    uint8_t* ptr = (uint8_t*) h->cmd_base;
                    memcpy(ptr, backend_, strlen(backend_)+1); ptr += strlen(backend_)+1;
                    memcpy(ptr, config_json_, strlen(config_json_)+1); 
                }

                void execute(QireeManager *qmgr, void* base) {
                    auto* h = reinterpret_cast<SharedHeader*>(base);
                    uint8_t* ptr = (uint8_t*) h->cmd_base;
                    const char *backend = (const char *)ptr;
                    const char *config_json = (const char *)ptr + strlen(backend)+1;
                    QMGR_RUN_QIREE(qiree_setup_executor, qmgr->manager_, backend, config_json);
                }
        };
        class QireeExecuteCommand : public ICommand {
            public:
                uint32_t num_shots_;
                CQireeResultRecord *encoded_;

                QireeCmdType commandId() const { return QireeExecute; }

                size_t requiredSize() const {
                    return sizeof(CQireeResultRecord) * (num_shots_+1);
                }

                void packInputs(void* base) const {
                    auto* h = reinterpret_cast<SharedHeader*>(base);
                    h->numInputs = 1;
                    h->inputSizes[0] = (uint32_t) num_shots_;
                }

                void execute(QireeManager *qmgr, void* base) {
                    auto* h = reinterpret_cast<SharedHeader*>(base);
                    uint8_t* ptr = (uint8_t*) h->cmd_base;
                    CQireeResultRecord *encoded = (CQireeResultRecord*)ptr;
                    uint32_t num_shots = h->inputSizes[0];
                    QMGR_RUN_QIREE(qiree_execute, qmgr->manager_, num_shots);
                    QMGR_RUN_QIREE(qiree_save_result_items, qmgr->manager_, encoded, num_shots);
                }

                void unpackOutputs(void* base) {
                    auto* h = reinterpret_cast<SharedHeader*>(base);
                    uint8_t* ptr = (uint8_t*) h->cmd_base;
                    memcpy((uint8_t *)encoded_, ptr, sizeof(CQireeResultRecord)*(num_shots_+1)); 
                }
        };

        //  ProcessQiree (ENCAPSULATED WORKER)
        class ProcessQiree {
            private:
                pid_t pid_;
                bool started_;

                std::unique_ptr<CommandChannel> shared_;
                SharedHeader* header_;
                QireeManager qmgr_;
                // Both client and server -side command table
                std::unordered_map<int, std::unique_ptr<ICommand>> worker_table_;

            private:
                // Server loop is now INSIDE the class
                void serverLoop(void* base) {
                    auto* h = reinterpret_cast<SharedHeader*>(base);

                    while (true) {
                        CmdStatus st = h->loadStatus();

                        if (st == CmdStatus::InputReady) {
                            h->storeStatus(CmdStatus::Processing);

                            int cmdId = h->cmdId;
                            auto it = worker_table_.find(cmdId);
                            if (it != worker_table_.end())
                                it->second->execute(&qmgr_, base);
                            else
                                std::cerr << "Unknown command " << cmdId << "\n";

                            h->storeStatus(CmdStatus::OutputReady);
                        }
                        else if (st == CmdStatus::Exit) {
                            break;
                        }
                    }
                }

            public:
                ProcessQiree() : started_(false) {
                    shared_ = std::make_unique<CommandChannel>(sizeof(SharedHeader));
                    header_ = reinterpret_cast<SharedHeader*>(shared_->base());
                    memset((uint8_t*)header_, 0, sizeof(SharedHeader));
                    qmgr_.manager_ = NULL;
                    qmgr_.ld_ = NULL;
                    worker_table_[QireeCreate] = std::make_unique<QireeCreateCommand>();
                    worker_table_[QireeDestroy] = std::make_unique<QireeDestroyCommand>();
                    worker_table_[QireeLoad] = std::make_unique<QireeLoadCommand>(); // stubs if needed
                    worker_table_[QireeMaxResults] = std::make_unique<QireeMaxResultsCommand>();
                    worker_table_[QireeExecute] = std::make_unique<QireeExecuteCommand>();
                }
                void Init(LoaderQIREE *ld) { 
                    qmgr_.manager_ = NULL ; qmgr_.ld_ = ld; 
                }

                void qiree_create() {
                    invoke(worker_table_[QireeCreate].get());
                }
                void qiree_destroy() {
                    invoke(worker_table_[QireeDestroy].get());
                }
                void qiree_max_result_items(uint32_t num_shots, uint32_t *result_items_) {
                    QireeMaxResultsCommand *cmd = 
                        static_cast<QireeMaxResultsCommand*>(worker_table_[QireeMaxResults].get());
                    cmd->num_shots_ = num_shots;
                    invoke(cmd);
                    *result_items_ = cmd->result_items_;
                }
                void qiree_load(const char *llvm_module, size_t data_len, int *nquant_results, int *nclass_results) {
                    QireeLoadCommand *cmd = 
                        static_cast<QireeLoadCommand*>(worker_table_[QireeLoad].get());
                    cmd->module_ = llvm_module;
                    cmd->data_len_ = data_len;
                    invoke(cmd);
                    *nquant_results = cmd->nquant_results_;
                    *nclass_results = cmd->nclass_results_;
                }
                void qiree_setup_executor(const char *backend, const char *config_json) {
                    QireeSetupExecutorCommand *cmd = 
                        static_cast<QireeSetupExecutorCommand*>(worker_table_[QireeSetupExecutor].get());
                    cmd->backend_ = backend;
                    cmd->config_json_ = config_json;
                    invoke(cmd);
                }
                void qiree_execute(uint32_t num_shots, CQireeResultRecord *encoded) {
                    QireeExecuteCommand *cmd = 
                        static_cast<QireeExecuteCommand*>(worker_table_[QireeExecute].get());
                    cmd->num_shots_ = num_shots;
                    cmd->encoded_ = encoded;
                    invoke(cmd);
                }


                void exit() { header_->storeStatus(CmdStatus::Exit); }

                void invoke(ICommand *cmd) {
                    void* base = shared_->base();
                    auto* h = header_;

                    h->storeStatus(CmdStatus::Idle);

                    if (!started_) {
                        started_ = true;
                        pid_ = fork();

                        if (pid_ == 0) {
                            serverLoop(base);
                            _exit(0);
                        }
                    }
                    cmd->create();
                    h->cmd_base = cmd->base();
                    cmd->packInputs(base);
                    h->cmdId = cmd->commandId();
                    h->storeStatus(CmdStatus::InputReady);

                    while (h->loadStatus() != CmdStatus::OutputReady)
                        usleep(1000);

                    cmd->unpackOutputs(base);
                    h->storeStatus(CmdStatus::Idle);
                    cmd->release();
                }
        };

    } /* namespace rt */
} /* namespace iris */


