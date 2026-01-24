#ifndef IRIS_SRC_RT_THREAD_H
#define IRIS_SRC_RT_THREAD_H

#include "Config.h"
#include <atomic>
#include <semaphore.h>
#include <pthread.h>

namespace iris {
namespace rt {

class Thread {
public:
  Thread();
  virtual ~Thread();

  // Start a new pthread that executes Run()
  void Start();

  // Run synchronously in the calling thread (no pthread created).
  // Note: Stop() / destructor will NOT join in this mode.
  void StartWithOutThread();

  // Request the thread to stop and join it (if a pthread was created).
  virtual void Stop();

  // Block on the internal semaphore until Invoke() is called.
  virtual void Sleep();

  // Wake up a sleeping thread (sem_post).
  virtual void Invoke();

  // Set the running flag (used by Run() loop)
  void set_running(bool flag = true) {
    running_.store(flag, std::memory_order_relaxed);
  }

  // Query running state (helper for derived Run() implementations)
  bool running() const {
    return running_.load(std::memory_order_relaxed);
  }

  pthread_t thread() const { return thread_; }
  pthread_t self()  const { return pthread_self(); }

protected:
  // User must implement this.
  virtual void Run() = 0;

  // Internal thread entry point
  static void* ThreadFunc(void* argp);

protected:
  pthread_t         thread_;
  std::atomic<bool> running_;
  std::atomic<bool> sleeping_;
  sem_t             sem_;

  // Protects thread_ and lifecycle of the pthread
  pthread_mutex_t   mutex_;
};

} /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_THREAD_H */
