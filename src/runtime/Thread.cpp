#include "Thread.h"
#include "Debug.h"

#include <stdexcept>
#include <cerrno>

namespace iris {
namespace rt {

Thread::Thread()
  : thread_(0),
    running_(false),
    sleeping_(false) {

  if (pthread_mutex_init(&mutex_, nullptr) != 0) {
    throw std::runtime_error("pthread_mutex_init failed");
  }

  if (sem_init(&sem_, 0, 0) != 0) {
    pthread_mutex_destroy(&mutex_);
    throw std::runtime_error("sem_init failed");
  }
}

Thread::~Thread() {
  // Try to stop/join any running pthread
  Stop();

  sem_destroy(&sem_);
  pthread_mutex_destroy(&mutex_);
}

void Thread::Start() {
  pthread_mutex_lock(&mutex_);

  if (thread_ != 0) {
    // Already started
    pthread_mutex_unlock(&mutex_);
    return;
  }

  running_.store(true, std::memory_order_relaxed);

  int rc = pthread_create(&thread_, nullptr, &Thread::ThreadFunc, this);
  if (rc != 0) {
    thread_ = 0;
    running_.store(false, std::memory_order_relaxed);
    pthread_mutex_unlock(&mutex_);
    throw std::runtime_error("pthread_create failed");
  }

  pthread_mutex_unlock(&mutex_);
}

void Thread::StartWithOutThread() {
  // No pthread involved – just run in caller thread.
  // Caller is responsible for not calling Start()/Stop() concurrently.
  set_running(true);
  Run();
}

void Thread::Stop() {
  pthread_t local_thread = 0;

  // Take ownership of the thread handle under the mutex
  pthread_mutex_lock(&mutex_);
  if (thread_ != 0) {
    running_.store(false, std::memory_order_relaxed);
    local_thread = thread_;
    thread_ = 0;
  }
  pthread_mutex_unlock(&mutex_);

  if (local_thread != 0) {
    // Wake the thread in case it's sleeping on the semaphore
    Invoke();

    // Do NOT join on self – that would deadlock.
    if (!pthread_equal(local_thread, pthread_self())) {
      pthread_join(local_thread, nullptr);
    }
    // If Stop() is called from the worker thread itself, we just rely on
    // running_ being false and the thread returning from Run().
  }
}

void Thread::Sleep() {
  sleeping_.store(true, std::memory_order_relaxed);

  // Robust sem_wait: retry on EINTR
  while (true) {
    if (sem_wait(&sem_) == 0) break;
    if (errno != EINTR) break;
  }

  sleeping_.store(false, std::memory_order_relaxed);
}

void Thread::Invoke() {
  // Wake up the semaphore (can be called from any thread)
  sem_post(&sem_);
}

void* Thread::ThreadFunc(void* argp) {
  Thread* self = static_cast<Thread*>(argp);
  self->Run();
  return nullptr;
}

} /* namespace rt */
} /* namespace iris */

