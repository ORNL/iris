#include "Thread.h"
#include "Debug.h"

#include <stdexcept>
#include <cerrno>

namespace iris {
namespace rt {

Thread::Thread() :
    thread_(0),
    running_(false),
    sleeping_(false)
#ifdef __APPLE__
    , signaled_(false)
#endif
{
  if (pthread_mutex_init(&mutex_, nullptr) != 0) {
    throw std::runtime_error("pthread_mutex_init failed");
  }

#ifdef __APPLE__
  if (pthread_cond_init(&sem_, nullptr) != 0) {
    pthread_mutex_destroy(&mutex_);
    throw std::runtime_error("pthread_cond_init failed");
  }
#else
  if (sem_init(&sem_, 0, 0) != 0) {
    pthread_mutex_destroy(&mutex_);
    throw std::runtime_error("sem_init failed");
  }
#endif
}

Thread::~Thread() {
  // Try to stop/join any running pthread
  Stop();

#ifdef __APPLE__
  pthread_cond_destroy(&sem_);
#else
  sem_destroy(&sem_);
#endif
  pthread_mutex_destroy(&mutex_);
}

void Thread::Start() {
  pthread_mutex_lock(&mutex_);

#ifdef __APPLE__
  // Don't use thread_ != 0 as a sentinel on macOS; use running_ as the guard.
  // But keep change minimal: if already running, return.
  if (running_.load(std::memory_order_relaxed)) {
    pthread_mutex_unlock(&mutex_);
    return;
  }
#else
  if (thread_ != 0) {
    // Already started
    pthread_mutex_unlock(&mutex_);
    return;
  }
#endif

  running_.store(true, std::memory_order_relaxed);

#ifdef __APPLE__
  signaled_ = false;
#endif

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
#ifdef __APPLE__
  set_running(false);
#endif
}

void Thread::Stop() {
  pthread_t local_thread = 0;

  // Take ownership of the thread handle under the mutex
  pthread_mutex_lock(&mutex_);
#ifdef __APPLE__
  bool do_join = false;
  if (running_.load(std::memory_order_relaxed)) {
      running_.store(false, std::memory_order_relaxed);
      local_thread = thread_;
      do_join = true;
      // ensure any waiter can make progress
      signaled_ = true;
      pthread_cond_broadcast(&sem_);
  }
#else
  if (thread_ != 0) {
    running_.store(false, std::memory_order_relaxed);
    local_thread = thread_;
    thread_ = 0;
  }
#endif
  pthread_mutex_unlock(&mutex_);

#ifdef __APPLE__
  if (do_join) Invoke();
  if (do_join) {
    // Do NOT join on self – deadlock
    if (!pthread_equal(local_thread, pthread_self())) {
      pthread_join(local_thread, nullptr);
    }
  }
#else
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
#endif
}

void Thread::Sleep() {
  sleeping_.store(true, std::memory_order_relaxed);
#ifdef __APPLE__
  pthread_mutex_lock(&mutex_);
  while (!signaled_ && running_.load(std::memory_order_relaxed)) {
      pthread_cond_wait(&sem_, &mutex_);
  }
  // consume the "signal" (binary semaphore behavior)
  signaled_ = false;
  pthread_mutex_unlock(&mutex_);
#else
  // Robust sem_wait: retry on EINTR
  while (true) {
    if (sem_wait(&sem_) == 0) break;
    if (errno != EINTR) break;
  }
#endif
  sleeping_.store(false, std::memory_order_relaxed);
}

void Thread::Invoke() {
  // Wake up the semaphore (can be called from any thread)
#ifdef __APPLE__
  // Must lock + set predicate, otherwise signal can be lost
  pthread_mutex_lock(&mutex_);
  signaled_ = true;
  pthread_cond_signal(&sem_);
  pthread_mutex_unlock(&mutex_);
#else
  sem_post(&sem_);
#endif
}

void* Thread::ThreadFunc(void* argp) {
  Thread* self = static_cast<Thread*>(argp);
  self->Run();
  return nullptr;
}

} /* namespace rt */
} /* namespace iris */

