#ifndef IRIS_SRC_RT_PROFILER_H
#define IRIS_SRC_RT_PROFILER_H

namespace iris {
namespace rt {

class Message;
class Platform;
class Task;

class Profiler {
public:
  Profiler(Platform* platform);
  virtual ~Profiler();

  virtual int CompleteTask(Task* task) = 0;

protected:
  virtual int Main();
  virtual int Exit() = 0;
  virtual const char* FileExtension() = 0;

  int OpenFD();
  int CloseFD();
  int Write(const char* s, int tab = 0);

  const char* policy_str(int policy);

private:
  int Flush();

protected:
  Platform* platform_;

private:
  int fd_;
  char path_[256];
  Message* msg_;
};

} /* namespace rt */
} /* namespace iris */


#endif /*IRIS_SRC_RT_PROFILER_H */
