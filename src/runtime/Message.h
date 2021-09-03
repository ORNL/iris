#ifndef BRISBANE_SRC_RT_MESSAGE_H
#define BRISBANE_SRC_RT_MESSAGE_H

#include <stdint.h>
#include <stdlib.h>

namespace brisbane {
namespace rt {

#define BRISBANE_MSG_SIZE           512

class Message {
public:
  Message(long header = -1);
  ~Message();

  bool WriteHeader(int32_t v);
  bool WritePID(pid_t v);
  bool WriteBool(bool v);
  bool WriteChar(char v);
  bool WriteInt(int32_t v);
  bool WriteUInt(uint32_t v);
  bool WriteLong(int64_t v);
  bool WriteULong(uint64_t v);
  bool WriteFloat(float v);
  bool WriteDouble(double v);
  bool WriteString(const char* v);
  bool WritePtr(void *ptr);
  bool Write(const void* v, size_t size);

  int32_t ReadHeader();
  pid_t ReadPID();
  bool ReadBool();
  int32_t ReadInt();
  uint32_t ReadUInt();
  int64_t ReadLong();
  uint64_t ReadULong();
  float ReadFloat();
  double ReadDouble();
  char* ReadString();
  char* ReadString(size_t len);
  void* ReadPtr();
  void* Read(size_t size);

  char* buf() { return buf_; }
  size_t offset() { return offset_; }
  void Clear();

private:
  char buf_[BRISBANE_MSG_SIZE] __attribute__ ((aligned(0x10)));
  size_t offset_;
};

} /* namespace rt */
} /* namespace brisbane */


#endif /*BRISBANE_SRC_RT_MESSAGE_H */
