#ifndef IRIS_SRC_RT_MESSAGE_H
#define IRIS_SRC_RT_MESSAGE_H

#include <stdint.h>
#include <stdlib.h>

namespace iris {
namespace rt {

#define IRIS_MSG_SIZE           256

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
  size_t free_buf_size() { return IRIS_MSG_SIZE - offset_; }
  void Clear();

private:
  char buf_[IRIS_MSG_SIZE+4] __attribute__ ((aligned(0x10)));
  size_t offset_;
};

} /* namespace rt */
} /* namespace iris */


#endif /*IRIS_SRC_RT_MESSAGE_H */
