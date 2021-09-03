#include "Message.h"
#include "Debug.h"
#include <string.h>
#include <unistd.h>

namespace brisbane {
namespace rt {

Message::Message(long header) {
  offset_ = 0;
  if (header >= 0) WriteHeader(header);
}

Message::~Message() {
}

bool Message::WriteHeader(int32_t v) {
  return WriteInt(v);
}

bool Message::WritePID(pid_t v) {
  return WriteInt(v);
}

bool Message::WriteBool(bool v) {
  return Write(&v, sizeof(v));
}

bool Message::WriteChar(char v) {
  return Write(&v, sizeof(v));
}

bool Message::WriteInt(int32_t v) {
  return Write(&v, sizeof(v));
}

bool Message::WriteUInt(uint32_t v) {
  return Write(&v, sizeof(v));
}

bool Message::WriteLong(int64_t v) {
  return Write(&v, sizeof(v));
}

bool Message::WriteULong(uint64_t v) {
  return Write(&v, sizeof(v));
}

bool Message::WriteFloat(float v) {
  return Write(&v, sizeof(v));
}

bool Message::WriteDouble(double v) {
  return Write(&v, sizeof(v));
}

bool Message::WriteString(const char* v) {
  return Write(v, strlen(v));
}

bool Message::WritePtr(void *ptr) {
  return Write(reinterpret_cast<void *>(&ptr), sizeof(void *));
}

bool Message::Write(const void* v, size_t size) {
  if (offset_ + size >= BRISBANE_MSG_SIZE) return false;
  memcpy(buf_ + offset_, v, size);
  offset_ += size;
  return true;
}

int32_t Message::ReadHeader() {
  return ReadInt();
}

pid_t Message::ReadPID() {
  return ReadInt();
}

bool Message::ReadBool() {
  return *(bool*) Read(sizeof(bool));
}

int32_t Message::ReadInt() {
  return *(int32_t*) Read(sizeof(int32_t));
}

uint32_t Message::ReadUInt() {
  return *(uint32_t*) Read(sizeof(uint32_t));
}

int64_t Message::ReadLong() {
  return *(int64_t*) Read(sizeof(int64_t));
}

uint64_t Message::ReadULong() {
  return *(uint64_t*) Read(sizeof(uint64_t));
}

float Message::ReadFloat() {
  return *(float*) Read(sizeof(float));
}

double Message::ReadDouble() {
  return *(double*) Read(sizeof(double));
}

char* Message::ReadString() {
  return (char*) Read(strlen(buf_ + offset_));
}

char* Message::ReadString(size_t len) {
  return (char*) Read(len);
}

void* Message::ReadPtr() {
  return *((void **) Read(sizeof(void *)));
}

void* Message::Read(size_t size) {
  void* p = buf_ + offset_;
  offset_ += size;
  return p;
}

void Message::Clear() {
  offset_ = 0;
}

} /* namespace rt */
} /* namespace brisbane */
