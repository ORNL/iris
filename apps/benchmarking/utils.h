#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int read_file(const char* path, char** string, size_t* len) {
  int fd = open((const char*) path, O_RDONLY);
  if (fd == -1) {
    *len = 0UL;
    return 0;
  }
  off_t s = lseek(fd, 0, SEEK_END);
  *string = (char*) malloc(s);
  *len = s;
  lseek(fd, 0, SEEK_SET);
  ssize_t r = read(fd, *string, s);
  if (r != s) {
    printf("[%s:%d] read[%zd] vs [%lu]\n", __FILE__, __LINE__, r, s);
    return 0;
  }
  close(fd);
  return 1;
}

