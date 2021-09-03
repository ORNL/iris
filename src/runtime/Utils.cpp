#include <brisbane/brisbane.h>
#include "Utils.h"
#include "Config.h"
#include "Debug.h"
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>

namespace brisbane {
namespace rt {

void Utils::Logo(bool color) {
  if (color) {
    srand(time(NULL));
    char str[12];
    sprintf(str, "\x1b[%d;3%dm", rand() & 1, rand() % 8 + 1);
    printf("%s", str);
  }
  printf("     ██╗     ██╗██████╗ ██╗███████╗        ██╗  \n");
  printf("     ╚██╗    ██║██╔══██╗██║██╔════╝        ╚██╗ \n");
  printf("█████╗╚██╗   ██║██████╔╝██║███████╗   █████╗╚██╗\n");
  printf("╚════╝██╔╝   ██║██╔══██╗██║╚════██║   ╚════╝██╔╝\n");
  printf("     ██╔╝    ██║██║  ██║██║███████║        ██╔╝ \n");
  printf("     ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝╚══════╝        ╚═╝  \n");
  if (color) {
    printf(RESET);
    fprintf(stderr, RESET);
  }
}

int Utils::ReadFile(char* path, char** string, size_t* len) {
  int fd = open((const char*) path, O_RDONLY);
  if (fd == -1) {
    *len = 0UL;
    return BRISBANE_ERR;
  }
  off_t s = lseek(fd, 0, SEEK_END);
  *string = (char*) malloc(s);
  *len = s;
  lseek(fd, 0, SEEK_SET);
  ssize_t r = read(fd, *string, s);
  if (r != s) {
    _error("read[%zd] vs [%lu]", r, s);
    return BRISBANE_ERR;
  }
  close(fd);
  return BRISBANE_OK;
}

void Utils::Datetime(char* str) {
  time_t t = time(NULL);
  struct tm* tm = localtime(&t);
  strftime(str, 256, "%Y%m%d%H%M%S", tm);
}

} /* namespace rt */
} /* namespace brisbane */
