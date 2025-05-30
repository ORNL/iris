#include <iris/iris.h>
#include <openssl/md5.h>
#include <cstdio>
#include <sys/types.h>
#include <sys/stat.h>
#include <csignal>
#include "aiwc_utils.h"
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <string.h>
#include <iostream>

namespace iris {
namespace rt {
namespace plugin {

bool AIWC_Utils::IsAIWCDevice(char* name,char* vendor){
    if (strstr(name,"Oclgrind Simulator") != NULL && strstr(vendor,"Oclgrind") != NULL){ //strstr(version,"Oclgrind 19.10") != NULL)
        return true;
    }
    return false;
}

char* AIWC_Utils::ComputeDigest(char* src){
  size_t srclen = strlen(src);
  unsigned char digest[MD5_DIGEST_LENGTH];
  //the compute hash of the kernel file is used for the file name; this is to determine if a fresh set of AIWC metrics should be computed
  MD5((const unsigned char*)src, srclen, digest);
  //convert to hex representation
  static char digest_str[MD5_DIGEST_LENGTH*2];
  for (int i = 0,j=0; i < MD5_DIGEST_LENGTH; i++,j+=2){
    sprintf(&digest_str[j],"%02hhx", digest[i]);
  }
  return digest_str;
}

int AIWC_Utils::ReadFile(char* path, char** string, size_t* len) {
  int fd = open((const char*) path, O_RDONLY);
  if (fd == -1) {
    *len = 0UL;
    return IRIS_ERROR;
  }
  off_t s = lseek(fd, 0, SEEK_END);
  *string = (char*) malloc(s);
  *len = s;
  lseek(fd, 0, SEEK_SET);
  ssize_t r = read(fd, *string, s);
  if (r != s) {
      std::cerr << "read["<<r<<"] vs ["<<s<<"]" << std::endl;
    return IRIS_ERROR;
  }
  close(fd);
  return IRIS_SUCCESS;
}

char* AIWC_Utils::ComputeFileDigest(char* path){
  char* src = NULL;
  char digest[MD5_DIGEST_LENGTH];
  size_t srclen = 0;
  if (AIWC_Utils::ReadFile(path, &src, &srclen) == IRIS_ERROR){
    return NULL;
  }
  return(AIWC_Utils::ComputeDigest(src));
}

void AIWC_Utils::SetEnvironment(const char *name, const char *value)
{
      setenv(name, value, 1);
}

const char* AIWC_Utils::MetricLocation(char* digest){
  if (digest == NULL){
    return NULL;
  }
  //ensure the existence of our AIWC metrics kernel storage directory.
  struct stat st;
  const char* metric_dir = "/tmp/iris/";
  if (stat(metric_dir,&st) != 0){
     if(mkdir(metric_dir,  S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH)!=0){
        std::cerr << "Cannot create AIWC metric storage at " << metric_dir << std::endl;
        abort();
     }
  }

  metric_dir = "/tmp/iris/aiwc/";
  if (stat(metric_dir,&st) != 0){
     if(mkdir(metric_dir,  S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH)!=0){
       printf("Cannot create AIWC metric storage at %s",metric_dir);
       abort();
     }
  }
  char* metric_path = new char[100];
  strcpy(metric_path,"/tmp/iris/aiwc/");
  strcat(metric_path,digest);
  return metric_path;
}


bool AIWC_Utils::HaveMetrics(char* digest){
  const char* metric_url = MetricLocation(digest);
  if(metric_url == NULL){
    return false;
  }

  //check for the kernel file exists
  struct stat st;
  if (stat(metric_url,&st) != 0){
    return false;
  }
  //TODO: the file exists but we should also check that it can be loaded and contains valid entries
  return true;
}

/*
bool AIWC_Utils::MetricsForKernelFileExist(char* path) {
  char* src = NULL;
  unsigned char digest[MD5_DIGEST_LENGTH];
  size_t srclen = 0;
  if (Utils::ReadFile(path, &src, &srclen) == BRISBANE_ERR){
    _error("can't read kernel file %s", path);
    return false;
  }
  //the compute hash of the kernel file is used for the file name; this is to determine if a fresh set of AIWC metrics should be computed
  MD5((const unsigned char*)src, srclen, digest);
  //convert to hex representation
  char aiwc_metric_filename[MD5_DIGEST_LENGTH*2];
  for (int i = 0,j=0; i < MD5_DIGEST_LENGTH; i++,j+=2){
    sprintf(&aiwc_metric_filename[j],"%02hhx", digest[i]);
  }
  //ensure the existence of our AIWC metrics kernel storage directory.
  struct stat st;
  const char* metric_dir = "/etc/iris";
  if (stat(metric_dir,&st) != 0){
     if(mkdir(metric_dir,  S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH)!=0){
       _error("Cannot create AIWC metric storage at %s",metric_dir);
     }
     return false;//if we just created the directory of course the kernel metrics aren't in there.
  }
  //check for the kernel file's hash
  char* metric_path = strcat((char*)metric_dir,aiwc_metric_filename);
  if (stat(metric_path,&st) != 0){
    return false;
  }
  std::raise(SIGINT);
  return true;
}*/
} /* namespace plugin */
} /* namespace rt */
} /* namespace iris */


