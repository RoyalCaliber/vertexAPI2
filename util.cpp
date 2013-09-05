#include "util.h"
#include <cstdarg>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>

int64_t currentTime()
{
  struct timeval tv;  
  gettimeofday(&tv, 0);
  return tv.tv_sec * 1000000l + tv.tv_usec;
}


int parseCmdLineSimple(int argc, char** argv, const char* fmt, ...)
{
  va_list args;
  va_start(args, fmt);
  int iArg = 1;
  for( char f = *fmt; f; ++f )
  {
    switch( f )
    {
      case 's': *(va_arg(args, char**)) = strdup(argv[iArg]); break;
      default:
        printf("parseCmdLineSimple: bad format character '%c'\n", f);
        exit(1);
    }
    ++iArg;
  }
  va_end(args);
}


