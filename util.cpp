#include "util.h"
#include <cstdarg>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <map>

int64_t currentTime()
{
  struct timeval tv;
  gettimeofday(&tv, 0);
  return tv.tv_sec * 1000000l + tv.tv_usec;
}


int parseCmdLineSimple(int argc, char** argv, const char* fmt, ...)
{
  std::map<char, bool> optChars;

  //scan all optional arguments
  for(int i = 1; i < argc; ++i)
  {
    const char* arg = argv[i];
    if( arg[0] == '-' )
    {
      if( arg[1] == 0 || arg[2] != 0 )
      {
        printf("parseCmdLineSimple: invalid option '%s'\n", arg);
        return 0;
      }
      optChars[arg[1]] = true;
    }
  }

  //now go through positional arguments  
  va_list args;
  va_start(args, fmt);
  int iArg = 1;
  bool isOpt = false;
  bool required = true;
  for( const char *f = fmt; *f; ++f )
  {
    if( isOpt )
    {
      isOpt = false;
      *(va_arg(args, bool*)) = (optChars.find(*f) != optChars.end());
    }
    else if( *f == '-' )
      isOpt = true;
    else if( *f == '|' )
    {
      required = false;
    }
    else
    {
      while( iArg < argc && argv[iArg][0] == '-' )
        ++iArg;
      if( iArg == argc )
      {
        if( required )
        {
          printf("parseCmdLineSimple: expected argument of type %c\n", *f);
          return 0;
        }
        else
          return 1;
      }
      
      switch( *f )
      {
        case 's': *(va_arg(args, char**)) = strdup(argv[iArg]); break;
        case 'i': *(va_arg(args, int*))   = atoi(argv[iArg]); break;
        case 'f': *(va_arg(args, float*)) = atof(argv[iArg]); break;
        default:
          printf("parseCmdLineSimple: bad format character '%c'\n", *f);
          return 0;
      }
      ++iArg;
    }
  }
  va_end(args);
  return 1;
}


