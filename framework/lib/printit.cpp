#include "printit.h"

//http://stackoverflow.com/questions/41400/how-to-wrap-a-function-with-variable-length-arguments

void printIt(const char* fmt, ...)
{
#ifdef VERBOSE
   va_list args;
   va_start(args, fmt);
   vprintf(fmt, args);
   va_end(args);
#endif
}
