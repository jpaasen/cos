#pragma once

#include <stdio.h>
#ifdef VERBOSE
#include <stdarg.h>
#endif

//http://stackoverflow.com/questions/41400/how-to-wrap-a-function-with-variable-length-arguments

void printIt(const char* fmt, ...);
