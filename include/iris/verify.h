#ifndef VERIFY_H
#define VERIFY_H

#ifndef _WIN32
#define C_ASSERT(test) \
    switch(0) {\
      case 0:\
      case test:;\
    }
#endif // _WIN32


#ifdef VERIFY_PRINT_ERROR
#ifndef VERIFY_EPRINTF
#include <stdio.h>
#define VERIFY_EPRINTF(format,args) printf(format,args)
#endif
#else
#define VERIFY_EPRINTF(format,args) (void)0
#endif

#ifdef VERIFY_PRINT_INFO
#ifndef VERIFY_IPRINTF
#include <stdio.h>
#define VERIFY_IPRINTF(args) printf(args)
#endif 
#else
#define VERIFY_IPRINTF(args) (void)0
#endif

#ifndef __V_STR__
	#define __V_STR__(x) #x ":"
#endif //__STR__
#ifndef __V_TOSTR__
	#define __V_TOSTR__(x) __V_STR__(x)
#endif // __TOSTR__
#ifndef __V_FILE_LINE__
	#define __V_FILE_LINE__ __FILE__ ":" __V_TOSTR__(__LINE__)
#endif /*__FILE_LINE__*/

#ifndef VERIFY
	#define VERIFY(val) \
	   do {\
		  VERIFY_IPRINTF(__V_FILE_LINE__":info: calling: " #val "\n");\
		  if(0 == (val)) {\
			 nErr = nErr == 0 ? -1 : nErr;\
			 VERIFY_EPRINTF(__V_FILE_LINE__":error: %d: " #val "\n", nErr);\
			 goto bail;\
		  } else {\
			 VERIFY_IPRINTF(__V_FILE_LINE__":info: passed: " #val "\n");\
		  }\
	   } while(0)
#endif //VERIFY

#endif //VERIFY_H

