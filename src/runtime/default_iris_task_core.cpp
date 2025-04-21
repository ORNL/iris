/*
 * Author : Narasinga Rao Miniskar
 * Date   : Jun 16 2022
 * Contact for bug : miniskarnr@ornl.gov
 *
 */ 
#include "iris/iris.h"
#include "iris/iris_macros.h"
#include <stdint.h>
#include <stdlib.h>

#define DTYPE float

#define CPP_MATHOP abs
#define MATHOP_TYPE fabsf 
#define CORE_MATHOP_FN fabsf
#include "default_iris_task_macros_1_input.h"

#define CPP_MATHOP cos
#define MATHOP_TYPE cosf
#define CORE_MATHOP_FN cosf
#include "default_iris_task_macros_1_input.h"

#define CPP_MATHOP sin
#define MATHOP_TYPE sinf
#define CORE_MATHOP_FN sinf
#include "default_iris_task_macros_1_input.h"

#define CPP_MATHOP tan
#define MATHOP_TYPE tanf
#define CORE_MATHOP_FN tanf
#include "default_iris_task_macros_1_input.h"

#define CPP_MATHOP asin
#define MATHOP_TYPE asinf
#define CORE_MATHOP_FN asinf
#include "default_iris_task_macros_1_input.h"

#define CPP_MATHOP acos
#define MATHOP_TYPE acosf
#define CORE_MATHOP_FN acosf
#include "default_iris_task_macros_1_input.h"

#define CPP_MATHOP atan
#define MATHOP_TYPE atanf
#define CORE_MATHOP_FN atanf
#include "default_iris_task_macros_1_input.h"

#define CPP_MATHOP exp
#define MATHOP_TYPE expf
#define CORE_MATHOP_FN expf
#include "default_iris_task_macros_1_input.h"

#define CPP_MATHOP log
#define MATHOP_TYPE logf
#define CORE_MATHOP_FN logf
#include "default_iris_task_macros_1_input.h"

#define CPP_MATHOP cbrt
#define MATHOP_TYPE cbrtf
#define CORE_MATHOP_FN cbrtf
#include "default_iris_task_macros_1_input.h"

#define CPP_MATHOP pow
#define MATHOP_TYPE powf
#define CORE_MATHOP_FN powf
#include "default_iris_task_macros_2_input.h"

#define CPP_REDUCTION sum
#define REDUCTION    sumf
#include "default_iris_task_macros_reduction.h"

#define CPP_REDUCTION prod
#define REDUCTION    prodf
#include "default_iris_task_macros_reduction.h"

#define CPP_REDUCTION min
#define REDUCTION    minf
#include "default_iris_task_macros_reduction.h"

#define CPP_REDUCTION max
#define REDUCTION    maxf
#include "default_iris_task_macros_reduction.h"

#undef DTYPE


#define DTYPE double

#define CPP_MATHOP abs
#define MATHOP_TYPE fabsd
#define CORE_MATHOP_FN fabs
#include "default_iris_task_macros_1_input.h"

#define CPP_MATHOP cos
#define MATHOP_TYPE cosd
#define CORE_MATHOP_FN cos
#include "default_iris_task_macros_1_input.h"

#define CPP_MATHOP sin
#define MATHOP_TYPE sind
#define CORE_MATHOP_FN sin
#include "default_iris_task_macros_1_input.h"

#define CPP_MATHOP tan
#define MATHOP_TYPE tand
#define CORE_MATHOP_FN tan
#include "default_iris_task_macros_1_input.h"

#define CPP_MATHOP asin
#define MATHOP_TYPE asind
#define CORE_MATHOP_FN asin
#include "default_iris_task_macros_1_input.h"

#define CPP_MATHOP acos
#define MATHOP_TYPE acosd
#define CORE_MATHOP_FN acos
#include "default_iris_task_macros_1_input.h"

#define CPP_MATHOP atan
#define MATHOP_TYPE atand
#define CORE_MATHOP_FN atan
#include "default_iris_task_macros_1_input.h"

#define CPP_MATHOP exp
#define MATHOP_TYPE expd
#define CORE_MATHOP_FN exp
#include "default_iris_task_macros_1_input.h"

#define CPP_MATHOP log
#define MATHOP_TYPE logd
#define CORE_MATHOP_FN log
#include "default_iris_task_macros_1_input.h"

#define CPP_MATHOP cbrt
#define MATHOP_TYPE cbrtd
#define CORE_MATHOP_FN cbrt
#include "default_iris_task_macros_1_input.h"

#define CPP_MATHOP pow
#define MATHOP_TYPE powd
#define CORE_MATHOP_FN pow
#include "default_iris_task_macros_2_input.h"

#undef DTYPE

#define IRIS_OP_ADD(X, Y)   ((X) + (Y))
#define IRIS_OP_SUB(X, Y)   ((X) - (Y))
#define IRIS_OP_MUL(X, Y)   ((X) * (Y))
#define IRIS_OP_DIV(X, Y)   ((X) / (Y))

#define DTYPE float
#define CPP_MATHOP IRIS_OP_ADD
#define MATHOP_TYPE fadd
#define CORE_MATHOP_FN CPP_MATHOP
#include "default_iris_task_macros_2_input.h"

#define CPP_MATHOP IRIS_OP_SUB
#define MATHOP_TYPE fsub
#define CORE_MATHOP_FN CPP_MATHOP
#include "default_iris_task_macros_2_input.h"

#define CPP_MATHOP IRIS_OP_MUL
#define MATHOP_TYPE fmul
#define CORE_MATHOP_FN CPP_MATHOP
#include "default_iris_task_macros_2_input.h"

#define CPP_MATHOP IRIS_OP_DIV
#define MATHOP_TYPE fdiv
#define CORE_MATHOP_FN CPP_MATHOP
#include "default_iris_task_macros_2_input.h"

#undef DTYPE 

#define DTYPE double
#define CPP_MATHOP IRIS_OP_ADD
#define MATHOP_TYPE dadd
#define CORE_MATHOP_FN CPP_MATHOP
#include "default_iris_task_macros_2_input.h"

#define CPP_MATHOP IRIS_OP_SUB
#define MATHOP_TYPE dsub
#define CORE_MATHOP_FN CPP_MATHOP
#include "default_iris_task_macros_2_input.h"

#define CPP_MATHOP IRIS_OP_MUL
#define MATHOP_TYPE dmul
#define CORE_MATHOP_FN CPP_MATHOP
#include "default_iris_task_macros_2_input.h"

#define CPP_MATHOP IRIS_OP_DIV
#define MATHOP_TYPE ddiv
#define CORE_MATHOP_FN CPP_MATHOP
#include "default_iris_task_macros_2_input.h"

#define CPP_REDUCTION sum
#define REDUCTION    sumd
#include "default_iris_task_macros_reduction.h"

#define CPP_REDUCTION prod
#define REDUCTION    prodd
#include "default_iris_task_macros_reduction.h"

#define CPP_REDUCTION min
#define REDUCTION    mind
#include "default_iris_task_macros_reduction.h"

#define CPP_REDUCTION max
#define REDUCTION    maxd
#include "default_iris_task_macros_reduction.h"

#undef DTYPE

