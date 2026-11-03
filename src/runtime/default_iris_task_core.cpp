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

#define IRIS_OP_ADD(X, Y)   ((X) + (Y))
#define IRIS_OP_SUB(X, Y)   ((X) - (Y))
#define IRIS_OP_MUL(X, Y)   ((X) * (Y))
#define IRIS_OP_DIV(X, Y)   ((X) / (Y))

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

#define CPP_MATHOP sqrt
#define MATHOP_TYPE sqrtf
#define CORE_MATHOP_FN sqrtf
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

/// ADD/SUB/DIV/MUL/SILU/DIFF-EXP
#define CPP_MATHOP add 
#define MATHOP_TYPE addf
#define CORE_MATHOP_FN IRIS_OP_ADD
#include "default_iris_task_macros_2_input.h"

#define CPP_MATHOP sub
#define MATHOP_TYPE subf
#define CORE_MATHOP_FN IRIS_OP_SUB
#include "default_iris_task_macros_2_input.h"

#define CPP_MATHOP mul
#define MATHOP_TYPE mulf
#define CORE_MATHOP_FN IRIS_OP_MUL
#include "default_iris_task_macros_2_input.h"

#define CPP_MATHOP div
#define MATHOP_TYPE divf
#define CORE_MATHOP_FN IRIS_OP_DIV
#include "default_iris_task_macros_2_input.h"

#define IRIS_OP_SILU(X, Y)   ((X) * (1.0f / (1.0f + expf(-(X)))) *  (Y))
#define CPP_MATHOP silu
#define MATHOP_TYPE siluf
#define CORE_MATHOP_FN IRIS_OP_SILU
#include "default_iris_task_macros_2_input.h"
#undef IRIS_OP_SILU

#define IRIS_OP_DIFF_EXP(X, Y)   (expf((X) - (Y)))
#define CPP_MATHOP diff_exp
#define MATHOP_TYPE diff_expf
#define CORE_MATHOP_FN IRIS_OP_DIFF_EXP
#include "default_iris_task_macros_2_input.h"
#undef IRIS_OP_DIFF_EXP

/// Reduction: sum, prod, max, min, dot
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

#define CPP_REDUCTION dot
#define OP_REDUCTION  sum
#define REDUCTION    dotf
#include "default_iris_task_macros_dot_product.h"

#define CPP_ROPE normal
#define ROPE_TYPE normalf
#include "default_iris_task_macros_ai_rope.h"


#undef DTYPE // End of float based methods


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

#define CPP_MATHOP sqrt
#define MATHOP_TYPE sqrtd
#define CORE_MATHOP_FN sqrt
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

/// ADD/SUB/DIV/MUL/SILU/DIFF-EXP
#define CPP_MATHOP add
#define MATHOP_TYPE addd
#define CORE_MATHOP_FN IRIS_OP_ADD
#include "default_iris_task_macros_2_input.h"

#define CPP_MATHOP sub
#define MATHOP_TYPE subd
#define CORE_MATHOP_FN IRIS_OP_SUB
#include "default_iris_task_macros_2_input.h"

#define CPP_MATHOP mul
#define MATHOP_TYPE muld
#define CORE_MATHOP_FN IRIS_OP_MUL
#include "default_iris_task_macros_2_input.h"

#define CPP_MATHOP div
#define MATHOP_TYPE divd
#define CORE_MATHOP_FN IRIS_OP_DIV
#include "default_iris_task_macros_2_input.h"

#define IRIS_OP_SILU(X, Y)   ((X) * (1.0f / (1.0f + exp(-(X)))) *  (Y))
#define CPP_MATHOP silu
#define MATHOP_TYPE silud
#define CORE_MATHOP_FN IRIS_OP_SILU
#include "default_iris_task_macros_2_input.h"
#undef IRIS_OP_SILU

#define IRIS_OP_DIFF_EXP(X, Y)   (exp((X) - (Y)))
#define CPP_MATHOP diff_exp
#define MATHOP_TYPE diff_expd
#define CORE_MATHOP_FN IRIS_OP_DIFF_EXP
#include "default_iris_task_macros_2_input.h"
#undef IRIS_OP_DIFF_EXP

/// Reduction: sum, prod, max, min, dot
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

#define CPP_REDUCTION dot
#define OP_REDUCTION  sum
#define REDUCTION    dotd
#include "default_iris_task_macros_dot_product.h"

#undef DTYPE

