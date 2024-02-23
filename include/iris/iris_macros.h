#ifndef __IRIS_MACROS_H
#define __IRIS_MACROS_H

#define PARAM_MEM(X)              &(X)
#define IRIS_PTR(X)               X *

#ifndef UNDEF_IRIS_MACROS


#define PARAM_EXPAND(...) __VA_ARGS__ //needed for MSVC compatibility

#define PARAM_JOIN_EXPAND( a , b )     a##b
#define PARAM_JOIN( a , b )            PARAM_JOIN_EXPAND( a , b )

#define PARAM_SECOND_EXPAND( a , b , ... )    b
#define PARAM_SECOND(...)                     PARAM_EXPAND( PARAM_SECOND_EXPAND( __VA_ARGS__ ) )

#define PARAM_HIDDENfloat              unused,0
#define PARAM_HIDDENdouble             unused,2
#define PARAM_CHECK0( value )     PARAM_SECOND( PARAM_JOIN( PARAM_HIDDEN , value ) , 1 , unused )

#define PARAM_DATATYPE0 iris_float
#define PARAM_DATATYPE2 iris_double
#define PARAM_DATATYPE1 0
#define PARAM_DATATYPE( value )   PARAM_JOIN( PARAM_DATATYPE , PARAM_CHECK0( value ) )



#define PARAM_DT_CODE(arg2)  PARAM_DATATYPE(arg2)

#define CONCATENATE(arg1, arg2)   CONCATENATE1(arg1, arg2)
#define CONCATENATE1(arg1, arg2)  CONCATENATE2(arg1, arg2)
#define CONCATENATE2(arg1, arg2)  arg1##arg2

#define FOR_EACH_0(WHAT, X, ...)  
#define FOR_EACH_1(WHAT, X, ...)  WHAT(X) 
#define FOR_EACH_2(WHAT, X, ...)  WHAT(X) FOR_EACH_1(WHAT, __VA_ARGS__) 
#define FOR_EACH_3(WHAT, X, ...)  WHAT(X) FOR_EACH_2(WHAT, __VA_ARGS__) 
#define FOR_EACH_4(WHAT, X, ...)  WHAT(X) FOR_EACH_3(WHAT, __VA_ARGS__) 
#define FOR_EACH_5(WHAT, X, ...)  WHAT(X) FOR_EACH_4(WHAT, __VA_ARGS__) 
#define FOR_EACH_6(WHAT, X, ...)  WHAT(X) FOR_EACH_5(WHAT, __VA_ARGS__) 
#define FOR_EACH_7(WHAT, X, ...)  WHAT(X) FOR_EACH_6(WHAT, __VA_ARGS__) 
#define FOR_EACH_8(WHAT, X, ...)  WHAT(X) FOR_EACH_7(WHAT, __VA_ARGS__) 
#define FOR_EACH_9(WHAT, X, ...)  WHAT(X) FOR_EACH_8(WHAT, __VA_ARGS__) 
#define FOR_EACH_10(WHAT, X, ...) WHAT(X) FOR_EACH_9(WHAT, __VA_ARGS__)
#define FOR_EACH_11(WHAT, X, ...) WHAT(X) FOR_EACH_10(WHAT, __VA_ARGS__)
#define FOR_EACH_12(WHAT, X, ...) WHAT(X) FOR_EACH_11(WHAT, __VA_ARGS__)
#define FOR_EACH_13(WHAT, X, ...) WHAT(X) FOR_EACH_12(WHAT, __VA_ARGS__)
#define FOR_EACH_14(WHAT, X, ...) WHAT(X) FOR_EACH_13(WHAT, __VA_ARGS__)
#define FOR_EACH_15(WHAT, X, ...) WHAT(X) FOR_EACH_14(WHAT, __VA_ARGS__)
#define FOR_EACH_16(WHAT, X, ...) WHAT(X) FOR_EACH_15(WHAT, __VA_ARGS__)
#define FOR_EACH_17(WHAT, X, ...) WHAT(X) FOR_EACH_16(WHAT, __VA_ARGS__)
#define FOR_EACH_18(WHAT, X, ...) WHAT(X) FOR_EACH_17(WHAT, __VA_ARGS__)
#define FOR_EACH_19(WHAT, X, ...) WHAT(X) FOR_EACH_18(WHAT, __VA_ARGS__)
#define FOR_EACH_20(WHAT, X, ...) WHAT(X) FOR_EACH_19(WHAT, __VA_ARGS__)
#define FOR_EACH_21(WHAT, X, ...) WHAT(X) FOR_EACH_20(WHAT, __VA_ARGS__)
#define FOR_EACH_22(WHAT, X, ...) WHAT(X) FOR_EACH_21(WHAT, __VA_ARGS__)
#define FOR_EACH_23(WHAT, X, ...) WHAT(X) FOR_EACH_22(WHAT, __VA_ARGS__)
#define FOR_EACH_24(WHAT, X, ...) WHAT(X) FOR_EACH_23(WHAT, __VA_ARGS__)
#define FOR_EACH_25(WHAT, X, ...) WHAT(X) FOR_EACH_24(WHAT, __VA_ARGS__)

#define FOR_EACH_NARG(...) FOR_EACH_NARG_(__VA_ARGS__, FOR_EACH_RSEQ_N())
#define FOR_EACH_NARG_(...) FOR_EACH_ARG_N(__VA_ARGS__) 
#define FOR_EACH_ARG_N(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, N, ...) N 
#define FOR_EACH_RSEQ_N() 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0

#define FOR_EACH_(N, WHAT, x, ...) CONCATENATE(FOR_EACH_, N)(WHAT, x, __VA_ARGS__)
#define FOR_EACH(WHAT, x, ...)    FOR_EACH_(FOR_EACH_NARG(x, __VA_ARGS__), WHAT, x, __VA_ARGS__)


#define FOR_PEACH_0(PRM, WHAT, X, ...)  
#define FOR_PEACH_1(PRM, WHAT, X, ...)  WHAT(PRM, X) 
#define FOR_PEACH_2(PRM, WHAT, X, ...)  WHAT(PRM, X) FOR_PEACH_1(PRM, WHAT, __VA_ARGS__) 
#define FOR_PEACH_3(PRM, WHAT, X, ...)  WHAT(PRM, X) FOR_PEACH_2(PRM, WHAT, __VA_ARGS__) 
#define FOR_PEACH_4(PRM, WHAT, X, ...)  WHAT(PRM, X) FOR_PEACH_3(PRM, WHAT, __VA_ARGS__) 
#define FOR_PEACH_5(PRM, WHAT, X, ...)  WHAT(PRM, X) FOR_PEACH_4(PRM, WHAT, __VA_ARGS__) 
#define FOR_PEACH_6(PRM, WHAT, X, ...)  WHAT(PRM, X) FOR_PEACH_5(PRM, WHAT, __VA_ARGS__) 
#define FOR_PEACH_7(PRM, WHAT, X, ...)  WHAT(PRM, X) FOR_PEACH_6(PRM, WHAT, __VA_ARGS__) 
#define FOR_PEACH_8(PRM, WHAT, X, ...)  WHAT(PRM, X) FOR_PEACH_7(PRM, WHAT, __VA_ARGS__) 
#define FOR_PEACH_9(PRM, WHAT, X, ...)  WHAT(PRM, X) FOR_PEACH_8(PRM, WHAT, __VA_ARGS__) 
#define FOR_PEACH_10(PRM, WHAT, X, ...) WHAT(PRM, X) FOR_PEACH_9(PRM, WHAT, __VA_ARGS__)
#define FOR_PEACH_11(PRM, WHAT, X, ...) WHAT(PRM, X) FOR_PEACH_10(PRM, WHAT, __VA_ARGS__)
#define FOR_PEACH_12(PRM, WHAT, X, ...) WHAT(PRM, X) FOR_PEACH_11(PRM, WHAT, __VA_ARGS__)
#define FOR_PEACH_13(PRM, WHAT, X, ...) WHAT(PRM, X) FOR_PEACH_12(PRM, WHAT, __VA_ARGS__)
#define FOR_PEACH_14(PRM, WHAT, X, ...) WHAT(PRM, X) FOR_PEACH_13(PRM, WHAT, __VA_ARGS__)
#define FOR_PEACH_15(PRM, WHAT, X, ...) WHAT(PRM, X) FOR_PEACH_14(PRM, WHAT, __VA_ARGS__)
#define FOR_PEACH_16(PRM, WHAT, X, ...) WHAT(PRM, X) FOR_PEACH_15(PRM, WHAT, __VA_ARGS__)
#define FOR_PEACH_17(PRM, WHAT, X, ...) WHAT(PRM, X) FOR_PEACH_16(PRM, WHAT, __VA_ARGS__)
#define FOR_PEACH_18(PRM, WHAT, X, ...) WHAT(PRM, X) FOR_PEACH_17(PRM, WHAT, __VA_ARGS__)
#define FOR_PEACH_19(PRM, WHAT, X, ...) WHAT(PRM, X) FOR_PEACH_18(PRM, WHAT, __VA_ARGS__)
#define FOR_PEACH_20(PRM, WHAT, X, ...) WHAT(PRM, X) FOR_PEACH_19(PRM, WHAT, __VA_ARGS__)
#define FOR_PEACH_21(PRM, WHAT, X, ...) WHAT(PRM, X) FOR_PEACH_20(PRM, WHAT, __VA_ARGS__)
#define FOR_PEACH_22(PRM, WHAT, X, ...) WHAT(PRM, X) FOR_PEACH_21(PRM, WHAT, __VA_ARGS__)
#define FOR_PEACH_23(PRM, WHAT, X, ...) WHAT(PRM, X) FOR_PEACH_22(PRM, WHAT, __VA_ARGS__)
#define FOR_PEACH_24(PRM, WHAT, X, ...) WHAT(PRM, X) FOR_PEACH_23(PRM, WHAT, __VA_ARGS__)
#define FOR_PEACH_25(PRM, WHAT, X, ...) WHAT(PRM, X) FOR_PEACH_24(PRM, WHAT, __VA_ARGS__)

#define FOR_PEACH_NARG(...) FOR_PEACH_NARG_(__VA_ARGS__, FOR_PEACH_RSEQ_N())
#define FOR_PEACH_NARG_(...) FOR_PEACH_ARG_N(__VA_ARGS__) 
#define FOR_PEACH_ARG_N(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, N, ...) N 
#define FOR_PEACH_RSEQ_N() 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0

#define FOR_PEACH_(N, PRM, WHAT, x, ...) CONCATENATE(FOR_PEACH_, N)(PRM, WHAT, x, __VA_ARGS__)
#define FOR_PEACH(PRM, WHAT, x, ...)    FOR_PEACH_(FOR_PEACH_NARG(x, __VA_ARGS__), PRM, WHAT, x, __VA_ARGS__)



#define FOR_EACH_COMMA_0(WHAT, X, ...)  
#define FOR_EACH_COMMA_1(WHAT, X, ...)  WHAT(X) 
#define FOR_EACH_COMMA_2(WHAT, X, ...)  WHAT(X), FOR_EACH_COMMA_1(WHAT, __VA_ARGS__) 
#define FOR_EACH_COMMA_3(WHAT, X, ...)  WHAT(X), FOR_EACH_COMMA_2(WHAT, __VA_ARGS__) 
#define FOR_EACH_COMMA_4(WHAT, X, ...)  WHAT(X), FOR_EACH_COMMA_3(WHAT, __VA_ARGS__) 
#define FOR_EACH_COMMA_5(WHAT, X, ...)  WHAT(X), FOR_EACH_COMMA_4(WHAT, __VA_ARGS__) 
#define FOR_EACH_COMMA_6(WHAT, X, ...)  WHAT(X), FOR_EACH_COMMA_5(WHAT, __VA_ARGS__) 
#define FOR_EACH_COMMA_7(WHAT, X, ...)  WHAT(X), FOR_EACH_COMMA_6(WHAT, __VA_ARGS__) 
#define FOR_EACH_COMMA_8(WHAT, X, ...)  WHAT(X), FOR_EACH_COMMA_7(WHAT, __VA_ARGS__) 
#define FOR_EACH_COMMA_9(WHAT, X, ...)  WHAT(X), FOR_EACH_COMMA_8(WHAT, __VA_ARGS__) 
#define FOR_EACH_COMMA_10(WHAT, X, ...) WHAT(X), FOR_EACH_COMMA_9(WHAT, __VA_ARGS__)
#define FOR_EACH_COMMA_11(WHAT, X, ...) WHAT(X), FOR_EACH_COMMA_10(WHAT, __VA_ARGS__)
#define FOR_EACH_COMMA_12(WHAT, X, ...) WHAT(X), FOR_EACH_COMMA_11(WHAT, __VA_ARGS__)
#define FOR_EACH_COMMA_13(WHAT, X, ...) WHAT(X), FOR_EACH_COMMA_12(WHAT, __VA_ARGS__)
#define FOR_EACH_COMMA_14(WHAT, X, ...) WHAT(X), FOR_EACH_COMMA_13(WHAT, __VA_ARGS__)
#define FOR_EACH_COMMA_15(WHAT, X, ...) WHAT(X), FOR_EACH_COMMA_14(WHAT, __VA_ARGS__)
#define FOR_EACH_COMMA_16(WHAT, X, ...) WHAT(X), FOR_EACH_COMMA_15(WHAT, __VA_ARGS__)
#define FOR_EACH_COMMA_17(WHAT, X, ...) WHAT(X), FOR_EACH_COMMA_16(WHAT, __VA_ARGS__)
#define FOR_EACH_COMMA_18(WHAT, X, ...) WHAT(X), FOR_EACH_COMMA_17(WHAT, __VA_ARGS__)
#define FOR_EACH_COMMA_19(WHAT, X, ...) WHAT(X), FOR_EACH_COMMA_18(WHAT, __VA_ARGS__)
#define FOR_EACH_COMMA_20(WHAT, X, ...) WHAT(X), FOR_EACH_COMMA_19(WHAT, __VA_ARGS__)
#define FOR_EACH_COMMA_21(WHAT, X, ...) WHAT(X), FOR_EACH_COMMA_20(WHAT, __VA_ARGS__)
#define FOR_EACH_COMMA_22(WHAT, X, ...) WHAT(X), FOR_EACH_COMMA_21(WHAT, __VA_ARGS__)
#define FOR_EACH_COMMA_23(WHAT, X, ...) WHAT(X), FOR_EACH_COMMA_22(WHAT, __VA_ARGS__)
#define FOR_EACH_COMMA_24(WHAT, X, ...) WHAT(X), FOR_EACH_COMMA_23(WHAT, __VA_ARGS__)
#define FOR_EACH_COMMA_25(WHAT, X, ...) WHAT(X), FOR_EACH_COMMA_24(WHAT, __VA_ARGS__)

#define FOR_EACH_COMMA_NARG(...) FOR_EACH_COMMA_NARG_(__VA_ARGS__, FOR_EACH_COMMA_RSEQ_N())
#define FOR_EACH_COMMA_NARG_(...) FOR_EACH_COMMA_ARG_N(__VA_ARGS__) 
#define FOR_EACH_COMMA_ARG_N(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, N, ...) N 
#define FOR_EACH_COMMA_RSEQ_N() 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0

#define FOR_EACH_COMMA_(N, WHAT, x, ...) CONCATENATE(FOR_EACH_COMMA_, N)(WHAT, x, __VA_ARGS__)
#define FOR_EACH_COMMA(WHAT, x, ...)    FOR_EACH_COMMA_(FOR_EACH_COMMA_NARG(x, __VA_ARGS__), WHAT, x, __VA_ARGS__)


#define _ARG16(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, ...) _15
#define HAS_COMMA(...) _ARG16(__VA_ARGS__, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0)
#define _TRIGGER_PARENTHESIS_(...) ,
#define ISEMPTY(...)                                                    \
    _ISEMPTY(                                                               \
                      /* test if there is just one argument, eventually an empty    \
                       *              one */                                                     \
                      HAS_COMMA(__VA_ARGS__),                                       \
                      /* test if _TRIGGER_PARENTHESIS_ together with the argument   \
                       *              adds a comma */                                            \
                      HAS_COMMA(_TRIGGER_PARENTHESIS_ __VA_ARGS__),                 \
                      /* test if the argument together with a parenthesis           \
                       *              adds a comma */                                            \
                      HAS_COMMA(__VA_ARGS__ (/*empty*/)),                           \
                      /* test if placing it between _TRIGGER_PARENTHESIS_ and the   \
                       *              parenthesis adds a comma */                                \
                      HAS_COMMA(_TRIGGER_PARENTHESIS_ __VA_ARGS__ (/*empty*/))      \
                      )
#define PASTE5(_0, _1, _2, _3, _4) _0 ## _1 ## _2 ## _3 ## _4
#define _ISEMPTY(_0, _1, _2, _3) HAS_COMMA(PASTE5(_IS_EMPTY_CASE_, _0, _1, _2, _3))
#define _IS_EMPTY_CASE_0001 ,

#define PCONCATENATE(arg1, arg2)   PCONCATENATE1(arg1, arg2)
#define PCONCATENATE1(arg1, arg2)  PCONCATENATE2(arg1, arg2)
#define PCONCATENATE2(arg1, arg2)  arg1##arg2

#define CCONCATENATE(arg1, arg2)   CCONCATENATE1(arg1, arg2)
#define CCONCATENATE1(arg1, arg2)  CCONCATENATE2(arg1, arg2)
#define CCONCATENATE2(arg1, arg2)  arg1##arg2

#define PMI_ARG_1(X)                        iris_ftf
#define PMI_ARG_0(X)                        X 
#define PMI_CORE(...)                       PCONCATENATE(PMI_ARG_, ISEMPTY(__VA_ARGS__))(__VA_ARGS__)

#define P_PARAM(NAME, DATA_TYPE ...)                                                       &NAME,
#define P_PARAM_CONST(NAME, DATA_TYPE ...)                                                 &IRIS_VAR(NAME),
#define P_VEC_PARAM(NAME, DATA_TYPE ...)                                                   &__iris_ ## NAME, 
#define P_IN_TASK(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)                  &__iris_ ## IRIS_NAME,
#define P_IN_TASK_DEV_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)       &__iris_ ## IRIS_NAME,
#define P_IN_TASK_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)           &__iris_ ## IRIS_NAME,
#define P_OUT_TASK(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)                 &__iris_ ## IRIS_NAME,
#define P_OUT_TASK_DEV_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)      &__iris_ ## IRIS_NAME,
#define P_OUT_TASK_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)          &__iris_ ## IRIS_NAME,
#define P_IN_OUT_TASK(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)              &__iris_ ## IRIS_NAME,
#define P_IN_OUT_TASK_DEV_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)   &__iris_ ## IRIS_NAME,
#define P_IN_OUT_TASK_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)       &__iris_ ## IRIS_NAME,
#define P_DEPENDENCY(...)   
#define P_REPLACE_PARAMS(NAME)  CONCATENATE(P_, NAME)
#define IRIS_TASK_PARAMS(...)    FOR_EACH(P_REPLACE_PARAMS, __VA_ARGS__)

#define ITC_PARAM(NAME, DATA_TYPE ...)                                          
#define ITC_PARAM_CONST(NAME, DATA_TYPE, VALUE ...)                             DATA_TYPE IRIS_VAR(NAME); IRIS_VAR(NAME) = VALUE;
#define ITC_VEC_PARAM(NAME, DATA_TYPE ...)                                      
#define ITC_IN_TASK(IRIS_NAME, DATA_TYPE ...)     
#define ITC_IN_TASK_DEV_OFFSET(IRIS_NAME, DATA_TYPE ...)     
#define ITC_IN_TASK_OFFSET(IRIS_NAME, DATA_TYPE ...)     
#define ITC_OUT_TASK(IRIS_NAME, DATA_TYPE ...)    
#define ITC_OUT_TASK_DEV_OFFSET(IRIS_NAME, DATA_TYPE ...)    
#define ITC_OUT_TASK_OFFSET(IRIS_NAME, DATA_TYPE ...)    
#define ITC_IN_OUT_TASK(IRIS_NAME, DATA_TYPE ...)     
#define ITC_IN_OUT_TASK_DEV_OFFSET(IRIS_NAME, DATA_TYPE ...)     
#define ITC_IN_OUT_TASK_OFFSET(IRIS_NAME, DATA_TYPE ...)     
#define ITC_DEPENDENCY(...)   
#define ITC_REPLACE_PARAMS(NAME) CONCATENATE(ITC_, NAME)
#define IRIS_TASK_CONSTS(...)     \
                FOR_EACH(ITC_REPLACE_PARAMS, __VA_ARGS__)

#define PI_PARAM(NAME, DATA_TYPE, ...)                                                       (sizeof(NAME) | (PARAM_DT_CODE(DATA_TYPE))),
#define PI_PARAM_CONST(NAME, DATA_TYPE, ...)                                                 (sizeof(IRIS_VAR(NAME)) | (PARAM_DT_CODE(DATA_TYPE))),
#define PI_VEC_PARAM(NAME, DATA_TYPE ...)                                                   iris_r,
#define PI_IN_TASK(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)                  iris_r,
#define PI_IN_TASK_DEV_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)       iris_r,
#define PI_IN_TASK_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)           iris_r,
#define PI_OUT_TASK(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)                 iris_w,
#define PI_OUT_TASK_DEV_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)      iris_w,
#define PI_OUT_TASK_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)          iris_w,
#define PI_IN_OUT_TASK(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)              iris_rw,
#define PI_IN_OUT_TASK_DEV_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)   iris_rw,
#define PI_IN_OUT_TASK_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)       iris_rw,
#define PI_DEPENDENCY(...)   
#define PI_REPLACE_PARAMS(NAME)      CONCATENATE(PI_, NAME)
#define IRIS_TASK_PARAMS_INFO(...)    \
        FOR_EACH(PI_REPLACE_PARAMS, __VA_ARGS__)

#define PMI_PARAM(NAME, DATA_TYPE, ...)                                        PMI_CORE(__VA_ARGS__),
#define PMI_PARAM_CONST(NAME, DATA_TYPE, VALUE, ...)                           PMI_CORE(__VA_ARGS__),
#define PMI_VEC_PARAM(NAME, DATA_TYPE, ...)                                    PMI_CORE(__VA_ARGS__),
#define PMI_IN_TASK(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE, ...)   PMI_CORE(__VA_ARGS__),
#define PMI_IN_TASK_DEV_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE, OFFSET, ...)                             PMI_CORE(__VA_ARGS__),
#define PMI_IN_TASK_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE, OFFSET, HOST_SIZE, DEV_SIZE, DIM, ...)       PMI_CORE(__VA_ARGS__),
#define PMI_OUT_TASK(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE, ...)                                               PMI_CORE(__VA_ARGS__),
#define PMI_OUT_TASK_DEV_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE, OFFSET, HOST_SIZE, DEV_SIZE, DIM, ...)  PMI_CORE(__VA_ARGS__),
#define PMI_OUT_TASK_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE, OFFSET, HOST_SIZE, DEV_SIZE, DIM, ...)      PMI_CORE(__VA_ARGS__),
#define PMI_IN_OUT_TASK(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE, ...)                                            PMI_CORE(__VA_ARGS__),
#define PMI_IN_OUT_TASK_DEV_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE, OFFSET, HOST_SIZE, DEV_SIZE, DIM, ...)  PMI_CORE(__VA_ARGS__),
#define PMI_IN_OUT_TASK_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE, OFFSET, HOST_SIZE, DEV_SIZE, DIM, ...)   PMI_CORE(__VA_ARGS__),
#define PMI_DEPENDENCY(...)   
#define PMI_REPLACE_PARAMS(NAME)      CONCATENATE(PMI_, NAME)
#define IRIS_TASK_PARAMS_MAP(...)    \
        FOR_EACH(PMI_REPLACE_PARAMS, __VA_ARGS__)

#define MEM_DECL_PARAM(IRIS_NAME, ...)               
#define MEM_DECL_PARAM_CONST(IRIS_NAME, ...)               
#define MEM_DECL_VEC_PARAM(IRIS_NAME, ...)              iris_mem  __iris_ ## IRIS_NAME; 
#define MEM_DECL_IN_TASK(IRIS_NAME, ...)                iris_mem  __iris_ ## IRIS_NAME; 
#define MEM_DECL_IN_TASK_DEV_OFFSET(IRIS_NAME, ...)     iris_mem  __iris_ ## IRIS_NAME; 
#define MEM_DECL_IN_TASK_OFFSET(IRIS_NAME, ...)         iris_mem  __iris_ ## IRIS_NAME; 
#define MEM_DECL_OUT_TASK(IRIS_NAME, ...)               iris_mem  __iris_ ## IRIS_NAME; 
#define MEM_DECL_OUT_TASK_DEV_OFFSET(IRIS_NAME, ...)    iris_mem  __iris_ ## IRIS_NAME; 
#define MEM_DECL_OUT_TASK_OFFSET(IRIS_NAME, ...)        iris_mem  __iris_ ## IRIS_NAME; 
#define MEM_DECL_IN_OUT_TASK(IRIS_NAME, ...)            iris_mem  __iris_ ## IRIS_NAME; 
#define MEM_DECL_IN_OUT_TASK_DEV_OFFSET(IRIS_NAME, ...) iris_mem  __iris_ ## IRIS_NAME; 
#define MEM_DECL_IN_OUT_TASK_OFFSET(IRIS_NAME, ...)     iris_mem  __iris_ ## IRIS_NAME; 
#define MEM_DEPENDENCY(...)   
#define MEM_DECL_REPLACE_PARAMS(NAME)      CONCATENATE(MEM_DECL_, NAME)
#define IRIS_MEM_DECLARE(...)    \
        FOR_EACH(MEM_DECL_REPLACE_PARAMS, __VA_ARGS__)

#define MEM_CREATE_PARAM(NAME, DATA_TYPE ...)               
#define MEM_CREATE_PARAM_CONST(NAME, DATA_TYPE ...)               
#define MEM_CREATE_VEC_PARAM(NAME, DATA_TYPE ...)                                        iris_mem_create(sizeof(NAME), &__iris_ ## NAME);
#define MEM_CREATE_IN_TASK(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)       iris_mem_create((SIZE), &__iris_ ## IRIS_NAME);
#define MEM_CREATE_IN_TASK_DEV_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)       iris_mem_create((SIZE), &__iris_ ## IRIS_NAME);
#define MEM_CREATE_IN_TASK_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE, OFFSET, HOST_SIZE, DEV_SIZE, DIM ...)       iris_mem_create((SIZE), &__iris_ ## IRIS_NAME);
#define MEM_CREATE_OUT_TASK(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)       iris_mem_create((SIZE), &__iris_ ## IRIS_NAME);
#define MEM_CREATE_OUT_TASK_DEV_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE, OFFSET, HOST_SIZE, DEV_SIZE, DIM ...)      iris_mem_create((SIZE), &__iris_ ## IRIS_NAME);
#define MEM_CREATE_OUT_TASK_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE, OFFSET, HOST_SIZE, DEV_SIZE, DIM ...)      iris_mem_create((SIZE), &__iris_ ## IRIS_NAME);
#define MEM_CREATE_IN_OUT_TASK(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)   iris_mem_create((SIZE), &__iris_ ## IRIS_NAME);
#define MEM_CREATE_IN_OUT_TASK_DEV_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE, OFFSET, HOST_SIZE, DEV_SIZE, DIM ...)   iris_mem_create((SIZE), &__iris_ ## IRIS_NAME);
#define MEM_CREATE_IN_OUT_TASK_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE, OFFSET, HOST_SIZE, DEV_SIZE, DIM ...)   iris_mem_create((SIZE), &__iris_ ## IRIS_NAME);
#define MEM_CREATE_DEPENDENCY(...)   
#define MEM_CREATE_REPLACE_PARAMS(NAME)      CONCATENATE(MEM_CREATE_, NAME)
#define IRIS_MEM_CREATE_INTERNAL(...)    \
        FOR_EACH(MEM_CREATE_REPLACE_PARAMS, __VA_ARGS__)

#define MEM_REL_PARAM(NAME, DATA_TYPE ...)               
#define MEM_REL_PARAM_CONST(NAME, DATA_TYPE ...)               
#define MEM_REL_VEC_PARAM(NAME, DATA_TYPE ...)                                                   iris_mem_release(__iris_ ## NAME);
#define MEM_REL_IN_TASK(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)                  iris_mem_release(__iris_ ## IRIS_NAME);
#define MEM_REL_IN_TASK_DEV_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)       iris_mem_release(__iris_ ## IRIS_NAME);
#define MEM_REL_IN_TASK_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)           iris_mem_release(__iris_ ## IRIS_NAME);
#define MEM_REL_OUT_TASK(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)                 iris_mem_release(__iris_ ## IRIS_NAME);
#define MEM_REL_OUT_TASK_DEV_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)      iris_mem_release(__iris_ ## IRIS_NAME);
#define MEM_REL_OUT_TASK_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)          iris_mem_release(__iris_ ## IRIS_NAME);
#define MEM_REL_IN_OUT_TASK(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)              iris_mem_release(__iris_ ## IRIS_NAME);
#define MEM_REL_IN_OUT_TASK_DEV_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)   iris_mem_release(__iris_ ## IRIS_NAME);
#define MEM_REL_IN_OUT_TASK_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)       iris_mem_release(__iris_ ## IRIS_NAME);
#define MEM_REL_DEPENDENCY(...)   
#define MEM_REL_REPLACE_PARAMS(NAME)      CONCATENATE(MEM_REL_, NAME)
#define IRIS_MEM_RELEASE(...)    FOR_EACH(MEM_REL_REPLACE_PARAMS, __VA_ARGS__)

#define HDPTR_VEC_PARAM(NAME, DATA_TYPE ...)                                                   &NAME
#define HDPTR_IN_TASK(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)                  VARIABLE
#define HDPTR_IN_TASK_DEV_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)       VARIABLE
#define HDPTR_IN_TASK_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)           VARIABLE
#define HDPTR_OUT_TASK(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)                 VARIABLE
#define HDPTR_OUT_TASK_DEV_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)      VARIABLE
#define HDPTR_OUT_TASK_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)          VARIABLE
#define HDPTR_IN_OUT_TASK(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)              VARIABLE
#define HDPTR_IN_OUT_TASK_DEV_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)   VARIABLE
#define HDPTR_IN_OUT_TASK_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)       VARIABLE
#define HDPTR(NAME)   CONCATENATE(HDPTR_, NAME)                                               
#define HDSIZE_VEC_PARAM(NAME, DATA_TYPE ...)                                                       sizeof(NAME)
#define HDSIZE_IN_TASK(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)                      (SIZE)
#define HDSIZE_IN_TASK_DEV_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE, OFFSET ...)                           __iris_ ## IRIS_NAME, OFFSET, SIZE, VARIABLE 
#define HDSIZE_IN_TASK_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE, OFFSET, HOST_SIZE, DEV_SIZE, DIM ...)     __iris_ ## IRIS_NAME, OFFSET, HOST_SIZE, DEV_SIZE, sizeof(ELEMENT_TYPE), DIM, VARIABLE
#define HDSIZE_OUT_TASK(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)                     (SIZE)
#define HDSIZE_OUT_TASK_DEV_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE, OFFSET ...)                          __iris_ ## IRIS_NAME, OFFSET, SIZE, VARIABLE
#define HDSIZE_OUT_TASK_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE, OFFSET, HOST_SIZE, DEV_SIZE, DIM ...)    __iris_ ## IRIS_NAME, OFFSET, HOST_SIZE, DEV_SIZE, sizeof(ELEMENT_TYPE), DIM, VARIABLE
#define HDSIZE_IN_OUT_TASK(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)                  (SIZE)
#define HDSIZE_IN_OUT_TASK_DEV_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE, OFFSET ...)                       __iris_ ## IRIS_NAME, OFFSET, SIZE, VARIABLE
#define HDSIZE_IN_OUT_TASK_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE, OFFSET, HOST_SIZE, DEV_SIZE, DIM ...) __iris_ ## IRIS_NAME, OFFSET, HOST_SIZE, DEV_SIZE, sizeof(ELEMENT_TYPE), DIM, VARIABLE
#define HDSIZE(NAME)   CONCATENATE(HDSIZE_, NAME)
#define HD_DUMMY(NAME, SIZE, Z)   
#define HDP_VEC_PARAM(NAME, DATA_TYPE ...)                                                   NAME
#define HDP_IN_TASK(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)                  IRIS_NAME
#define HDP_IN_TASK_DEV_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)       IRIS_NAME
#define HDP_IN_TASK_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)           IRIS_NAME
#define HDP_OUT_TASK(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)                 IRIS_NAME
#define HDP_OUT_TASK_DEV_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)      IRIS_NAME
#define HDP_OUT_TASK_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)          IRIS_NAME
#define HDP_IN_OUT_TASK(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)              IRIS_NAME
#define HDP_IN_OUT_TASK_DEV_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)   IRIS_NAME
#define HDP_IN_OUT_TASK_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)       IRIS_NAME
#define HDP_VAR(NAME)                CONCATENATE(HDP_, NAME)
#define IRIS_VAR(NAME)               __iris_ ## NAME
#define H2D_TASK_CORE(PRM, NAME, SIZE, XP)      iris_task_h2d(PRM, IRIS_VAR(NAME), 0, (SIZE), XP);
#define H2D_TASK(PRM, NAME, SIZE)               H2D_TASK_CORE(PRM, HDP_VAR(NAME), SIZE, HDPTR(NAME))
#define H2D_TASK_OFFSET(PRM, NAME, PARAMS)               iris_task_h2d_offsets(PRM, PARAMS);
#define H2D_TASK_DEV_OFFSET(PRM, NAME, PARAMS)               iris_task_h2d(PRM, PARAMS);

#define H2D_PARAM(NAME, DATA_TYPE ...)                                                       HD_DUMMY
#define H2D_PARAM_CONST(NAME, DATA_TYPE ...)                                                 HD_DUMMY
#define H2D_VEC_PARAM(NAME, DATA_TYPE ...)                                                   H2D_TASK
#define H2D_IN_TASK(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)                  H2D_TASK
#define H2D_IN_TASK_DEV_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)       H2D_TASK_DEV_OFFSET
#define H2D_IN_TASK_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)           H2D_TASK_OFFSET
#define H2D_OUT_TASK(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)                 HD_DUMMY
#define H2D_OUT_TASK_DEV_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)      HD_DUMMY
#define H2D_OUT_TASK_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)          HD_DUMMY
#define H2D_IN_OUT_TASK(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)              H2D_TASK
#define H2D_IN_OUT_TASK_DEV_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)   H2D_TASK_DEV_OFFSET
#define H2D_IN_OUT_TASK_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)       H2D_TASK_OFFSET
#define H2D_DEPENDENCY(...)                                                                  HD_DUMMY 
#define H2D_REPLACE_PARAMS(PRM, NAME) CONCATENATE(H2D_, NAME)(PRM, NAME, HDSIZE(NAME))
#define IRIS_ADD_H2D(TASK_VAR, ...)     \
    FOR_PEACH(TASK_VAR, H2D_REPLACE_PARAMS, __VA_ARGS__)

#define D2H_TASK_CORE(PRM, NAME, SIZE, XP)                              iris_task_d2h(PRM, IRIS_VAR(NAME), 0, (SIZE), XP);
#define D2H_TASK(PRM, NAME, SIZE)           D2H_TASK_CORE(PRM, HDP_VAR(NAME), SIZE, HDPTR(NAME))
#define D2H_TASK_OFFSET(PRM, NAME, PARAMS)               iris_task_d2h_offsets(PRM, PARAMS);
#define D2H_TASK_DEV_OFFSET(PRM, NAME, PARAMS)               iris_task_d2h(PRM, PARAMS);

#define D2H_PARAM(NAME, DATA_TYPE ...)                                                      HD_DUMMY
#define D2H_PARAM_CONST(NAME, DATA_TYPE ...)                                                HD_DUMMY
#define D2H_VEC_PARAM(NAME, DATA_TYPE ...)                                                  HD_DUMMY
#define D2H_IN_TASK(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)                 HD_DUMMY
#define D2H_IN_TASK_DEV_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)      HD_DUMMY
#define D2H_IN_TASK_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)          HD_DUMMY
#define D2H_OUT_TASK(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)                D2H_TASK
#define D2H_OUT_TASK_DEV_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)     D2H_TASK_DEV_OFFSET
#define D2H_OUT_TASK_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)         D2H_TASK_OFFSET
#define D2H_IN_OUT_TASK(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)             D2H_TASK
#define D2H_IN_OUT_TASK_DEV_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)  D2H_TASK_DEV_OFFSET
#define D2H_IN_OUT_TASK_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)      D2H_TASK_OFFSET
#define D2H_DEPENDENCY(...)                                                                 HD_DUMMY
#define D2H_REPLACE_PARAMS(PRM, NAME) CONCATENATE(D2H_, NAME)(PRM, NAME, HDSIZE(NAME))
#define IRIS_ADD_D2H(TASK_VAR, ...)        FOR_PEACH(TASK_VAR, D2H_REPLACE_PARAMS, __VA_ARGS__)

#define TDEP_DECL_DUMMY(...)
#define TDEP_DECL_EXTRACT(...)                            iris_task ___task = __VA_ARGS__;
#define TDEP_DECL_PARAM(NAME, DATA_TYPE ...)              TDEP_DECL_DUMMY        
#define TDEP_DECL_PARAM_CONST(NAME, DATA_TYPE ...)        TDEP_DECL_DUMMY
#define TDEP_DECL_VEC_PARAM(NAME, DATA_TYPE ...)          TDEP_DECL_DUMMY
#define TDEP_DECL_IN_TASK(IRIS_NAME,  ...)                TDEP_DECL_DUMMY
#define TDEP_DECL_IN_TASK_DEV_OFFSET(IRIS_NAME,  ...)     TDEP_DECL_DUMMY
#define TDEP_DECL_IN_TASK_OFFSET(IRIS_NAME,  ...)         TDEP_DECL_DUMMY
#define TDEP_DECL_OUT_TASK(IRIS_NAME,  ...)               TDEP_DECL_DUMMY
#define TDEP_DECL_OUT_TASK_DEV_OFFSET(IRIS_NAME,  ...)    TDEP_DECL_DUMMY
#define TDEP_DECL_OUT_TASK_OFFSET(IRIS_NAME,  ...)        TDEP_DECL_DUMMY
#define TDEP_DECL_IN_OUT_TASK(IRIS_NAME,  ...)            TDEP_DECL_DUMMY
#define TDEP_DECL_IN_OUT_TASK_DEV_OFFSET(IRIS_NAME,  ...) TDEP_DECL_DUMMY
#define TDEP_DECL_IN_OUT_TASK_OFFSET(IRIS_NAME,  ...)     TDEP_DECL_DUMMY
#define TDEP_DECL_DEPENDENCY(...)                         TDEP_DECL_EXTRACT 
#define TDEP_DECL_REPLACE_PARAMS(TASK_VAR, NAME)          CONCATENATE(TDEP_DECL_, NAME)(TASK_VAR)
#define IRIS_TASK_DEPENDENCY_DECLARE(TASK_VAR, ...)  FOR_PEACH(TASK_VAR, TDEP_DECL_REPLACE_PARAMS, __VA_ARGS__)

#define TDEP_PARAM(NAME, DATA_TYPE ...)           
#define TDEP_PARAM_CONST(NAME, DATA_TYPE ...)     
#define TDEP_VEC_PARAM(NAME, DATA_TYPE ...)       
#define TDEP_IN_TASK(IRIS_NAME,  ...)             
#define TDEP_IN_TASK_DEV_OFFSET(IRIS_NAME,  ...)             
#define TDEP_IN_TASK_OFFSET(IRIS_NAME,  ...)             
#define TDEP_OUT_TASK(IRIS_NAME,  ...)            
#define TDEP_OUT_TASK_OFFSET(IRIS_NAME,  ...)            
#define TDEP_IN_OUT_TASK(IRIS_NAME,  ...)         
#define TDEP_IN_OUT_TASK_OFFSET(IRIS_NAME,  ...)         
#define TDEP_DEPENDENCY(...)                      IRIS_TASK_DEPENDENCY(___task, __VA_ARGS__) 
#define TDEP_REPLACE_PARAMS(NAME)       CONCATENATE(TDEP_, NAME)
#define IRIS_TASK_DEPENDENCY_CORE(TASK_VAR, ...)  FOR_EACH(TDEP_REPLACE_PARAMS, __VA_ARGS__)

#define IR_ARG_0(X)           (size_t) X,
#define IR_ARG_1(X)           X
#define IR_CORE(...)          CCONCATENATE(IR_ARG_, ISEMPTY(__VA_ARGS__))(__VA_ARGS__)

#define IR_OFFSET_CORE(X)     IR_CORE(X)
#define IR_OFFSET(...)        size_t __st_offset[] = { FOR_EACH(IR_OFFSET_CORE, __VA_ARGS__) }
#define IR_NULL_OFFSET        size_t *__st_offset = (size_t *)0
#define IRIS_TASK_OFFSET(X)   PCONCATENATE(IR_, X)

#define IR_GWS_CORE(X)        IR_CORE(X)
#define IR_GWS(...)           size_t __st_gws[] = { FOR_EACH(IR_GWS_CORE, __VA_ARGS__) }
#define IR_NULL_GWS           size_t *__st_gws = (size_t *)0
#define IRIS_TASK_GWS(X)      PCONCATENATE(IR_, X)

#define IR_LWS_CORE(X)        IR_CORE(X)
#define IR_LWS(...)           size_t __st_lws[] = { FOR_EACH(IR_LWS_CORE, __VA_ARGS__) }
#define IR_NULL_LWS           size_t *__st_lws = (size_t *)0
#define IRIS_TASK_LWS(X)      PCONCATENATE(IR_, X)

#define IRIS_TASK_CREATE_PARAMS(TASK_VAR)  iris_task TASK_VAR; iris_task_create(&TASK_VAR);
#define IRIS_TASK_CREATE(...)   FOR_EACH(IRIS_TASK_CREATE_PARAMS, __VA_ARGS__)
#define IRIS_MEMORY_PARAMS(NAME)  iris_mem  __iris_ ## NAME = NULL; 
#define IRIS_MEMORY(...)   FOR_EACH(IRIS_MEMORY_PARAMS, __VA_ARGS__)
#define IRIS_TASK_DEPENDENCY(TASK_VAR, ...)   { \
    iris_task __iris_task_deps[] = { __VA_ARGS__ }; \
    iris_task_depend(TASK_VAR, FOR_EACH_NARG(__VA_ARGS__), __iris_task_deps); \
}
#define IRIS_MEMORY_INTERMEDIATE_PARAMS(NAME)  iris_mem_intermediate(__iris_ ## NAME, 1);
#define IRIS_MEMORY_INTERMEDIATE(...)   FOR_EACH(IRIS_MEMORY_INTERMEDIATE_PARAMS, __VA_ARGS__)
#define IRIS_MEMORY_RELEASE_PARAMS(NAME)  iris_mem_release(__iris_ ## NAME);
#define IRIS_MEMORY_RELEASE(...)   FOR_EACH(IRIS_MEMORY_RELEASE_PARAMS, __VA_ARGS__)
#define IRIS_TASK_RELEASE_PARAMS(NAME)  
//iris_task_release(NAME);
#define IRIS_TASK_RELEASE(...)   FOR_EACH(IRIS_TASK_RELEASE_PARAMS, __VA_ARGS__)
#define IRIS_H2D(TASK_VAR, IRIS_MEM_VAR, OFFSET, SIZE, HOST)   iris_task_h2d(TASK_VAR, IRIS_MEM_VAR, OFFSET, SIZE, HOST)
#define IRIS_H2D_HOST_OFFSET(TASK_VAR, IRIS_MEM_VAR, OFFSET, HOST_SIZE, DEV_SIZE, ELEM_SIZE, DIM, SIZE, HOST)   iris_task_h2d_offsets(TASK_VAR, IRIS_MEM_VAR, OFFSET, HOST_SIZE, DEV_SIZE, ELEM_SIZE, DIM, SIZE, HOST)
#define IRIS_D2H(TASK_VAR, IRIS_MEM_VAR, OFFSET, SIZE, HOST)   iris_task_d2h(TASK_VAR, IRIS_MEM_VAR, OFFSET, SIZE, HOST)
#define IRIS_D2H_HOST_OFFSET(TASK_VAR, IRIS_MEM_VAR, OFFSET, HOST_SIZE, DEV_SIZE, ELEM_SIZE, DIM, SIZE, HOST)   iris_task_d2h_offsets(TASK_VAR, IRIS_MEM_VAR, OFFSET, HOST_SIZE, DEV_SIZE, ELEM_SIZE, DIM, SIZE, HOST)
#define IRIS_MEM_CREATE(IRIS_MEM_VAR, SIZE)   iris_mem  IRIS_MEM_VAR; iris_mem_create(SIZE, &IRIS_MEM_VAR)
#define IRIS_DATA_MEM_CREATE(IRIS_MEM_VAR, HOST, SIZE)   iris_mem  IRIS_MEM_VAR; iris_data_mem_create(&IRIS_MEM_VAR, HOST, SIZE)
#define IRIS_DATA_MEM_CREATE_WITH_OFFSETS(IRIS_MEM_VAR, ...)   iris_mem  IRIS_MEM_VAR; iris_data_mem_create_with_offsets(&IRIS_MEM_VAR, __VA_ARGS__)
#define IRIS_TASK_SUBMIT(TASK_VAR, TARGET_DEVICE)  iris_task_submit(TASK_VAR, TARGET_DEVICE, NULL, 1);
#define IRIS_TASK_WITH_DT(TASK_VAR, TASK_NAME, DIM, OFFSET, GWS, LWS, ...)  \
    { \
       IRIS_TASK_CONSTS(__VA_ARGS__); \
       void* __task_params[] = { IRIS_TASK_PARAMS(__VA_ARGS__) }; \
       int  __task_params_info[] = { IRIS_TASK_PARAMS_INFO(__VA_ARGS__) }; \
       int  __task_params_device_map[] = { IRIS_TASK_PARAMS_MAP(__VA_ARGS__) }; \
       IRIS_TASK_OFFSET(OFFSET); \
       IRIS_TASK_GWS(GWS); \
       IRIS_TASK_LWS(LWS); \
       IRIS_ADD_H2D(TASK_VAR, __VA_ARGS__); \
       iris_task_kernel(TASK_VAR, TASK_NAME, DIM, __st_offset, \
               __st_gws, __st_lws, sizeof(__task_params_info)/sizeof(int), \
               __task_params, __task_params_info); \
       iris_params_map(TASK_VAR, __task_params_device_map); \
       IRIS_ADD_D2H(TASK_VAR, __VA_ARGS__); \
       IRIS_TASK_DEPENDENCY_DECLARE(TASK_VAR, __VA_ARGS__); \
       IRIS_TASK_DEPENDENCY_CORE(TASK_VAR, __VA_ARGS__); \
    }
#define IRIS_TASK_NO_DT(TASK_VAR, TASK_NAME, DIM, OFFSET, GWS, LWS, ...)  \
    { \
       IRIS_TASK_CONSTS(__VA_ARGS__); \
       void* __task_params[] = { IRIS_TASK_PARAMS(__VA_ARGS__) }; \
       int  __task_params_info[] = { IRIS_TASK_PARAMS_INFO(__VA_ARGS__) }; \
       int  __task_params_device_map[] = { IRIS_TASK_PARAMS_MAP(__VA_ARGS__) }; \
       IRIS_TASK_OFFSET(OFFSET); \
       IRIS_TASK_GWS(GWS); \
       IRIS_TASK_LWS(LWS); \
       iris_task_kernel(TASK_VAR, TASK_NAME, DIM, __st_offset, \
               __st_gws, __st_lws, sizeof(__task_params_info)/sizeof(int), \
               __task_params, __task_params_info); \
       iris_params_map(TASK_VAR, __task_params_device_map); \
       IRIS_TASK_DEPENDENCY_DECLARE(TASK_VAR, __VA_ARGS__); \
       IRIS_TASK_DEPENDENCY_CORE(TASK_VAR, __VA_ARGS__); \
    }
#define IRIS_TASK_NO_SUBMIT(TASK_VAR, TASK_NAME, DIM, OFFSET, GWS, LWS, ...)  \
    iris_task TASK_VAR; \
    { \
       IRIS_TASK_CONSTS(__VA_ARGS__); \
       void* __task_params[] = { IRIS_TASK_PARAMS(__VA_ARGS__) }; \
       int  __task_params_info[] = { IRIS_TASK_PARAMS_INFO(__VA_ARGS__) }; \
       int  __task_params_device_map[] = { IRIS_TASK_PARAMS_MAP(__VA_ARGS__) }; \
       IRIS_TASK_OFFSET(OFFSET); \
       IRIS_TASK_GWS(GWS); \
       IRIS_TASK_LWS(LWS); \
       iris_task_create(&TASK_VAR); \
       IRIS_ADD_H2D(TASK_VAR, __VA_ARGS__); \
       iris_task_kernel(TASK_VAR, TASK_NAME, DIM, __st_offset, \
    	   __st_gws, __st_lws, sizeof(__task_params_info)/sizeof(int), \
    	   __task_params, __task_params_info); \
       iris_params_map(TASK_VAR, __task_params_device_map); \
       IRIS_ADD_D2H(TASK_VAR, __VA_ARGS__); \
       IRIS_TASK_DEPENDENCY_DECLARE(TASK_VAR, __VA_ARGS__); \
       IRIS_TASK_DEPENDENCY_CORE(TASK_VAR, __VA_ARGS__); \
    }
#define IRIS_TASK_CALL(TASK_VAR, TASK_NAME, TARGET_DEVICE, DIM, OFFSET, GWS, LWS, ...)  \
    iris_task TASK_VAR; \
    { \
       IRIS_TASK_CONSTS(__VA_ARGS__); \
       void* __task_params[] = { IRIS_TASK_PARAMS(__VA_ARGS__) }; \
       int  __task_params_info[] = { IRIS_TASK_PARAMS_INFO(__VA_ARGS__) }; \
       int  __task_params_device_map[] = { IRIS_TASK_PARAMS_MAP(__VA_ARGS__) }; \
       IRIS_TASK_OFFSET(OFFSET); \
       IRIS_TASK_GWS(GWS); \
       IRIS_TASK_LWS(LWS); \
       iris_task_create(&TASK_VAR); \
       IRIS_ADD_H2D(TASK_VAR, __VA_ARGS__); \
       iris_task_kernel(TASK_VAR, TASK_NAME, DIM, __st_offset, \
               __st_gws, __st_lws, sizeof(__task_params_info)/sizeof(int), \
               __task_params, __task_params_info); \
       iris_params_map(TASK_VAR, __task_params_device_map); \
       IRIS_ADD_D2H(TASK_VAR, __VA_ARGS__); \
       IRIS_TASK_DEPENDENCY_DECLARE(TASK_VAR, __VA_ARGS__); \
       IRIS_TASK_DEPENDENCY_CORE(TASK_VAR, __VA_ARGS__); \
       iris_task_submit(TASK_VAR, TARGET_DEVICE, NULL, 1); \
    }
#define IRIS_SINGLE_TASK(TASK_VAR, TASK_NAME, TARGET_DEVICE, DIM, OFFSET, GWS, LWS, ...)  \
    iris_task TASK_VAR; \
    { \
       IRIS_MEM_DECLARE(__VA_ARGS__); \
       IRIS_MEM_CREATE_INTERNAL(__VA_ARGS__); \
       IRIS_TASK_CONSTS(__VA_ARGS__); \
       void* __task_params[] = { IRIS_TASK_PARAMS(__VA_ARGS__) }; \
       int  __task_params_info[] = { IRIS_TASK_PARAMS_INFO(__VA_ARGS__) }; \
       int  __task_params_device_map[] = { IRIS_TASK_PARAMS_MAP(__VA_ARGS__) }; \
       IRIS_TASK_OFFSET(OFFSET); \
       IRIS_TASK_GWS(GWS); \
       IRIS_TASK_LWS(LWS); \
       iris_task_create(&TASK_VAR); \
       IRIS_ADD_H2D(TASK_VAR, __VA_ARGS__); \
       iris_task_kernel(TASK_VAR, TASK_NAME, DIM, __st_offset, \
               __st_gws, __st_lws, sizeof(__task_params_info)/sizeof(int), \
               __task_params, __task_params_info); \
       iris_params_map(TASK_VAR, __task_params_device_map); \
       IRIS_ADD_D2H(TASK_VAR, __VA_ARGS__); \
       IRIS_TASK_DEPENDENCY_DECLARE(TASK_VAR, __VA_ARGS__); \
       IRIS_TASK_DEPENDENCY_CORE(TASK_VAR, __VA_ARGS__); \
       iris_task_submit(TASK_VAR, TARGET_DEVICE, NULL, 1); \
       IRIS_MEM_RELEASE(__VA_ARGS__); \
    }


#define SIGNATURE_CORE_PARAM(NAME, DATA_TYPE, ...)                                                       DATA_TYPE NAME
#define SIGNATURE_CORE_PARAM_CONST(NAME, DATA_TYPE, VALUE, ...)                                          DATA_TYPE VALUE
#define SIGNATURE_CORE_VEC_PARAM(NAME, DATA_TYPE, ...)                                                   DATA_TYPE NAME
#define SIGNATURE_CORE_IN_TASK(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)                  DATA_TYPE IRIS_NAME
#define SIGNATURE_CORE_IN_TASK_DEV_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)       DATA_TYPE IRIS_NAME
#define SIGNATURE_CORE_IN_TASK_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)           DATA_TYPE IRIS_NAME
#define SIGNATURE_CORE_OUT_TASK(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)                 DATA_TYPE IRIS_NAME
#define SIGNATURE_CORE_OUT_TASK_DEV_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)      DATA_TYPE IRIS_NAME
#define SIGNATURE_CORE_OUT_TASK_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)          DATA_TYPE IRIS_NAME
#define SIGNATURE_CORE_IN_OUT_TASK(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)              DATA_TYPE IRIS_NAME
#define SIGNATURE_CORE_IN_OUT_TASK_DEV_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)   DATA_TYPE IRIS_NAME
#define SIGNATURE_CORE_IN_OUT_TASK_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)       DATA_TYPE IRIS_NAME
#define SIGNATURE_CORE_DEPENDENCY(...)   
#define SIGNATURE_CORE_REPLACE_PARAMS(NAME)      CONCATENATE(SIGNATURE_CORE_, NAME)
#define IRIS_TASK_SIGNATURE_CORE_PARAMS(...)    FOR_EACH_COMMA(SIGNATURE_CORE_REPLACE_PARAMS, __VA_ARGS__)

#define SIGNATURE_GRAPH_PARAM(NAME, DATA_TYPE, ...)                                                       DATA_TYPE NAME
#define SIGNATURE_GRAPH_PARAM_CONST(NAME, DATA_TYPE, VALUE, ...)                                          DATA_TYPE VALUE
#define SIGNATURE_GRAPH_VEC_PARAM(NAME, DATA_TYPE, ...)                                                   DATA_TYPE NAME
#define SIGNATURE_GRAPH_IN_TASK(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)                  iris_mem IRIS_VAR(IRIS_NAME)
#define SIGNATURE_GRAPH_IN_TASK_DEV_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)       iris_mem IRIS_VAR(IRIS_NAME)
#define SIGNATURE_GRAPH_IN_TASK_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)           iris_mem IRIS_VAR(IRIS_NAME)
#define SIGNATURE_GRAPH_OUT_TASK(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)                 iris_mem IRIS_VAR(IRIS_NAME)
#define SIGNATURE_GRAPH_OUT_TASK_DEV_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)      iris_mem IRIS_VAR(IRIS_NAME)
#define SIGNATURE_GRAPH_OUT_TASK_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)          iris_mem IRIS_VAR(IRIS_NAME)
#define SIGNATURE_GRAPH_IN_OUT_TASK(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)              iris_mem IRIS_VAR(IRIS_NAME)
#define SIGNATURE_GRAPH_IN_OUT_TASK_DEV_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)   iris_mem IRIS_VAR(IRIS_NAME)
#define SIGNATURE_GRAPH_IN_OUT_TASK_OFFSET(IRIS_NAME, DATA_TYPE, ELEMENT_TYPE, VARIABLE, SIZE ...)       iris_mem IRIS_VAR(IRIS_NAME)
#define SIGNATURE_GRAPH_DEPENDENCY(...)   
#define SIGNATURE_GRAPH_REPLACE_PARAMS(NAME)      CONCATENATE(SIGNATURE_GRAPH_, NAME)
#define IRIS_TASK_SIGNATURE_GRAPH_PARAMS(...)   FOR_EACH_COMMA(SIGNATURE_GRAPH_REPLACE_PARAMS, __VA_ARGS__)


#ifdef __cplusplus
#define IRIS_TASK_CORE_SIGNATURE_CPP(API_NAME, ...) \
    int API_NAME(int target_dev, IRIS_TASK_SIGNATURE_CORE_PARAMS(__VA_ARGS__))
#define IRIS_TASK_GRAPH_SIGNATURE_CPP(API_NAME, ...) \
    int API_NAME(iris_task task, IRIS_TASK_SIGNATURE_GRAPH_PARAMS(__VA_ARGS__))
#endif //__cplusplus

#ifdef __cplusplus
#define IRIS_TASK_CORE_SIGNATURE(API_NAME, ...) \
    extern "C" int API_NAME(int target_dev, IRIS_TASK_SIGNATURE_CORE_PARAMS(__VA_ARGS__))
#define IRIS_TASK_GRAPH_SIGNATURE(API_NAME, ...) \
    extern "C" int API_NAME(iris_task task, IRIS_TASK_SIGNATURE_GRAPH_PARAMS(__VA_ARGS__))
#else //__cplusplus
#define IRIS_TASK_CORE_SIGNATURE(API_NAME, ...) \
    int API_NAME(int target_dev, IRIS_TASK_SIGNATURE_CORE_PARAMS(__VA_ARGS__))
#define IRIS_TASK_GRAPH_SIGNATURE(API_NAME, ...) \
    int API_NAME(iris_task task, IRIS_TASK_SIGNATURE_GRAPH_PARAMS(__VA_ARGS__))
#endif //__cplusplus

#ifndef IRIS_API_DEFINITION
#define IRIS_TASK_API_DECL(API_NAME, ...) \
    IRIS_TASK_CORE_SIGNATURE(CORE_API_NAME, __VA_ARGS__); 
#define IRIS_TASK_APIS(CORE_API_NAME, GRAPH_API_NAME, KERNEL_NAME, DIM, OFFSET, GWS, LWS, ...) \
    IRIS_TASK_CORE_SIGNATURE(CORE_API_NAME, __VA_ARGS__); \
    IRIS_TASK_GRAPH_SIGNATURE(GRAPH_API_NAME, __VA_ARGS__); 
#ifdef __cplusplus
#define IRIS_TASK_APIS_CPP(CPP_API_NAME, CORE_API_NAME, GRAPH_API_NAME, KERNEL_NAME, DIM, OFFSET, GWS, LWS, ...) \
    IRIS_TASK_CORE_SIGNATURE(CORE_API_NAME, __VA_ARGS__); \
    IRIS_TASK_GRAPH_SIGNATURE(GRAPH_API_NAME, __VA_ARGS__); \
    IRIS_TASK_CORE_SIGNATURE_CPP(CPP_API_NAME, __VA_ARGS__); \
    IRIS_TASK_GRAPH_SIGNATURE_CPP(CPP_API_NAME, __VA_ARGS__); 
#else //__cplusplus
#define IRIS_TASK_APIS_CPP(CPP_API_NAME, ...)  IRIS_TASK_APIS(__VA_ARGS__) 
#endif //__cplusplus
#else // IRIS_API_DEFINITION
#define IRIS_TASK_APIS(CORE_API_NAME, GRAPH_API_NAME, KERNEL_NAME, DIM, OFFSET, GWS, LWS, ...) \
    IRIS_TASK_CORE_SIGNATURE(CORE_API_NAME, __VA_ARGS__) \
    { \
      IRIS_SINGLE_TASK(task, KERNEL_NAME, target_dev, DIM, OFFSET, GWS, LWS, __VA_ARGS__); \
      return IRIS_SUCCESS; \
    } \
    IRIS_TASK_GRAPH_SIGNATURE(GRAPH_API_NAME, __VA_ARGS__) \
    { \
      IRIS_TASK_NO_DT(task, KERNEL_NAME, DIM, OFFSET, GWS, LWS, __VA_ARGS__); \
      return IRIS_SUCCESS; \
    } 
#ifdef __cplusplus
#define IRIS_TASK_APIS_CPP(CPP_API_NAME, CORE_API_NAME, GRAPH_API_NAME, KERNEL_NAME, DIM, OFFSET, GWS, LWS, ...) \
    IRIS_TASK_CORE_SIGNATURE(CORE_API_NAME, __VA_ARGS__) \
    { \
      IRIS_SINGLE_TASK(task, KERNEL_NAME, target_dev, DIM, OFFSET, GWS, LWS, __VA_ARGS__); \
      return IRIS_SUCCESS; \
    } \
    IRIS_TASK_GRAPH_SIGNATURE(GRAPH_API_NAME, __VA_ARGS__) \
    { \
      IRIS_TASK_NO_DT(task, KERNEL_NAME, DIM, OFFSET, GWS, LWS, __VA_ARGS__); \
      return IRIS_SUCCESS; \
    } \
    IRIS_TASK_CORE_SIGNATURE_CPP(CPP_API_NAME, __VA_ARGS__) \
    { \
      IRIS_SINGLE_TASK(task, KERNEL_NAME, target_dev, DIM, OFFSET, GWS, LWS, __VA_ARGS__); \
      return IRIS_SUCCESS; \
    } \
    IRIS_TASK_GRAPH_SIGNATURE_CPP(CPP_API_NAME, __VA_ARGS__) \
    { \
      IRIS_TASK_NO_DT(task, KERNEL_NAME, DIM, OFFSET, GWS, LWS, __VA_ARGS__); \
      return IRIS_SUCCESS; \
    } 
#else //__cplusplus
#define IRIS_TASK_APIS_CPP(CPP_API_NAME, ...)  IRIS_TASK_APIS(__VA_ARGS__) 
#endif //__cplusplus
#endif // IRIS_API_DEFINITION


#define IRIS_CREATE_APIS(...)
#endif //UNDEF_IRIS_MACROS
#endif //__IRIS_MACROS_H
