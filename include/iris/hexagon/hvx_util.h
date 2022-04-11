
/*
 * hvx_util.h
 *
 * ------------------------------------------------------------------------------
 * Copyright (c) 2016-2018 QUALCOMM Technologies Incorporated.
 * All Rights Reserved Qualcomm Proprietary
 * ------------------------------------------------------------------------------
 *  Created on: 2016-05-30
 *      
 */
#ifndef HVX_UTIL_H_INCLUDED
#define HVX_UTIL_H_INCLUDED 1

#include <stdint.h>

#ifndef __hexagon__
#define HVX_LIBNATIVE
#endif

#ifdef __hexagon__
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
// Need this because of byte-enabled vector stores, which appear
// in 8.1.02. They are also in libnative, but are unusable there
// because of how HVX_VectorAddr is defined.
//
#include "hexagon_circ_brev_intrinsics.h"
#endif
#ifdef HVX_LIBNATIVE
#include "hvx_hexagon_protos.h"
#endif

#ifdef __hexagon__
#define HVX_INLINE_ALWAYS __inline __attribute__((always_inline))
#else
#define HVX_INLINE_ALWAYS
#endif


// for multiple function returns, etc
// All of these are declared in terms of
// the underlying types (HEXAGON_Vect512 etc)
// and then typdef'd according to __HVXDBL__
#ifdef __cplusplus
template <typename T,int N>
struct HEXAGON_pack {
    T val[N];
};
#ifndef __HVXDBL__
typedef HEXAGON_pack<HEXAGON_Vect512,2> HEXAGON_Vect512_x2;
typedef HEXAGON_pack<HEXAGON_Vect512,3> HEXAGON_Vect512_x3;
typedef HEXAGON_pack<HEXAGON_Vect512,4> HEXAGON_Vect512_x4;
#else
typedef HEXAGON_pack<HEXAGON_Vect2048,2> HEXAGON_Vect2048_x2;
typedef HEXAGON_pack<HEXAGON_Vect2048,3> HEXAGON_Vect2048_x3;
typedef HEXAGON_pack<HEXAGON_Vect2048,4> HEXAGON_Vect2048_x4;
#endif
typedef HEXAGON_pack<HEXAGON_Vect1024,2> HEXAGON_Vect1024_x2;
typedef HEXAGON_pack<HEXAGON_Vect1024,3> HEXAGON_Vect1024_x3;
typedef HEXAGON_pack<HEXAGON_Vect1024,4> HEXAGON_Vect1024_x4;
#else // not c++
#ifndef __HVXDBL__
struct HEXAGON_Vector512_x2 {
    HEXAGON_Vect512 val[2];
};
struct HEXAGON_Vect512_x3 {
    HEXAGON_Vect512 val[3];
};
struct HEXAGON_Vect512_x4 {
    HEXAGON_Vect512 val[4];
};
#else
struct HEXAGON_Vector2048_x2 {
    HEXAGON_Vector2048 val[2];
};
struct HEXAGON_Vector2048_x3 {
    HEXAGON_Vector2048 val[3];
};
struct HEXAGON_Vector2048_x4 {
    HEXAGON_Vector2048 val[4];
};
#endif
struct HEXAGON_Vector1024_x2 {
    HEXAGON_Vect1024 val[2];
};
struct HEXAGON_Vect1024_x3 {
    HEXAGON_Vect1024 val[3];
};
struct HEXAGON_Vect1024_x4 {
    HEXAGON_Vect1024 val[4];
};
#endif // c++

#ifndef __HVXDBL__
typedef HEXAGON_Vect512_x2 HVX_Vector_x2;
typedef HEXAGON_Vect512_x3 HVX_Vector_x3;
typedef HEXAGON_Vect512_x4 HVX_Vector_x4;
typedef HEXAGON_Vect1024_x2 HVX_VectorPair_x2;
typedef HEXAGON_Vect1024_x3 HVX_VectorPair_x3;
typedef HEXAGON_Vect1024_x4 HVX_VectorPair_x4;
#else
typedef HEXAGON_Vect1024_x2 HVX_Vector_x2;
typedef HEXAGON_Vect1024_x3 HVX_Vector_x3;
typedef HEXAGON_Vect1024_x4 HVX_Vector_x4;
typedef HEXAGON_Vect2048_x2 HVX_VectorPair_x2;
typedef HEXAGON_Vect2048_x3 HVX_VectorPair_x3;
typedef HEXAGON_Vect2048_x4 HVX_VectorPair_x4;
#endif
#ifdef __cplusplus
extern "C" {
#endif
//
// these unions are for declaring a table of HVX vectors initialized as i16, etc
//
// e.g.
//  static const HVX_Vector_union_i16 Ramp = { {  0,1,2,3,4, /*...*/ 31 } };
//
// HVX_Vector vramp = Ramp.as_v;
//
//
#ifndef __HVXDBL__
typedef union {
    int8_t as_i8[ sizeof(HEXAGON_Vect512)];
    HEXAGON_Vect512 as_v;
} HEXAGON_Vect512_union_i8;
typedef union {
    uint8_t as_u8[ sizeof(HEXAGON_Vect512)];
    HEXAGON_Vect512 as_v;
} HEXAGON_Vect512_union_u8;

typedef union {
    int16_t as_i16[ sizeof(HEXAGON_Vect512)/2];
    HEXAGON_Vect512 as_v;
} HEXAGON_Vect512_union_i16;
typedef union {
    uint16_t as_u16[  sizeof(HEXAGON_Vect512)/2];
    HEXAGON_Vect512 as_v;
} HEXAGON_Vect512_union_u16;
typedef union {
    int32_t as_i32[  sizeof(HEXAGON_Vect512)/4];
    HEXAGON_Vect512 as_v;
} HEXAGON_Vect512_union_i32;
typedef union {
    uint32_t as_u32[  sizeof(HEXAGON_Vect512)/4];
    HEXAGON_Vect512 as_v;
} HEXAGON_Vect512_union_u32;

typedef HEXAGON_Vect512_union_i8 HVX_Vector_union_i8;
typedef HEXAGON_Vect512_union_u8 HVX_Vector_union_u8;
typedef HEXAGON_Vect512_union_i16 HVX_Vector_union_i16;
typedef HEXAGON_Vect512_union_u16 HVX_Vector_union_u16;
typedef HEXAGON_Vect512_union_i32 HVX_Vector_union_i32;
typedef HEXAGON_Vect512_union_u32 HVX_Vector_union_u32;
#endif

typedef union {
    int8_t as_i8[ sizeof(HEXAGON_Vect1024)];
    HEXAGON_Vect1024 as_v;
} HEXAGON_Vect1024_union_i8;
typedef union {
    uint8_t as_u8[ sizeof(HEXAGON_Vect1024)];
    HEXAGON_Vect1024 as_v;
} HEXAGON_Vect1024_union_u8;

typedef union {
    int16_t as_i16[ sizeof(HEXAGON_Vect1024)/2];
    HEXAGON_Vect1024 as_v;
} HEXAGON_Vect1024_union_i16;
typedef union {
    uint16_t as_u16[  sizeof(HEXAGON_Vect1024)/2];
    HEXAGON_Vect1024 as_v;
} HEXAGON_Vect1024_union_u16;
typedef union {
    int32_t as_i32[  sizeof(HEXAGON_Vect1024)/4];
    HEXAGON_Vect1024 as_v;
} HEXAGON_Vect1024_union_i32;
typedef union {
    uint32_t as_u32[  sizeof(HEXAGON_Vect1024)/4];
    HEXAGON_Vect1024 as_v;
} HEXAGON_Vect1024_union_u32;

#ifdef __HVXDBL__
typedef HEXAGON_Vect1024_union_i8 HVX_Vector_union_i8;
typedef HEXAGON_Vect1024_union_u8 HVX_Vector_union_u8;
typedef HEXAGON_Vect1024_union_i16 HVX_Vector_union_i16;
typedef HEXAGON_Vect1024_union_u16 HVX_Vector_union_u16;
typedef HEXAGON_Vect1024_union_i32 HVX_Vector_union_i32;
typedef HEXAGON_Vect1024_union_u32 HVX_Vector_union_u32;
#else
typedef HEXAGON_Vect1024_union_i8 HVX_VectorPair_union_i8;
typedef HEXAGON_Vect1024_union_u8 HVX_VectorPair_union_u8;
typedef HEXAGON_Vect1024_union_i16 HVX_VectorPair_union_i16;
typedef HEXAGON_Vect1024_union_u16 HVX_VectorPair_union_u16;
typedef HEXAGON_Vect1024_union_i32 HVX_VectorPair_union_i32;
typedef HEXAGON_Vect1024_union_u32 HVX_VectorPair_union_u32;
#endif


#ifdef __HVXDBL__
typedef union {
    int8_t as_i8[ sizeof(HEXAGON_Vect2048)];
    HEXAGON_Vect2048 as_v;
} HEXAGON_Vect2048_union_i8;
typedef union {
    uint8_t as_u8[ sizeof(HEXAGON_Vect2048)];
    HEXAGON_Vect2048 as_v;
} HEXAGON_Vect2048_union_u8;

typedef union {
    int16_t as_i16[ sizeof(HEXAGON_Vect2048)/2];
    HEXAGON_Vect2048 as_v;
} HEXAGON_Vect2048_union_i16;
typedef union {
    uint16_t as_u16[  sizeof(HEXAGON_Vect2048)/2];
    HEXAGON_Vect2048 as_v;
} HEXAGON_Vect2048_union_u16;
typedef union {
    int32_t as_i32[  sizeof(HEXAGON_Vect2048)/4];
    HEXAGON_Vect2048 as_v;
} HEXAGON_Vect2048_union_i32;
typedef union {
    uint32_t as_u32[  sizeof(HEXAGON_Vect2048)/4];
    HEXAGON_Vect2048 as_v;
} HEXAGON_Vect2048_union_u32;
typedef HEXAGON_Vect2048_union_i8 HVX_VectorPair_union_i8;
typedef HEXAGON_Vect2048_union_u8 HVX_VectorPair_union_u8;
typedef HEXAGON_Vect2048_union_i16 HVX_VectorPair_union_i16;
typedef HEXAGON_Vect2048_union_u16 HVX_VectorPair_union_u16;
typedef HEXAGON_Vect2048_union_i32 HVX_VectorPair_union_i32;
typedef HEXAGON_Vect2048_union_u32 HVX_VectorPair_union_u32;

#endif

//
// The 'q6op_' are all things that can be done in a few
// hvx operations, and using the same type of naming convention with
// q6op_ instead of Q6_.
// A few of these are operations which are available exactly on
// later versions, and in this case the operations will map to those when
// available, and stay as inline functions when not.
//
//
// it is important that these *always* be inlined, since they
// they are all specific to the vector size, so an actual function
// would only work with the size it was compiled for.


// this is a Vb_vnavg_VbVb:
// done by xoring each input w. 0x80 and then Vb_vnavg_VubVub
//
#if defined( __hexagon__) &&  __HEXAGON_ARCH__  >= 65
#define q6op_Vb_vnavg_VbVb(a,b) Q6_Vb_vnavg_VbVb(a,b)
#else
static HVX_INLINE_ALWAYS
HVX_Vector
q6op_Vb_vnavg_VbVb( HVX_Vector a, HVX_Vector b)
{

    HVX_Vector k_80 = Q6_V_vsplat_R(0x80808080);
    return Q6_Vb_vnavg_VubVub( Q6_V_vxor_VV( a, k_80), Q6_V_vxor_VV( b, k_80) );
}
#endif

//
// Predicate shuffle - emulated for v60
//
static HVX_INLINE_ALWAYS
HVX_VectorPred q6op_Qb_vshuffe_QhQh( HVX_VectorPred Qs, HVX_VectorPred Qt )
{
#if	__HEXAGON_ARCH__ >= 62
	return Q6_Qb_vshuffe_QhQh(Qs,Qt);
#else
	HVX_VectorPred even_b_lanes = Q6_Q_vand_VR( Q6_V_vsplat_R(0x010001), 0x10001);

	return Q6_Q_or_QQ( Q6_Q_and_QQ( Qt,even_b_lanes), Q6_Q_and_QQn( Qs,even_b_lanes));
#endif
}
static HVX_INLINE_ALWAYS
HVX_VectorPred q6op_Qh_vshuffe_QwQw( HVX_VectorPred Qs, HVX_VectorPred Qt )
{
#if	__HEXAGON_ARCH__ >= 62
	return Q6_Qh_vshuffe_QwQw(Qs,Qt);
#else
	HVX_VectorPred even_h_lanes = Q6_Q_vand_VR( Q6_V_vsplat_R(0x0101), 0x101);

	return Q6_Q_or_QQ( Q6_Q_and_QQ( Qt,even_h_lanes), Q6_Q_and_QQn( Qs,even_h_lanes));
#endif
}

// vector & pred - emulated for v60
//

static HVX_INLINE_ALWAYS
HVX_Vector q6op_V_vand_QV(HVX_VectorPred Qv, HVX_Vector Vu)
{
#if	__HEXAGON_ARCH__ >= 62
    return Q6_V_vand_QV( Qv, Vu);
#else
    return Q6_V_vmux_QVV( Qv, Vu, Q6_V_vzero());
#endif
}

static HVX_INLINE_ALWAYS
HVX_Vector q6op_V_vand_QnV(HVX_VectorPred Qv, HVX_Vector Vu)
{
#if	__HEXAGON_ARCH__ >= 62
    return Q6_V_vand_QnV( Qv, Vu);
#else
    return Q6_V_vmux_QVV( Qv, Q6_V_vzero(),Vu);
#endif
}
//
// emulate Q6_Ww_vmpyacc_WwVhRh for < V65
//
static HVX_INLINE_ALWAYS
HVX_VectorPair q6op_Ww_vmpyacc_WwVhRh(HVX_VectorPair Vxx, HVX_Vector Vu, int Rt)
{
#if	__HEXAGON_ARCH__ >= 65
    return Q6_Ww_vmpyacc_WwVhRh( Vxx, Vu,Rt);
#else
    return Q6_Ww_vadd_WwWw( Vxx, Q6_Ww_vmpy_VhRh(Vu,Rt));
#endif

}

//
// 32x32 fractional multiply - expands to two ops
//  equiv to :
//    p  = (a*b + (1<<30)) >> 31     [with rounding]
//    p  = a*b >> 31     			[without rounding]
// The 'sat' only takes effect when both inputs
// are -0x80000000 and causes the result to saturate to 0x7fffffff
//
static HVX_INLINE_ALWAYS
HVX_Vector q6op_Vw_vmpy_VwVw_s1_rnd_sat( HVX_Vector vu, HVX_Vector vv) {
	return Q6_Vw_vmpyoacc_VwVwVh_s1_rnd_sat_shift( Q6_Vw_vmpye_VwVuh( vu, vv), vu, vv );
}
static HVX_INLINE_ALWAYS
HVX_Vector q6op_Vw_vmpy_VwVw_s1_sat( HVX_Vector vu, HVX_Vector vv) {
	return Q6_Vw_vmpyoacc_VwVwVh_s1_sat_shift( Q6_Vw_vmpye_VwVuh( vu, vv), vu, vv );
}


//
// Splat from 16 bits
// Note: you can use this for constants, e.g. q6op_Vh_vsplat_Rh(0x12), the compiler
// will just simplify it to Q6_V_vsplat_R(0x120012).
//
static HVX_INLINE_ALWAYS
HVX_Vector q6op_Vh_vsplat_R(int r)
{
#if defined( __hexagon__) &&  __HEXAGON_ARCH__ >= 62
    return Q6_Vh_vsplat_R(r);
#else
    return Q6_V_vsplat_R( Q6_R_combine_RlRl(r,r));
#endif
}
// splat from 8 bits
static HVX_INLINE_ALWAYS
HVX_Vector q6op_Vb_vsplat_R(int r)
{
#if defined( __hexagon__) &&  __HEXAGON_ARCH__ >= 62
    return Q6_Vb_vsplat_R(r);
#else
    return Q6_V_vsplat_R(Q6_R_vsplatb_R(r));
#endif
}

// legacy naming convention
#define q6op_Vh_vsplat_Rh(r) q6op_Vh_vsplat_R(r)

// add vector to scalar, 32 bits
static HVX_INLINE_ALWAYS
HVX_Vector q6op_Vw_vadd_VwRw(HVX_Vector v, int r)
{
    return Q6_Vw_vadd_VwVw( v, Q6_V_vsplat_R(r));
}
// add  scalar, {16,16} bits, to vector

static HVX_INLINE_ALWAYS HVX_Vector q6op_Vh_vadd_VhRh(HVX_Vector v, int r)
{
    return Q6_Vh_vadd_VhVh( v, Q6_V_vsplat_R(r));
}
// min and max vector vs scalar {16,16} bits
static HVX_INLINE_ALWAYS HVX_Vector q6op_Vh_vmin_VhRh(HVX_Vector v, int r)
{
    return Q6_Vh_vmin_VhVh( v, Q6_V_vsplat_R(r));
}
static HVX_INLINE_ALWAYS HVX_Vector q6op_Vh_vmax_VhRh(HVX_Vector v, int r)
{
    return Q6_Vh_vmax_VhVh( v, Q6_V_vsplat_R(r));
}
static HVX_INLINE_ALWAYS HVX_Vector q6op_Vuh_vmin_VuhRuh(HVX_Vector v, int r)
{
    return Q6_Vuh_vmin_VuhVuh( v, Q6_V_vsplat_R(r));
}
static HVX_INLINE_ALWAYS HVX_Vector q6op_Vuh_vmax_VuhRuh(HVX_Vector v, int r)
{
    return Q6_Vuh_vmax_VuhVuh( v, Q6_V_vsplat_R(r));
}

//////////////////////////////////////////////////////////////////////
// 'reducing shifts' which take a VectorPair instead of two vectors
//////////////////////////////////////////////////////////////////////
static HVX_INLINE_ALWAYS HVX_Vector q6op_Vh_vasr_WwR ( HVX_VectorPair a, int sh )
{
    return Q6_Vh_vasr_VwVwR( Q6_V_hi_W(a), Q6_V_lo_W(a), sh);
}
static HVX_INLINE_ALWAYS HVX_Vector q6op_Vh_vasr_WwR_sat ( HVX_VectorPair a, int sh )
{
    return Q6_Vh_vasr_VwVwR_sat( Q6_V_hi_W(a), Q6_V_lo_W(a), sh);
}

static HVX_INLINE_ALWAYS HVX_Vector q6op_Vh_vasr_WwR_rnd_sat ( HVX_VectorPair a, int sh )
{
    return Q6_Vh_vasr_VwVwR_rnd_sat( Q6_V_hi_W(a), Q6_V_lo_W(a), sh);
}
#if __HEXAGON_ARCH__ >= 62
static HVX_INLINE_ALWAYS HVX_Vector q6op_Vuh_vasr_WwR_rnd_sat ( HVX_VectorPair a, int sh )
{
    return Q6_Vuh_vasr_VwVwR_rnd_sat( Q6_V_hi_W(a), Q6_V_lo_W(a), sh);
}
#endif
static HVX_INLINE_ALWAYS HVX_Vector q6op_Vuh_vasr_WwR_sat ( HVX_VectorPair a, int sh )
{
    return Q6_Vuh_vasr_VwVwR_sat( Q6_V_hi_W(a), Q6_V_lo_W(a), sh);
}

static HVX_INLINE_ALWAYS HVX_Vector q6op_Vub_vasr_WhR_sat ( HVX_VectorPair a, int sh )
{
    return Q6_Vub_vasr_VhVhR_sat( Q6_V_hi_W(a), Q6_V_lo_W(a), sh);
}

static HVX_INLINE_ALWAYS HVX_Vector q6op_Vub_vasr_WhR_rnd_sat ( HVX_VectorPair a, int sh )
{
    return Q6_Vub_vasr_VhVhR_rnd_sat( Q6_V_hi_W(a), Q6_V_lo_W(a), sh);
}

#if __HEXAGON_ARCH__ >= 62
static HVX_INLINE_ALWAYS HVX_Vector q6op_Vb_vasr_WhR_sat ( HVX_VectorPair a, int sh )
{
    return Q6_Vb_vasr_VhVhR_sat( Q6_V_hi_W(a), Q6_V_lo_W(a), sh);
}
#endif


static HVX_INLINE_ALWAYS HVX_Vector q6op_Vb_vasr_WhR_rnd_sat ( HVX_VectorPair a, int sh )
{
    return Q6_Vb_vasr_VhVhR_rnd_sat( Q6_V_hi_W(a), Q6_V_lo_W(a), sh);
}

//////////////////////////////////////////////////////////////////////
// 'round' which take a VectorPair instead of two vectors
//////////////////////////////////////////////////////////////////////

static HVX_INLINE_ALWAYS HVX_Vector q6op_Vh_vround_Ww_sat ( HVX_VectorPair a )
{
    return Q6_Vh_vround_VwVw_sat( Q6_V_hi_W(a), Q6_V_lo_W(a) );
}
static HVX_INLINE_ALWAYS HVX_Vector q6op_Vuh_vround_Ww_sat ( HVX_VectorPair a )
{
    return Q6_Vuh_vround_VwVw_sat( Q6_V_hi_W(a), Q6_V_lo_W(a) );
}
#if __HEXAGON_ARCH__ >= 62

static HVX_INLINE_ALWAYS HVX_Vector q6op_Vuh_vround_Wuw_sat ( HVX_VectorPair a )
{
    return Q6_Vuh_vround_VuwVuw_sat( Q6_V_hi_W(a), Q6_V_lo_W(a) );
}
#endif

static HVX_INLINE_ALWAYS HVX_Vector q6op_Vb_vround_Wh_sat ( HVX_VectorPair a )
{
    return Q6_Vb_vround_VhVh_sat( Q6_V_hi_W(a), Q6_V_lo_W(a) );
}

static HVX_INLINE_ALWAYS HVX_Vector q6op_Vub_vround_Wh_sat ( HVX_VectorPair a )
{
    return Q6_Vub_vround_VhVh_sat( Q6_V_hi_W(a), Q6_V_lo_W(a) );
}
#if __HEXAGON_ARCH__ >= 62
static HVX_INLINE_ALWAYS HVX_Vector q6op_Vub_vround_VuhVuh_sat ( HVX_VectorPair a )
{
    return Q6_Vub_vround_VuhVuh_sat( Q6_V_hi_W(a), Q6_V_lo_W(a) );
}
#endif

//////////////////////////////////////////////////////////////////
// pack operations (with VectorPair input instead of two vectors)
/////////////////////////////////////////////////////////////////
static HVX_INLINE_ALWAYS HVX_Vector q6op_Vb_vpacke_Wh ( HVX_VectorPair a )
{
    return Q6_Vb_vpacke_VhVh( Q6_V_hi_W(a), Q6_V_lo_W(a) );
}
static HVX_INLINE_ALWAYS HVX_Vector q6op_Vh_vpacke_Ww ( HVX_VectorPair a )
{
    return Q6_Vh_vpacke_VwVw( Q6_V_hi_W(a), Q6_V_lo_W(a) );
}

static HVX_INLINE_ALWAYS HVX_Vector q6op_Vb_vpacko_Wh ( HVX_VectorPair a )
{
    return Q6_Vb_vpacko_VhVh( Q6_V_hi_W(a), Q6_V_lo_W(a) );
}
static HVX_INLINE_ALWAYS HVX_Vector q6op_Vh_vpacko_Ww ( HVX_VectorPair a )
{
    return Q6_Vh_vpacko_VwVw( Q6_V_hi_W(a), Q6_V_lo_W(a) );
}
static HVX_INLINE_ALWAYS HVX_Vector q6op_Vub_vpack_Wh_sat ( HVX_VectorPair a )
{
    return Q6_Vub_vpack_VhVh_sat( Q6_V_hi_W(a), Q6_V_lo_W(a) );
}
static HVX_INLINE_ALWAYS HVX_Vector q6op_Vb_vpack_Wh_sat ( HVX_VectorPair a )
{
    return Q6_Vb_vpack_VhVh_sat( Q6_V_hi_W(a), Q6_V_lo_W(a) );
}
static HVX_INLINE_ALWAYS HVX_Vector q6op_Vuh_vpack_Ww_sat ( HVX_VectorPair a )
{
    return Q6_Vuh_vpack_VwVw_sat( Q6_V_hi_W(a), Q6_V_lo_W(a) );
}
static HVX_INLINE_ALWAYS HVX_Vector q6op_Vh_vpack_Ww_sat ( HVX_VectorPair a )
{
    return Q6_Vh_vpack_VwVw_sat( Q6_V_hi_W(a), Q6_V_lo_W(a) );
}
////////////////////////////////////
// shift-by-R operations, applied to HVX_VectorPair
////////////////////////////////////

static HVX_INLINE_ALWAYS HVX_VectorPair q6op_Ww_vasl_WwR ( HVX_VectorPair a, int sh )
{
    return Q6_W_vcombine_VV(
            Q6_Vw_vasl_VwR ( Q6_V_hi_W(a), sh ),
            Q6_Vw_vasl_VwR ( Q6_V_lo_W(a), sh ));
}
static HVX_INLINE_ALWAYS HVX_VectorPair q6op_Ww_vasr_WwR ( HVX_VectorPair a, int sh )
{
    return Q6_W_vcombine_VV(
            Q6_Vw_vasr_VwR ( Q6_V_hi_W(a), sh ),
            Q6_Vw_vasr_VwR ( Q6_V_lo_W(a), sh ));
}

static HVX_INLINE_ALWAYS HVX_VectorPair q6op_Wuw_vlsr_WuwR ( HVX_VectorPair a, int sh )
{
    return Q6_W_vcombine_VV(
            Q6_Vuw_vlsr_VuwR ( Q6_V_hi_W(a), sh ),
            Q6_Vuw_vlsr_VuwR ( Q6_V_lo_W(a), sh ));
}


static HVX_INLINE_ALWAYS HVX_VectorPair q6op_Wh_vasl_WhR ( HVX_VectorPair a, int sh )
{
    return Q6_W_vcombine_VV(
            Q6_Vh_vasl_VhR ( Q6_V_hi_W(a), sh ),
            Q6_Vh_vasl_VhR ( Q6_V_lo_W(a), sh ));
}
static HVX_INLINE_ALWAYS HVX_VectorPair q6op_Wh_vasr_WhR ( HVX_VectorPair a, int sh )
{
    return Q6_W_vcombine_VV(
            Q6_Vh_vasr_VhR ( Q6_V_hi_W(a), sh ),
            Q6_Vh_vasr_VhR ( Q6_V_lo_W(a), sh ));
}

static HVX_INLINE_ALWAYS HVX_VectorPair q6op_Wuh_vlsr_WuhR ( HVX_VectorPair a, int sh )
{
    return Q6_W_vcombine_VV(
            Q6_Vuh_vlsr_VuhR ( Q6_V_hi_W(a), sh ),
            Q6_Vuh_vlsr_VuhR ( Q6_V_lo_W(a), sh ));
}

#if __HEXAGON_ARCH__ >= 62
static HVX_INLINE_ALWAYS HVX_VectorPair q6op_Wub_vlsr_WubR ( HVX_VectorPair a, int sh )
{

    return Q6_W_vcombine_VV(
            Q6_Vub_vlsr_VubR ( Q6_V_hi_W(a), sh ),
            Q6_Vub_vlsr_VubR ( Q6_V_lo_W(a), sh ));
}
#endif

// q6op_W_vzero()
// Oddly, Q6_W_vzero() is available starting in 8.1.02
// but only for V65 (??? this is not how intrinsics are supposed to work.
// other processors have zero too, and can do vsub_WwWw ... ???).
// This wrapper tests for the presence of that by testing for the #define
//
static HVX_INLINE_ALWAYS HVX_VectorPair
q6op_W_vzero() {
#if defined(__hexagon__) && defined( Q6_W_vzero)
    return Q6_W_vzero();
#else
    return Q6_W_vcombine_VV( Q6_V_vzero(),  Q6_V_vzero());
#endif
}


// v60 doesn't have vsetq2
static HVX_INLINE_ALWAYS
HVX_VectorPred q6op_Q_vsetq2_R( int rval)
{
#if __HEXAGON_ARCH__ >= 62
	return Q6_Q_vsetq2_R(rval);
#else
    HVX_VectorPred result = Q6_Q_vsetq_R(rval);
    static const unsigned VECN = sizeof(HVX_Vector);
// Q6_vmaskedstorenq_QAV is being used to detect compiler >= 8.1.02
// earlier ones will ICE on the short version of the below (generating
// VecPred based on scalar condition)
//
#if defined(Q6_vmaskedstorenq_QAV)
    if ( ((unsigned)rval &(VECN-1)) == 0 ) result = Q6_Q_not_Q(result);
#else
    //
    // if rval is a multiple of VECN, force to all ones, otherwise
    // leave it alone.
    //
    unsigned ones = -1;
    unsigned tcond = ( ((unsigned)rval &(VECN-1)) == 0 )? ones: 0;
    result =  Q6_Q_vandor_QVR( result, Q6_V_vsplat_R(ones),tcond );
#endif
    return result;
#endif
}


#ifndef __hexagon__
// functions to do fake conditional store on libnative.
// These functions are dependent of the vector size (so must be 'static').
// Note that a 'VecPred' is defined using the same type as a Vector, but it has all
// the 1's packed in in the initial bytes. I don't want to rely on that here, so
// I convert it to a vector before use.
static void fake_conditional_store( HVX_VectorPred const *cond, void *addr, void const *v){
    HVX_Vector cv = Q6_V_vand_QR( *cond, 0x01010101);   // convert to vector
    uint8_t const *cp = (uint8_t const*)&cv;
    uint8_t const *sp = (uint8_t const*)v;
    uint8_t*dp = (uint8_t*)addr;
    for( int i =0; i < (int)sizeof(HVX_Vector); i ++ ){
        if( cp[i] ){
            dp[i] = sp[i];
        }
    }    
}
static void fake_Nconditional_store( HVX_VectorPred const *cond, void *addr, void const *v){
    HVX_Vector cv = Q6_V_vand_QR( *cond, 0x01010101);   // convert to vector
    uint8_t const *cp = (uint8_t const*)&cv;
    uint8_t const *sp = (uint8_t const*)v;
    uint8_t*dp = (uint8_t*)addr;
    for( int i =0; i < (int)sizeof(HVX_Vector); i ++ ){
        if( !cp[i] ){
            dp[i] = sp[i];
        }
    }    
}
#endif

//
// unaligned vector load
// Done by stating that the compiler should read a vector within a packed
// data structure.
//
static HVX_INLINE_ALWAYS HVX_Vector q6op_V_vldu_A(  HVX_Vector const *addr )
{
#ifdef __hexagon__
    struct varr { HVX_Vector v;}__attribute__((packed)) const *pp;
    pp = (struct varr const *)addr;
    return pp->v;
#else
    return *addr;       // assume libnative host can do this
#endif
}
// unaligned vector store.
static HVX_INLINE_ALWAYS void q6op_vstu_AV(  HVX_Vector *addr, HVX_Vector v )
{
#ifdef __hexagon__
    struct varr { HVX_Vector v;}__attribute__((packed)) *pp;
    pp = (struct varr *)addr;
    pp->v = v;
#else
    *addr = v;       // assume libnative host can do this
#endif
}

// 
//
// conditional vector store
//   void q6op_vstcc_[n]QAV[_nt] ( cond, addr, vec );
// These do *not* work in HVXDBL on 7.4.01 (compiler will abort with getRegForInlineAsmConstraint Unhandled data type)
// occurs if these are used (it's ok to just have them in the header though). Seem to be ok with 8.0.05
//
#if 1
static HVX_INLINE_ALWAYS void q6op_vstcc_QAV(  HVX_VectorPred cond, HVX_Vector *addr, HVX_Vector v )
{
#ifdef __hexagon__
#ifdef Q6_vmaskedstoreq_QAV
    Q6_vmaskedstoreq_QAV( cond, addr, v );
#else 
  __asm__ __volatile__( "if(%0) vmem(%1)=%2;" : : "q"(cond),"r"(addr),"v"(v) : "memory");
#endif
#else
    fake_conditional_store( &cond,addr,&v);
#endif
}
static HVX_INLINE_ALWAYS void q6op_vstcc_QnAV(  HVX_VectorPred cond, HVX_Vector *addr, HVX_Vector v )
{
#ifdef __hexagon__
#ifdef Q6_vmaskedstorenq_QAV
    Q6_vmaskedstorenq_QAV( cond, addr, v );
#else 
  __asm__ __volatile__( "if(!%0) vmem(%1)=%2;" : : "q"(cond),"r"(addr),"v"(v) : "memory");
#endif
#else
    fake_Nconditional_store( &cond,addr,&v);
#endif
}
#else
static HVX_INLINE_ALWAYS void q6op_vstcc_QAV(  HVX_VectorPred cond, HVX_Vector *addr, HVX_Vector v )
{
    Q6_vmaskedstoreq_QAV( cond, addr, v );
}
static HVX_INLINE_ALWAYS void q6op_vstcc_QnAV(  HVX_VectorPred cond, HVX_Vector *addr, HVX_Vector v )
{
    Q6_vmaskedstorenq_QAV( cond, addr, v );
}

#endif
static HVX_INLINE_ALWAYS void q6op_vstcc_QAV_nt(  HVX_VectorPred cond, HVX_Vector *addr, HVX_Vector v )
{
#ifdef __hexagon__
#ifdef Q6_vmaskedstorentq_QAV
    Q6_vmaskedstorentq_QAV( cond, addr, v );
#else 
  __asm__ __volatile__( "if(%0) vmem(%1):nt=%2;" : : "q"(cond),"r"(addr),"v"(v) : "memory");
#endif
#else
    fake_conditional_store( &cond,addr,&v);
#endif
}
static HVX_INLINE_ALWAYS void q6op_vstcc_QnAV_nt(  HVX_VectorPred cond, HVX_Vector *addr, HVX_Vector v )
{
#ifdef __hexagon__
#ifdef Q6_vmaskedstorentnq_QAV
    Q6_vmaskedstorentnq_QAV( cond, addr, v );
#else 
  __asm__ __volatile__( "if(!%0) vmem(%1):nt=%2;" : : "q"(cond),"r"(addr),"v"(v) : "memory");
#endif
#else
    fake_Nconditional_store( &cond,addr,&v);
#endif
}


//
// wrapper for l2fetch; these do nothing
// on 'libnative'
//
static __inline void q6op_L2fetch_AP(void const * addr, unsigned long long param)
{
#ifdef __hexagon__
    unsigned int addru = (unsigned int)addr;
    __asm__ __volatile__ ("l2fetch(%0,%1)" : : "r"(addru), "r"(param));
#endif
}
static __inline void q6op_L2fetch_AR(void const *  addr, unsigned int param)
{
#ifdef __hexagon__
    unsigned int addru = (unsigned int)addr;
    __asm__ __volatile__ ("l2fetch(%0,%1)" : : "r"(addru), "r"(param));
#endif
}

//
// acquire/release HVX
// Only declared if SIM_ACQUIRE_HVX is defined
// (need to include <hexagon_standalone.h>)
// or if __hexagon__ not defined.
//
#if defined( SIM_ACQUIRE_HVX) || !defined(__hexagon__)
static HVX_INLINE_ALWAYS void acquire_HVX() { 
#if defined(__hexagon__)
    SIM_ACQUIRE_HVX;
#ifdef __HVXDBL__
    SIM_SET_HVX_DOUBLE_MODE ;
#endif
#endif
}
static __inline void release_HVX() { 
#if defined(__hexagon__)
	// delay so writes finish
	for( int i= 0; i < 12;  i++ ){
		asm volatile ("nop\n  nop\n nop");
	}
    SIM_RELEASE_HVX;
#endif
}

#endif


#ifdef __cplusplus
} //extern "C"
#endif

#ifdef __cplusplus

#ifdef __HVXDBL__
#define hvxUtil hvxUtil128
#else
#define hvxUtil hvxUtil64
#endif

namespace hvxUtil {
    static const int NVEC = sizeof(HVX_Vector);
}


//
// acquire/release objects.
// This is used as
//       { 
//          HvxAcquireToken t;
//          call_some_function();
//       }
//
#if defined(SIM_ACQUIRE_HVX) || !defined(__hexagon__)
template<int W>
class HvxAcquireToken_templ {
   public:
    inline HvxAcquireToken_templ(){ acquire_HVX(); }
    inline ~ HvxAcquireToken_templ(){ release_HVX(); }
} __attribute__((unused));
typedef HvxAcquireToken_templ<sizeof(HVX_Vector)> HvxAcquireToken;
#endif

//  q6op_V_valign_VV<N>(v1,v0) is equivalent to Q6_V_valign_VVR(v1,v0,N)
// .. but will special case to q6op_V_valign_VVI or q6op_V_vlalign_VVI 
//  where possible; and bypass when N is a multiple of vector width.
// 
template <int R>
static inline HVX_Vector 
q6op_V_valign_VV(HVX_Vector v1, HVX_Vector v0  )
{
    static const int NV = sizeof(HVX_Vector);
    static const int RM = (unsigned)R&(NV-1);
    if( RM ==0){
        return v0;
    }else if(RM <= 7 ){
        return Q6_V_valign_VVI( v1, v0 , RM&7);
    }else if(RM>= NV-7 ){
        return Q6_V_vlalign_VVI( v1, v0, (NV-RM)&7);
    }else{
        return Q6_V_valign_VVR( v1, v0 , R);
    }

} 
// these are intended for debug only (but should work on actual
// target, just not very fast...)
//  e.g. vextract<int16_t>( vec, n )
//  gives the nth int16 from the vector.
//
template <typename ET>
static inline ET vextract( HVX_Vector const & v, int idx ){
    static const unsigned NV = sizeof (HVX_Vector)/sizeof(ET);
    union { HVX_Vector as_v; ET as_t[NV ]; } u;
    u.as_v = v; return u.as_t[ (unsigned)idx % NV ];
}
template <typename ET>
static inline ET vextract( HVX_VectorPair const & v, int idx ){
    static const unsigned NV = sizeof (HVX_VectorPair)/sizeof(ET);
    union { HVX_VectorPair as_v; ET as_t[NV ]; } u;
    u.as_v = v; return u.as_t[ (unsigned)idx % NV ];
}
//
// extract from condition requires a special op because of the way
// libnative is implemented
//
template <typename ET>
static inline ET vextractQ( HVX_VectorPred const & q, int idx ){
    return vextract<ET>( Q6_V_vand_QR( q, 0x01010101), idx);
}


// vextractp assumes that elements are dealt across
// two vectors in a pair, and interprets the index
// accordingly:
//               0 -> element 0 of first half
//               1 - > element 0 of second half 
//               2 -> element 1 of first half
template <typename ET>
static inline ET vextractp( HVX_VectorPair const & v, int idx ){
    static const unsigned NV = sizeof (HVX_VectorPair)/sizeof(ET);
    unsigned idx2 = ((unsigned)idx>>1)%(NV/2) + ((idx&1)?(NV/2):0);
    union { HVX_VectorPair as_v; ET as_t[NV ]; } u;
    u.as_v = v; return u.as_t[ idx2 ];
}


// this function returns a vector containing bytes 0..NVEC-1

namespace hvxUtil {
    static inline  HVX_Vector q6op_Vb_vindices() {
#if __HEXAGON_ARCH__ >= 65
    	return Q6_Vb_prefixsum_Q( Q6_Q_not_Q(Q6_Q_vsetq_R(1)));
#else
    	static const HVX_Vector_union_u8 indices = {{
		     0,   1,   2,   3,    4,   5,   6,   7,    8,   9,  10,  11,   12,  13,  14,  15,
		    16,  17,  18,  19,   20,  21,  22,  23,   24,  25,  26,  27,   28,  29,  30,  31,
		    32,  33,  34,  35,   36,  37,  38,  39,   40,  41,  42,  43,   44,  45,  46,  47,
		    48,  49,  50,  51,   52,  53,  54,  55,   56,  57,  58,  59,   60,  61,  62,  63
#ifdef __HVXDBL__
		  , 64,  65,  66,  67,   68,  69,  70,  71,   72,  73,  74,  75,   76,  77,  78,  79,
		    80,  81,  82,  83,   84,  85,  86,  87,   88,  89,  90,  91,   92,  93,  94,  95,
		    96,  97,  98,  99,  100, 101, 102, 103,  104, 105, 106, 107,  108, 109, 110, 111,
		   112, 113, 114, 115,  116, 117, 118, 119,  120, 121, 122, 123,  124, 125, 126, 127
#endif
	    }};
	    return indices.as_v;
#endif
    }   // end func
} // end namespace
using hvxUtil::q6op_Vb_vindices;


#endif   //C++



#endif





