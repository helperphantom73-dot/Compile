#!/usr/bin/env python3
"""Axon v11.0 Universal Cross-OS & Enterprise Auto-Deploy Upgrader.
Place alongside axon_bundle.py and run:
    python axon_v11_upgrade.py
"""
import os
import re
import subprocess
import sys
import textwrap


def unpack_v10_if_needed():
    if os.path.isfile("axon/cli.py") and os.path.isfile("axon/runtime/axon_runtime.c"):
        return
    bundle = next((f for f in ("axon_bundle.py", "bundle.py") if os.path.isfile(f)), None)
    if not bundle:
        raise FileNotFoundError("axon_bundle.py not found alongside axon_v11_upgrade.py")
    print(f"[v11] Unpacking base from {bundle} ...")
    src = open(bundle, encoding="utf-8").read().split("def main():")[0]
    ns = {}
    exec(src, ns)
    for leg in ("axon/backend.py", "tests/test_axon_suite.py"):
        if os.path.isfile(leg):
            os.remove(leg)
    for path, content in ns["FILES"].items():
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        if path.endswith("Makefile"):
            content = content.replace("@@", "\t")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)


def write_file(path, content):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)


def patch_file(path, fn):
    old = open(path, encoding="utf-8").read()
    new = fn(old)
    if new != old:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(new)


V11_RUNTIME_H = r'''#ifndef AXON_RUNTIME_H
#define AXON_RUNTIME_H
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#ifdef __cplusplus
extern "C" {
#endif
#define AXON_MAX_DIMS 4
#define AXON_DTYPE_F32 0
#define AXON_DTYPE_FP4 2
#define AXON_DTYPE_TERNARY 4
#define AXON_OK 0
#define AXON_ERR_ARG (-1)
#define AXON_ERR_SHAPE (-2)
#define AXON_ERR_DTYPE (-3)
#define AXON_ERR_NOMEM (-4)
#define AXON_ERR_FORMAT (-5)
typedef struct AxonTensor {
    void *data; int64_t ndim; int64_t shape[AXON_MAX_DIMS]; int64_t strides[AXON_MAX_DIMS];
    int32_t dtype; int32_t reserved; float scale; bool owns_data; bool in_arena;
} AxonTensor;
typedef struct AxonBrainStats {
    int64_t active_regions;
    int64_t total_regions;
    int64_t paged_regions;
    int64_t unique_regions;
    int64_t paged_bytes_active;
    int64_t paged_bytes_total;
    int64_t dense_f32_bytes;
    int64_t executed_flops;
    int64_t dense_flops;
} AxonBrainStats;
typedef struct AxonKVCache {
    float *k_cache;
    float *v_cache;
    int64_t max_steps;
    int64_t dim;
    int64_t len;
    bool in_arena;
} AxonKVCache;

void axon_arena_init(size_t cap); void axon_arena_reset(void); void axon_arena_destroy(void);
bool axon_arena_active(void); size_t axon_arena_used(void); size_t axon_arena_capacity(void);
size_t axon_arena_mark(void); void axon_arena_rewind(size_t mark);
AxonTensor *axon_tensor_alloc(int64_t ndim, const int64_t *shape, int32_t dtype);
AxonTensor *axon_tensor_alloc_persistent(int64_t ndim, const int64_t *shape, int32_t dtype);
void axon_tensor_free(AxonTensor *t); int64_t axon_tensor_numel(const AxonTensor *t);
int axon_tensor_fill_f32(AxonTensor *t, const float *v, int64_t count);
int axon_tensor_to_f32(const AxonTensor *t, float *dst, int64_t count);
void axon_print_tensor(const AxonTensor *t); void axon_print_tensor_n(const AxonTensor *t, int64_t lim);
int64_t axon_live_allocs(void); void axon_set_weights_blob(const void *blob, int64_t size);
int axon_load_weights_file(const char *path); void axon_unload_weights_file(void);
AxonTensor *axon_weight_tensor(const char *name); void axon_weights_get(const void **p, int64_t *n);
void axon_rt_lock(void); void axon_rt_unlock(void);
void axon_brain_stats_reset(void);
void axon_brain_record_stats(int64_t active_reg, int64_t total_reg, int64_t paged_reg, int64_t unique_reg,
                             int64_t bytes_active, int64_t bytes_total, int64_t f32_bytes,
                             int64_t exec_flops, int64_t dense_flops);
void axon_get_brain_stats(AxonBrainStats *out);
AxonKVCache *axon_kv_cache_alloc(int64_t max_steps, int64_t dim);
void axon_kv_cache_free(AxonKVCache *kv);
int axon_kv_cache_append(AxonKVCache *kv, const float *k_step, const float *v_step);
int axon_kv_cache_attend(const AxonKVCache *kv, const float *q_step, float *out_step);

int axon_matmul(const AxonTensor *A, const AxonTensor *B, AxonTensor *C);
int axon_neuro_matmul(const AxonTensor *X, const AxonTensor *W, const AxonTensor *S, AxonTensor *C);
int axon_neuro_matmul_paged(const AxonTensor *X, const AxonTensor *W, const AxonTensor *S, AxonTensor *C, int32_t keep, int32_t region_blocked);
int axon_add(const AxonTensor *A, const AxonTensor *B, AxonTensor *C);
int axon_sub(const AxonTensor *A, const AxonTensor *B, AxonTensor *C);
int axon_mul(const AxonTensor *A, const AxonTensor *B, AxonTensor *C);
int axon_relu(const AxonTensor *A, AxonTensor *C); int axon_gelu(const AxonTensor *A, AxonTensor *C);
int axon_silu(const AxonTensor *A, AxonTensor *C); int axon_sigmoid(const AxonTensor *A, AxonTensor *C);
int axon_softmax(const AxonTensor *A, AxonTensor *C); int axon_transpose(const AxonTensor *A, AxonTensor *C);
int axon_layer_norm(const AxonTensor *X, const AxonTensor *G, const AxonTensor *B, AxonTensor *C);
int axon_rms_norm(const AxonTensor *X, const AxonTensor *G, AxonTensor *C);
int axon_rope(const AxonTensor *X, AxonTensor *C);
int axon_attn_causal(const AxonTensor *Q, const AxonTensor *K, const AxonTensor *V, AxonTensor *C);
int axon_equilibrium(const AxonTensor *X, const AxonTensor *W, const AxonTensor *B, AxonTensor *C);
int axon_mm_relu(const AxonTensor *A, const AxonTensor *B, AxonTensor *C);
int axon_mm_gelu(const AxonTensor *A, const AxonTensor *B, AxonTensor *C);
int axon_mm_silu(const AxonTensor *A, const AxonTensor *B, AxonTensor *C);
int axon_mm_bias(const AxonTensor *A, const AxonTensor *B, const AxonTensor *bias, AxonTensor *C);
int axon_mm_bias_relu(const AxonTensor *A, const AxonTensor *B, const AxonTensor *bias, AxonTensor *C);
int axon_mm_bias_gelu(const AxonTensor *A, const AxonTensor *B, const AxonTensor *bias, AxonTensor *C);
int axon_mm_bias_silu(const AxonTensor *A, const AxonTensor *B, const AxonTensor *bias, AxonTensor *C);
int axon_bench_matmul(int n, double *out_ms, float *out_c); void axon_lsh_benchmark(int dim);
int axon_init(void); void axon_shutdown(void); AxonTensor *axon_forward(AxonTensor **inputs);
int64_t axon_get_input_ndim(void); void axon_get_input_shape(int64_t *out);
int64_t axon_get_input_numel(void); int64_t axon_get_output_numel(void);
int64_t axon_get_total_flops(void); int64_t axon_output_is_temp(void);
#ifdef __cplusplus
}
#endif
#endif
'''


V11_RUNTIME_C = r'''#ifndef _WIN32
#  define _POSIX_C_SOURCE 200809L
#  define _DEFAULT_SOURCE
#endif
#include "axon_runtime.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  include <malloc.h>
#else
#  include <fcntl.h>
#  include <pthread.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

#ifndef _MSC_VER
#  include <stdatomic.h>
#endif

#if defined(__has_include)
#  if __has_include("axon_tuning.h")
#    include "axon_tuning.h"
#  endif
#endif
#ifndef AXON_TILE
#define AXON_TILE 64
#endif
#if AXON_TILE < 1
#error "AXON_TILE must be >= 1"
#endif
#if AXON_TILE > 64
#define AXON_BT 64
#else
#define AXON_BT AXON_TILE
#endif
#if AXON_BT < 1
#error "AXON_BT must be >= 1"
#endif
#ifndef AXON_PAR_MIN
#define AXON_PAR_MIN 65536
#endif
#define AXON_ALIGN 64
#define AXON_PAGING_MIN_BYTES ((int64_t)4 * 1024 * 1024)
#define SQRT_2_OVER_PI 0.7978845608028654f
#define GELU_C 0.044715f
#define LN_EPS 1e-5f
#define REQ(c,e) do{ if(!(c)) return (e); }while(0)

#ifdef _MSC_VER
#  define AXON_TLS __declspec(thread)
#  define AXON_STATIC_ASSERT(cond, msg) typedef char axon_sa_##__LINE__[(cond) ? 1 : -1]
#else
#  define AXON_TLS _Thread_local
#  define AXON_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#endif

static AXON_TLS uint8_t *g_base = NULL;
static AXON_TLS size_t g_cap = 0, g_off = 0;
static AXON_TLS AxonBrainStats g_brain_stats = {0};

#ifdef _MSC_VER
static volatile LONG64 g_live = 0;
static void live_inc(void){ InterlockedIncrement64(&g_live); }
static void live_dec(void){ InterlockedDecrement64(&g_live); }
int64_t axon_live_allocs(void){ return (int64_t)InterlockedCompareExchange64(&g_live, 0, 0); }
#else
static atomic_int_fast64_t g_live = 0;
static void live_inc(void){ atomic_fetch_add_explicit(&g_live, 1, memory_order_relaxed); }
static void live_dec(void){ atomic_fetch_sub_explicit(&g_live, 1, memory_order_relaxed); }
int64_t axon_live_allocs(void){ return (int64_t)atomic_load_explicit(&g_live, memory_order_relaxed); }
#endif

static void *axon_aligned_alloc(size_t align, size_t size){
#ifdef _WIN32
    return _aligned_malloc(size, align);
#else
    void *p = NULL;
    if(posix_memalign(&p, align, size) != 0) return NULL;
    return p;
#endif
}
static void axon_aligned_free(void *p){
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}
static double axon_now_ms(void){
#ifdef _WIN32
    static LARGE_INTEGER freq;
    static int init = 0;
    if(!init){ QueryPerformanceFrequency(&freq); init = 1; }
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return ((double)now.QuadPart * 1000.0) / (double)freq.QuadPart;
#else
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e3 + (double)t.tv_nsec / 1e6;
#endif
}

void axon_arena_init(size_t cap){ if(g_base||cap==0)return; void*p=axon_aligned_alloc(AXON_ALIGN,cap); if(p){g_base=(uint8_t*)p;g_cap=cap;g_off=0;} }
static void* arena_alloc(size_t n){ if(!g_base)return NULL; size_t a=(n+(AXON_ALIGN-1))&~((size_t)(AXON_ALIGN-1)); if(a<n||g_off>g_cap||a>g_cap-g_off)return NULL; void*p=g_base+g_off; g_off+=a; return p; }
void axon_arena_reset(void){g_off=0;}
void axon_arena_destroy(void){axon_aligned_free(g_base);g_base=NULL;g_cap=0;g_off=0;}
bool axon_arena_active(void){return g_base!=NULL;}
size_t axon_arena_used(void){return g_off;}
size_t axon_arena_capacity(void){return g_cap;}
size_t axon_arena_mark(void){return g_off;}
void axon_arena_rewind(size_t mark){ if(mark<=g_cap) g_off=mark; }

void axon_brain_stats_reset(void){ memset(&g_brain_stats, 0, sizeof(g_brain_stats)); }
void axon_brain_record_stats(int64_t active_reg, int64_t total_reg, int64_t paged_reg, int64_t unique_reg,
                             int64_t bytes_active, int64_t bytes_total, int64_t f32_bytes,
                             int64_t exec_flops, int64_t dense_flops){
    g_brain_stats.active_regions += active_reg;
    g_brain_stats.total_regions += total_reg;
    g_brain_stats.paged_regions += paged_reg;
    g_brain_stats.unique_regions += unique_reg;
    g_brain_stats.paged_bytes_active += bytes_active;
    g_brain_stats.paged_bytes_total += bytes_total;
    g_brain_stats.dense_f32_bytes += f32_bytes;
    g_brain_stats.executed_flops += exec_flops;
    g_brain_stats.dense_flops += dense_flops;
}
void axon_get_brain_stats(AxonBrainStats *out){ if(out) *out = g_brain_stats; }

AxonKVCache *axon_kv_cache_alloc(int64_t max_steps, int64_t dim){
    if(max_steps<=0 || dim<=0 || max_steps > INT64_MAX / dim) return NULL;
    size_t vec_bytes = (size_t)max_steps * (size_t)dim * sizeof(float);
    if(axon_arena_active()){
        size_t mark = axon_arena_mark();
        AxonKVCache *kv = (AxonKVCache*)arena_alloc(sizeof(AxonKVCache));
        float *kc = (float*)arena_alloc(vec_bytes);
        float *vc = (float*)arena_alloc(vec_bytes);
        if(kv && kc && vc){
            memset(kc, 0, vec_bytes); memset(vc, 0, vec_bytes);
            kv->k_cache=kc; kv->v_cache=vc; kv->max_steps=max_steps; kv->dim=dim; kv->len=0; kv->in_arena=true;
            return kv;
        }
        axon_arena_rewind(mark);
    }
    AxonKVCache *kv = (AxonKVCache*)malloc(sizeof(AxonKVCache));
    if(!kv) return NULL;
    float *kc = (float*)calloc((size_t)(max_steps*dim), sizeof(float));
    float *vc = (float*)calloc((size_t)(max_steps*dim), sizeof(float));
    if(!kc || !vc){ free(kc); free(vc); free(kv); return NULL; }
    live_inc();
    kv->k_cache=kc; kv->v_cache=vc; kv->max_steps=max_steps; kv->dim=dim; kv->len=0; kv->in_arena=false;
    return kv;
}
void axon_kv_cache_free(AxonKVCache *kv){
    if(!kv) return;
    if(!kv->in_arena){
        free(kv->k_cache); free(kv->v_cache); free(kv); live_dec();
    }
}
int axon_kv_cache_append(AxonKVCache *kv, const float *k_step, const float *v_step){
    REQ(kv && k_step && v_step && kv->k_cache && kv->v_cache, AXON_ERR_ARG);
    if(kv->len >= kv->max_steps){
        memmove(kv->k_cache, kv->k_cache + kv->dim, (size_t)((kv->max_steps - 1) * kv->dim) * sizeof(float));
        memmove(kv->v_cache, kv->v_cache + kv->dim, (size_t)((kv->max_steps - 1) * kv->dim) * sizeof(float));
        kv->len = kv->max_steps - 1;
    }
    memcpy(kv->k_cache + kv->len * kv->dim, k_step, (size_t)kv->dim * sizeof(float));
    memcpy(kv->v_cache + kv->len * kv->dim, v_step, (size_t)kv->dim * sizeof(float));
    kv->len += 1;
    return AXON_OK;
}
int axon_kv_cache_attend(const AxonKVCache *kv, const float *q_step, float *out_step){
    REQ(kv && q_step && out_step && kv->k_cache && kv->v_cache && kv->dim > 0, AXON_ERR_ARG);
    const int64_t D = kv->dim, L = kv->len;
    for(int64_t d=0; d<D; ++d) out_step[d]=0.0f;
    if(L<=0) return AXON_OK;
    const float inv = 1.0f / sqrtf((float)D);
    float m = -INFINITY, l = 0.0f;
    for(int64_t j=0; j<L; ++j){
        const float *kj = kv->k_cache + j * D;
        const float *vj = kv->v_cache + j * D;
        float dot = 0.0f;
        for(int64_t d=0; d<D; ++d) dot += q_step[d] * kj[d];
        float sc = dot * inv;
        float mn = (sc > m) ? sc : m;
        float alpha = (m == -INFINITY) ? 0.0f : expf(m - mn);
        float beta = expf(sc - mn);
        for(int64_t d=0; d<D; ++d) out_step[d] = out_step[d] * alpha + beta * vj[d];
        l = l * alpha + beta;
        m = mn;
    }
    float il = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for(int64_t d=0; d<D; ++d) out_step[d] *= il;
    return AXON_OK;
}

static size_t elemsz(int32_t dt,int64_t total){ if(dt==AXON_DTYPE_TERNARY)return (size_t)((total+3)/4); if(dt==AXON_DTYPE_FP4)return (size_t)((total+1)/2); return (size_t)total*sizeof(float); }
static AxonTensor* tensor_new(int64_t ndim,const int64_t*shape,int32_t dtype,int persistent){
    if(ndim<1||ndim>AXON_MAX_DIMS||!shape)return NULL;
    if(dtype!=AXON_DTYPE_F32&&dtype!=AXON_DTYPE_TERNARY&&dtype!=AXON_DTYPE_FP4)return NULL;
    int64_t total=1;
    for(int64_t i=0;i<ndim;++i){ if(shape[i]<=0||shape[i]>INT64_MAX/total)return NULL; total*=shape[i]; }
    if((uint64_t)total>(uint64_t)(SIZE_MAX/sizeof(float)))return NULL;
    size_t bytes=elemsz(dtype,total); if(bytes<AXON_ALIGN)bytes=AXON_ALIGN;
    int use_arena=(!persistent&&axon_arena_active());
    AxonTensor*t=use_arena?(AxonTensor*)arena_alloc(sizeof(AxonTensor)):NULL;
    if(!t){ t=(AxonTensor*)malloc(sizeof(AxonTensor)); if(!t)return NULL; use_arena=0; live_inc(); }
    t->ndim=ndim;t->dtype=dtype;t->reserved=0;t->scale=1.0f;t->in_arena=(bool)use_arena;t->owns_data=false;t->data=NULL;
    memset(t->shape,0,sizeof(t->shape));memset(t->strides,0,sizeof(t->strides));
    for(int64_t i=0;i<ndim;++i)t->shape[i]=shape[i];
    int64_t s=1; for(int64_t i=ndim-1;i>=0;--i){t->strides[i]=s;s*=t->shape[i];}
    if(use_arena){ t->data=arena_alloc(bytes); if(t->data)return t; }
    void*d=axon_aligned_alloc(AXON_ALIGN,bytes);
    if(!d){ if(!t->in_arena){free(t);live_dec();} return NULL; }
    t->data=d;t->owns_data=true; if(dtype==AXON_DTYPE_F32)memset(d,0,bytes); return t;
}
static AxonTensor* tensor_view_new(int64_t ndim,const int64_t*shape,int32_t dtype,const void*ptr){
    if(ndim<1||ndim>AXON_MAX_DIMS||!shape||!ptr)return NULL;
    int64_t total=1; for(int64_t i=0;i<ndim;++i){ if(shape[i]<=0||shape[i]>INT64_MAX/total)return NULL; total*=shape[i]; }
    AxonTensor*t=(AxonTensor*)malloc(sizeof(AxonTensor)); if(!t)return NULL; live_inc();
    t->ndim=ndim;t->dtype=dtype;t->reserved=0;t->scale=1.0f;t->in_arena=false;t->owns_data=false;t->data=(void*)ptr;
    memset(t->shape,0,sizeof(t->shape));memset(t->strides,0,sizeof(t->strides));
    for(int64_t i=0;i<ndim;++i)t->shape[i]=shape[i];
    int64_t s=1; for(int64_t i=ndim-1;i>=0;--i){t->strides[i]=s;s*=t->shape[i];}
    return t;
}
AxonTensor* axon_tensor_alloc(int64_t ndim,const int64_t*shape,int32_t dtype){return tensor_new(ndim,shape,dtype,0);}
AxonTensor* axon_tensor_alloc_persistent(int64_t ndim,const int64_t*shape,int32_t dtype){return tensor_new(ndim,shape,dtype,1);}
void axon_tensor_free(AxonTensor*t){ if(!t)return; if(t->owns_data)axon_aligned_free(t->data); if(!t->in_arena){free(t);live_dec();} }
int64_t axon_tensor_numel(const AxonTensor*t){ if(!t||t->ndim<1)return 0; int64_t n=1; for(int64_t i=0;i<t->ndim;++i)n*=t->shape[i]; return n; }
int axon_tensor_fill_f32(AxonTensor*t,const float*v,int64_t count){ REQ(t&&t->data&&v,AXON_ERR_ARG); REQ(t->dtype==AXON_DTYPE_F32,AXON_ERR_DTYPE); int64_t n=axon_tensor_numel(t); int64_t lim=count<n?count:n; memcpy(t->data,v,(size_t)lim*sizeof(float)); return AXON_OK; }

static const float FP4_LUT[16]={0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f,-0.f,-.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};
int axon_tensor_to_f32(const AxonTensor*t,float*dst,int64_t count){
    REQ(t&&t->data&&dst&&count>=0,AXON_ERR_ARG);
    int64_t total=axon_tensor_numel(t);
    int64_t lim=(count<total)?count:total;
    if(t->dtype==AXON_DTYPE_F32){
        memcpy(dst,t->data,(size_t)lim*sizeof(float));
        return AXON_OK;
    }
    if(t->dtype==AXON_DTYPE_TERNARY){
        const uint8_t*w=(const uint8_t*)t->data;
        for(int64_t i=0;i<lim;++i){
            uint8_t c=(uint8_t)((w[i>>2]>>((i&3)*2))&3);
            dst[i]=(c==1)?t->scale:((c==2)?-t->scale:0.0f);
        }
        return AXON_OK;
    }
    if(t->dtype==AXON_DTYPE_FP4){
        const uint8_t*p=(const uint8_t*)t->data; uint32_t nb=0,nr=0; memcpy(&nb,p,4); memcpy(&nr,p+4,4);
        const uint8_t*scales=p+8,*packed=scales+nb,*res=packed+(size_t)nb*16u;
        for(int64_t i=0;i<lim;++i){
            uint32_t blk=(uint32_t)(i>>5); int off=(int)(i&31); uint8_t byte=packed[(size_t)blk*16u+(size_t)(off>>1)];
            dst[i]=FP4_LUT[(off&1)?((byte>>4)&0x0F):(byte&0x0F)]*ldexpf(1.0f,(int)scales[blk]-127);
        }
        for(uint32_t r=0;r<nr;++r){
            uint32_t idx; float diff; memcpy(&idx,res+(size_t)r*8u,4); memcpy(&diff,res+(size_t)r*8u+4,4);
            if((int64_t)idx<lim) dst[idx]+=diff;
        }
        return AXON_OK;
    }
    return AXON_ERR_DTYPE;
}
void axon_print_tensor_n(const AxonTensor*t,int64_t max_lim){
    if(!t){printf("Tensor: NULL\n");return;}
    printf("Tensor(ndim=%lld, shape=[",(long long)t->ndim);
    for(int64_t i=0;i<t->ndim;++i)printf("%lld%s",(long long)t->shape[i],(i+1<t->ndim)?", ":"");
    printf("], dtype=%d) = [\n",(int)t->dtype);
    int64_t total=axon_tensor_numel(t), lim=(max_lim>0&&max_lim<total)?max_lim:total;
    if(t->dtype==AXON_DTYPE_TERNARY){
        const uint8_t*w=(const uint8_t*)t->data;
        for(int64_t i=0;i<lim;++i){uint8_t c=(uint8_t)((w[i>>2]>>((i&3)*2))&3); printf("  %.9g\n",c==1?t->scale:(c==2?-t->scale:0.0f));}
    } else if(t->dtype==AXON_DTYPE_FP4){
        const uint8_t*p=(const uint8_t*)t->data; uint32_t nb=0,nr=0; memcpy(&nb,p,4); memcpy(&nr,p+4,4);
        const uint8_t*scales=p+8; const uint8_t*packed=scales+nb; const uint8_t*res=packed+(size_t)nb*16u;
        for(int64_t i=0;i<lim;++i){
            uint32_t blk=(uint32_t)(i>>5); int off=(int)(i&31); uint8_t byte=packed[(size_t)blk*16u+(size_t)(off>>1)];
            float v=FP4_LUT[(off&1)?((byte>>4)&0x0F):(byte&0x0F)]*ldexpf(1.0f,(int)scales[blk]-127);
            for(uint32_t r=0;r<nr;++r){ uint32_t idx; float diff; memcpy(&idx,res+(size_t)r*8u,4); memcpy(&diff,res+(size_t)r*8u+4,4); if((int64_t)idx==i)v+=diff; }
            printf("  %.9g\n",(double)v);
        }
    } else { const float*d=(const float*)t->data; for(int64_t i=0;i<lim;++i)printf("  %.9g\n",(double)d[i]); }
    if(total>lim)printf("  ... (%lld more)\n",(long long)(total-lim));
    printf("]\n");
}
void axon_print_tensor(const AxonTensor*t){ axon_print_tensor_n(t,16); }

#ifdef _WIN32
static CRITICAL_SECTION g_mu;
static INIT_ONCE g_mu_once = INIT_ONCE_STATIC_INIT;
static BOOL CALLBACK mu_init_win(PINIT_ONCE o, PVOID p, PVOID *c){ (void)o;(void)p;(void)c; InitializeCriticalSection(&g_mu); return TRUE; }
void axon_rt_lock(void){ InitOnceExecuteOnce(&g_mu_once, mu_init_win, NULL, NULL); EnterCriticalSection(&g_mu); }
void axon_rt_unlock(void){ LeaveCriticalSection(&g_mu); }
#else
static pthread_mutex_t g_mu; static pthread_once_t g_mu_once=PTHREAD_ONCE_INIT;
static void mu_init(void){ pthread_mutexattr_t a; pthread_mutexattr_init(&a); pthread_mutexattr_settype(&a,PTHREAD_MUTEX_RECURSIVE); pthread_mutex_init(&g_mu,&a); pthread_mutexattr_destroy(&a); }
void axon_rt_lock(void){ pthread_once(&g_mu_once,mu_init); pthread_mutex_lock(&g_mu); }
void axon_rt_unlock(void){ pthread_mutex_unlock(&g_mu); }
#endif

#ifdef _MSC_VER
#  pragma pack(push, 1)
typedef struct { uint64_t hash,name_off,data_off,nbytes,dtype,ndim; int64_t shape[4]; } AxonEntry;
typedef struct { char magic[4]; uint32_t version,count,reserved; uint64_t total_size; } AxonHeader;
#  pragma pack(pop)
#else
typedef struct __attribute__((packed)){ uint64_t hash,name_off,data_off,nbytes,dtype,ndim; int64_t shape[4]; } AxonEntry;
typedef struct __attribute__((packed)){ char magic[4]; uint32_t version,count,reserved; uint64_t total_size; } AxonHeader;
#endif
AXON_STATIC_ASSERT(sizeof(AxonHeader) == 24, "AxonHeader must be 24 bytes");
AXON_STATIC_ASSERT(sizeof(AxonEntry) == 80, "AxonEntry must be 80 bytes");

static const uint8_t*g_blob=NULL; static uint64_t g_blob_size=0;
static void*g_mmap_ptr=NULL; static size_t g_mmap_size=0;
#ifdef _WIN32
static HANDLE g_win_file = INVALID_HANDLE_VALUE;
static HANDLE g_win_map = NULL;
typedef struct { PVOID VirtualAddress; SIZE_T NumberOfBytes; } AXON_WIN_RANGE;
typedef BOOL (WINAPI *AXON_PREFETCH_FN)(HANDLE, ULONG_PTR, AXON_WIN_RANGE*, ULONG);
#endif

static int blob_valid(void){
    if(!g_blob||g_blob_size<sizeof(AxonHeader))return 0;
    AxonHeader h; memcpy(&h,g_blob,sizeof(h));
    return memcmp(h.magic,"AXN3",4)==0&&h.version==3&&h.total_size==g_blob_size&&sizeof(AxonHeader)+(uint64_t)h.count*sizeof(AxonEntry)<=g_blob_size;
}
static int is_in_mmap(const void *ptr){
    if(!g_mmap_ptr||g_mmap_size==0||!ptr) return 0;
    uintptr_t p=(uintptr_t)ptr, base=(uintptr_t)g_mmap_ptr;
    return p>=base && p<base+g_mmap_size;
}
static void advise_range(const void *ptr, size_t len, int want_hot){
    if(!g_mmap_ptr || g_mmap_size==0 || !ptr || len==0) return;
    uintptr_t base = (uintptr_t)g_mmap_ptr;
    uintptr_t end_map = base + g_mmap_size;
    uintptr_t p = (uintptr_t)ptr;
    if(p < base || p >= end_map) return;
    uintptr_t q = p + len;
    if(q < p || q > end_map) q = end_map;
#ifdef _WIN32
    if(want_hot){
        HMODULE k32 = GetModuleHandleA("kernel32.dll");
        if(k32){
            AXON_PREFETCH_FN pfn = (AXON_PREFETCH_FN)(void*)GetProcAddress(k32, "PrefetchVirtualMemory");
            if(pfn){
                AXON_WIN_RANGE rng = { (PVOID)p, (SIZE_T)(q - p) };
                pfn(GetCurrentProcess(), 1, &rng, 0);
            }
        }
    }
#elif defined(__unix__) || defined(__APPLE__)
    long ps_l = sysconf(_SC_PAGESIZE);
    size_t ps = (ps_l > 0) ? (size_t)ps_l : 4096u;
    if(!want_hot){
#ifdef MADV_DONTNEED
        uintptr_t a = (p + ps - 1u) & ~(ps - 1u);
        uintptr_t b = q & ~(ps - 1u);
        if(b > a) madvise((void*)a, (size_t)(b - a), MADV_DONTNEED);
#endif
        return;
    }
#ifdef MADV_WILLNEED
    uintptr_t a = p & ~(ps - 1u);
    uintptr_t b = (q + ps - 1u) & ~(ps - 1u);
    if(b > end_map) b = end_map;
    if(b > a) madvise((void*)a, (size_t)(b - a), MADV_WILLNEED);
#endif
#else
    (void)want_hot;
#endif
}

void axon_set_weights_blob(const void*blob,int64_t size){ axon_rt_lock(); g_blob=(const uint8_t*)blob; g_blob_size=size>0?(uint64_t)size:0; axon_rt_unlock(); }
void axon_unload_weights_file(void){
    axon_rt_lock();
#ifdef _WIN32
    if(g_mmap_ptr){ UnmapViewOfFile(g_mmap_ptr); g_mmap_ptr=NULL; g_mmap_size=0; g_blob=NULL; g_blob_size=0; }
    if(g_win_map){ CloseHandle(g_win_map); g_win_map=NULL; }
    if(g_win_file != INVALID_HANDLE_VALUE){ CloseHandle(g_win_file); g_win_file=INVALID_HANDLE_VALUE; }
#elif defined(__unix__) || defined(__APPLE__)
    if(g_mmap_ptr){ munmap(g_mmap_ptr,g_mmap_size); g_mmap_ptr=NULL; g_mmap_size=0; g_blob=NULL; g_blob_size=0; }
#endif
    axon_rt_unlock();
}
int axon_load_weights_file(const char*path){
    if(!path) return AXON_ERR_ARG;
    axon_unload_weights_file();
#ifdef _WIN32
    HANDLE hf = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, NULL);
    if(hf == INVALID_HANDLE_VALUE) return AXON_ERR_ARG;
    LARGE_INTEGER fsz;
    if(!GetFileSizeEx(hf, &fsz) || fsz.QuadPart < (LONGLONG)sizeof(AxonHeader)){ CloseHandle(hf); return AXON_ERR_FORMAT; }
    HANDLE hm = CreateFileMappingA(hf, NULL, PAGE_READONLY, 0, 0, NULL);
    if(!hm){ CloseHandle(hf); return AXON_ERR_NOMEM; }
    void *p = MapViewOfFile(hm, FILE_MAP_READ, 0, 0, 0);
    if(!p){ CloseHandle(hm); CloseHandle(hf); return AXON_ERR_NOMEM; }
    axon_rt_lock();
    g_win_file = hf; g_win_map = hm;
    g_mmap_ptr = p; g_mmap_size = (size_t)fsz.QuadPart;
    g_blob = (const uint8_t*)p; g_blob_size = (uint64_t)fsz.QuadPart;
    int ok = blob_valid();
    axon_rt_unlock();
    if(!ok){ axon_unload_weights_file(); return AXON_ERR_FORMAT; }
    return AXON_OK;
#elif defined(__unix__) || defined(__APPLE__)
    int fd=open(path,O_RDONLY); if(fd<0)return AXON_ERR_ARG;
    struct stat st; if(fstat(fd,&st)!=0||st.st_size<(off_t)sizeof(AxonHeader)){close(fd);return AXON_ERR_FORMAT;}
    size_t sz=(size_t)st.st_size; void*p=mmap(NULL,sz,PROT_READ,MAP_PRIVATE,fd,0); close(fd);
    if(p==MAP_FAILED)return AXON_ERR_NOMEM;
#ifdef MADV_RANDOM
    madvise(p,sz,MADV_RANDOM);
#endif
    axon_rt_lock(); g_mmap_ptr=p; g_mmap_size=sz; g_blob=(const uint8_t*)p; g_blob_size=(uint64_t)sz;
    int ok=blob_valid(); axon_rt_unlock();
    if(!ok){ axon_unload_weights_file(); return AXON_ERR_FORMAT; }
    return AXON_OK;
#else
    return AXON_ERR_ARG;
#endif
}
static uint64_t fnv1a(const char*s){ uint64_t h=14695981039346656037ULL; while(*s){h^=(uint8_t)(*s++);h*=1099511628211ULL;} return h; }
static const AxonEntry* find_entry(const char*name){
    if(!blob_valid())return NULL;
    AxonHeader h; memcpy(&h,g_blob,sizeof(h));
    uint64_t names_base=sizeof(AxonHeader)+(uint64_t)h.count*sizeof(AxonEntry);
    if(names_base>g_blob_size)return NULL;
    uint64_t key=fnv1a(name);
    for(uint64_t i=0;i<h.count;++i){
        AxonEntry e; memcpy(&e,g_blob+sizeof(AxonHeader)+i*sizeof(AxonEntry),sizeof(e));
        if(e.hash!=key||e.name_off>=g_blob_size-names_base)continue;
        const char*nm=(const char*)(g_blob+names_base+e.name_off); size_t maxlen=(size_t)(g_blob_size-(names_base+e.name_off));
        if(memchr(nm,0,maxlen)==NULL||strcmp(nm,name)!=0)continue;
        if(e.data_off>g_blob_size||e.nbytes>g_blob_size-e.data_off||e.ndim<1||e.ndim>AXON_MAX_DIMS)continue;
        return (const AxonEntry*)(g_blob+sizeof(AxonHeader)+i*sizeof(AxonEntry));
    }
    return NULL;
}
AxonTensor* axon_weight_tensor(const char*name){
    if(!name)return NULL; const AxonEntry*e=find_entry(name); if(!e)return NULL;
    AxonEntry ent; memcpy(&ent,e,sizeof(ent)); const uint8_t*payload=g_blob+ent.data_off;
    int64_t shp[AXON_MAX_DIMS]={0}; memcpy(shp,ent.shape,sizeof(shp)); int64_t nd=(int64_t)ent.ndim;
    int64_t total=1; for(int64_t i=0;i<nd;++i){ if(shp[i]<=0)return NULL; total*=shp[i]; }
    if(is_in_mmap(payload) && (strncmp(name,"sig_",4)==0 || nd==1 || ent.nbytes<=4096u)){
        advise_range(payload,(size_t)ent.nbytes,1);
    }
    if(ent.dtype==AXON_DTYPE_F32){
        if(ent.nbytes!=(uint64_t)total*sizeof(float))return NULL;
        return tensor_view_new(nd,shp,AXON_DTYPE_F32,payload);
    }
    if(ent.dtype==AXON_DTYPE_TERNARY){
        if(ent.nbytes<8)return NULL; float scale; uint32_t n; memcpy(&scale,payload,4); memcpy(&n,payload+4,4);
        if((int64_t)n!=total||ent.nbytes<8+(uint64_t)((total+3)/4))return NULL;
        AxonTensor*t=tensor_view_new(nd,shp,AXON_DTYPE_TERNARY,payload+8);
        if(t)t->scale=scale; return t;
    }
    if(ent.dtype==AXON_DTYPE_FP4){
        if(ent.nbytes<8)return NULL; uint32_t nb,nr; memcpy(&nb,payload,4); memcpy(&nr,payload+4,4);
        uint64_t need=8ull+nb+16ull*nb+8ull*nr; if(ent.nbytes<need||(int64_t)nb*32<total)return NULL;
        AxonTensor*t=tensor_view_new(nd,shp,AXON_DTYPE_FP4,payload); if(!t)return NULL;
        const uint8_t*res=payload+8ull+nb+16ull*nb;
        int sorted_col=(nd==2)?1:0; int sorted_flat=1;
        if(nr>0){
            int64_t N=(nd==2)?shp[1]:1; int64_t prev_rj=-1; uint32_t prev_idx=0;
            for(uint32_t r=0;r<nr;++r){
                uint32_t idx; memcpy(&idx,res+(size_t)r*8u,4);
                if((int64_t)idx>=total){sorted_col=0;sorted_flat=0;break;}
                if(r>0 && idx<prev_idx) sorted_flat=0;
                prev_idx=idx;
                if(sorted_col){ int64_t rj=(int64_t)idx%N; if(rj<prev_rj) sorted_col=0; prev_rj=rj; }
            }
        }
        t->reserved = (sorted_col ? 1 : 0) | (sorted_flat ? 2 : 0);
        return t;
    }
    return NULL;
}

static inline float sigmoid_f(float v){ if(v>=0.f)return 1.f/(1.f+expf(-v)); float e=expf(v); return e/(1.f+e); }
static inline float act_apply(float v,int act){
    if(act==1)return v>0.f?v:0.f;
    if(act==2)return 0.5f*v*(1.f+tanhf(SQRT_2_OVER_PI*(v+GELU_C*v*v*v)));
    if(act==3)return v*sigmoid_f(v);
    if(act==4)return sigmoid_f(v);
    return v;
}
static uint32_t fp4_res_lb(const uint8_t*res,uint32_t nr,int64_t N,int64_t jj){
    uint32_t lo=0,hi=nr;
    while(lo<hi){ uint32_t mid=lo+(hi-lo)/2, midx=0; memcpy(&midx,res+(size_t)mid*8u,4); if(((int64_t)midx%N)<jj)lo=mid+1; else hi=mid; }
    return lo;
}
static uint32_t fp4_res_lb_flat(const uint8_t*res,uint32_t nr,int64_t start_idx){
    uint32_t lo=0,hi=nr;
    while(lo<hi){ uint32_t mid=lo+(hi-lo)/2, midx=0; memcpy(&midx,res+(size_t)mid*8u,4); if((int64_t)midx<start_idx)lo=mid+1; else hi=mid; }
    return lo;
}
static int gemm(const AxonTensor*A,const AxonTensor*B,const AxonTensor*bias,int act,AxonTensor*C){
    REQ(A&&B&&C&&A->data&&B->data&&C->data,AXON_ERR_ARG);
    REQ(A->ndim==2&&B->ndim==2&&C->ndim==2,AXON_ERR_SHAPE);
    REQ(A->dtype==AXON_DTYPE_F32&&C->dtype==AXON_DTYPE_F32,AXON_ERR_DTYPE);
    REQ(B->dtype==AXON_DTYPE_F32||B->dtype==AXON_DTYPE_TERNARY||B->dtype==AXON_DTYPE_FP4,AXON_ERR_DTYPE);
    const int64_t M=A->shape[0],K=A->shape[1],N=B->shape[1]; REQ(B->shape[0]==K&&C->shape[0]==M&&C->shape[1]==N,AXON_ERR_SHAPE);
    const float*bv=NULL; if(bias){ REQ(bias->data&&bias->ndim==1&&bias->shape[0]==N&&bias->dtype==AXON_DTYPE_F32,AXON_ERR_SHAPE); bv=(const float*)bias->data; }
    const float*a=(const float*)A->data; float*c=(float*)C->data; const int par=(M*N*K)>=AXON_PAR_MIN;
    if(B->dtype==AXON_DTYPE_TERNARY||B->dtype==AXON_DTYPE_FP4){
        const int is_fp4=(B->dtype==AXON_DTYPE_FP4);
        const uint8_t*tw=(const uint8_t*)B->data; const float lut[4]={0.f,B->scale,-B->scale,0.f};
        uint32_t nb=0,nr=0; const uint8_t*scales=NULL,*packed=NULL,*res=NULL;
        if(is_fp4){ memcpy(&nb,tw,4); memcpy(&nr,tw+4,4); scales=tw+8; packed=scales+nb; res=packed+(size_t)nb*16u; }
        const int64_t T=AXON_BT; const int sorted=((B->reserved&1)!=0);
        #pragma omp parallel for collapse(2) schedule(static) if(par)
        for(int64_t ii=0;ii<M;ii+=T)for(int64_t jj=0;jj<N;jj+=T){
            const int64_t ie=(ii+T<M)?ii+T:M, je=(jj+T<N)?jj+T:N, wj=je-jj;
            for(int64_t i=ii;i<ie;++i)for(int64_t j=jj;j<je;++j)c[i*N+j]=0.f;
            float bt[AXON_BT][AXON_BT];
            for(int64_t kk=0;kk<K;kk+=T){
                const int64_t ke=(kk+T<K)?kk+T:K; uint32_t cur_blk=UINT32_MAX; float cur_sc=1.0f;
                for(int64_t k=kk;k<ke;++k)for(int64_t j=jj;j<je;++j){
                    int64_t idx=k*N+j;
                    if(is_fp4){
                        uint32_t blk=(uint32_t)(idx>>5); if(blk!=cur_blk){cur_blk=blk;cur_sc=ldexpf(1.0f,(int)scales[blk]-127);}
                        int off=(int)(idx&31); uint8_t byte=packed[(size_t)blk*16u+(size_t)(off>>1)];
                        bt[k-kk][j-jj]=FP4_LUT[(off&1)?((byte>>4)&0x0F):(byte&0x0F)]*cur_sc;
                    } else { bt[k-kk][j-jj]=lut[(tw[idx>>2]>>((idx&3)*2))&3]; }
                }
                int64_t i=ii;
                for(;i+4<=ie;i+=4){
                    float*c0=c+(i+0)*N+jj,*c1=c+(i+1)*N+jj,*c2=c+(i+2)*N+jj,*c3=c+(i+3)*N+jj;
                    for(int64_t k=kk;k<ke;++k){
                        const float a0=a[(i+0)*K+k],a1=a[(i+1)*K+k],a2=a[(i+2)*K+k],a3=a[(i+3)*K+k]; const float*brow=bt[k-kk];
                        #pragma omp simd
                        for(int64_t j=0;j<wj;++j){const float bj=brow[j];c0[j]+=a0*bj;c1[j]+=a1*bj;c2[j]+=a2*bj;c3[j]+=a3*bj;}
                    }
                }
                for(;i<ie;++i){ float*ci=c+i*N+jj;
                    for(int64_t k=kk;k<ke;++k){ const float aik=a[i*K+k]; const float*brow=bt[k-kk];
                        #pragma omp simd
                        for(int64_t j=0;j<wj;++j)ci[j]+=aik*brow[j];
                    }
                }
            }
            if(is_fp4&&nr>0){
                if(sorted){
                    for(uint32_t r=fp4_res_lb(res,nr,N,jj);r<nr;++r){
                        uint32_t idx; float diff; memcpy(&idx,res+(size_t)r*8u,4); memcpy(&diff,res+(size_t)r*8u+4,4);
                        int64_t rk=(int64_t)idx/N, rj=(int64_t)idx%N; if(rj>=je)break;
                        for(int64_t i=ii;i<ie;++i)c[i*N+rj]+=a[i*K+rk]*diff;
                    }
                } else {
                    for(uint32_t r=0;r<nr;++r){
                        uint32_t idx; float diff; memcpy(&idx,res+(size_t)r*8u,4); memcpy(&diff,res+(size_t)r*8u+4,4);
                        if((int64_t)idx<K*N){ int64_t rk=(int64_t)idx/N, rj=(int64_t)idx%N;
                            if(rj>=jj&&rj<je)for(int64_t i=ii;i<ie;++i)c[i*N+rj]+=a[i*K+rk]*diff;
                        }
                    }
                }
            }
            for(int64_t i=ii;i<ie;++i){
                #pragma omp simd
                for(int64_t j=jj;j<je;++j){float v=c[i*N+j];if(bv)v+=bv[j];c[i*N+j]=act_apply(v,act);}
            }
        }
        return AXON_OK;
    }
    const float*b=(const float*)B->data;
    #pragma omp parallel for collapse(2) schedule(static) if(par)
    for(int64_t ii=0;ii<M;ii+=AXON_TILE)for(int64_t jj=0;jj<N;jj+=AXON_TILE){
        const int64_t ie=(ii+AXON_TILE<M)?ii+AXON_TILE:M, je=(jj+AXON_TILE<N)?jj+AXON_TILE:N, wj=je-jj;
        for(int64_t i=ii;i<ie;++i)for(int64_t j=jj;j<je;++j)c[i*N+j]=0.f;
        for(int64_t kk=0;kk<K;kk+=AXON_TILE){
            const int64_t ke=(kk+AXON_TILE<K)?kk+AXON_TILE:K; int64_t i=ii;
            for(;i+4<=ie;i+=4){
                float*c0=c+(i+0)*N+jj,*c1=c+(i+1)*N+jj,*c2=c+(i+2)*N+jj,*c3=c+(i+3)*N+jj;
                for(int64_t k=kk;k<ke;++k){
                    const float a0=a[(i+0)*K+k],a1=a[(i+1)*K+k],a2=a[(i+2)*K+k],a3=a[(i+3)*K+k]; const float*bk=b+k*N+jj;
                    #pragma omp simd
                    for(int64_t j=0;j<wj;++j){const float bj=bk[j];c0[j]+=a0*bj;c1[j]+=a1*bj;c2[j]+=a2*bj;c3[j]+=a3*bj;}
                }
            }
            for(;i<ie;++i){ float*ci=c+i*N+jj;
                for(int64_t k=kk;k<ke;++k){ const float aik=a[i*K+k]; const float*bk=b+k*N+jj;
                    #pragma omp simd
                    for(int64_t j=0;j<wj;++j)ci[j]+=aik*bk[j];
                }
            }
        }
        for(int64_t i=ii;i<ie;++i){
            #pragma omp simd
            for(int64_t j=jj;j<je;++j){ float v=c[i*N+j]; if(bv)v+=bv[j]; c[i*N+j]=act_apply(v,act); }
        }
    }
    return AXON_OK;
}
int axon_matmul(const AxonTensor*A,const AxonTensor*B,AxonTensor*C){return gemm(A,B,NULL,0,C);}
int axon_mm_relu(const AxonTensor*A,const AxonTensor*B,AxonTensor*C){return gemm(A,B,NULL,1,C);}
int axon_mm_gelu(const AxonTensor*A,const AxonTensor*B,AxonTensor*C){return gemm(A,B,NULL,2,C);}
int axon_mm_silu(const AxonTensor*A,const AxonTensor*B,AxonTensor*C){return gemm(A,B,NULL,3,C);}
int axon_mm_bias(const AxonTensor*A,const AxonTensor*B,const AxonTensor*b,AxonTensor*C){return gemm(A,B,b,0,C);}
int axon_mm_bias_relu(const AxonTensor*A,const AxonTensor*B,const AxonTensor*b,AxonTensor*C){return gemm(A,B,b,1,C);}
int axon_mm_bias_gelu(const AxonTensor*A,const AxonTensor*B,const AxonTensor*b,AxonTensor*C){return gemm(A,B,b,2,C);}
int axon_mm_bias_silu(const AxonTensor*A,const AxonTensor*B,const AxonTensor*b,AxonTensor*C){return gemm(A,B,b,3,C);}

static void region_byte_span(const AxonTensor *W, int64_t e0, int64_t e1, const uint8_t **out_ptr, size_t *out_len){
    const uint8_t *base = (const uint8_t*)W->data;
    if(W->dtype == AXON_DTYPE_F32){
        *out_ptr = base + (size_t)e0 * sizeof(float);
        *out_len = (size_t)(e1 - e0) * sizeof(float);
    } else if(W->dtype == AXON_DTYPE_TERNARY){
        size_t b0 = (size_t)(e0 >> 2);
        size_t b1 = (size_t)((e1 + 3) >> 2);
        *out_ptr = base + b0;
        *out_len = (b1 > b0) ? (b1 - b0) : 0u;
    } else {
        uint32_t nb = 0; memcpy(&nb, base, 4);
        const uint8_t *packed = base + 8u + nb;
        size_t b0 = (size_t)(e0 >> 1);
        size_t b1 = (size_t)((e1 + 1) >> 1);
        *out_ptr = packed + b0;
        *out_len = (b1 > b0) ? (b1 - b0) : 0u;
    }
}

int axon_neuro_matmul_paged(const AxonTensor*X,const AxonTensor*W,const AxonTensor*S,AxonTensor*C,int32_t keep_in,int32_t region_blocked){
    REQ(X&&W&&S&&C&&X->data&&W->data&&S->data&&C->data,AXON_ERR_ARG);
    REQ(X->ndim==2&&W->ndim==2&&S->ndim==2&&C->ndim==2,AXON_ERR_SHAPE);
    REQ(X->dtype==AXON_DTYPE_F32&&S->dtype==AXON_DTYPE_F32&&C->dtype==AXON_DTYPE_F32,AXON_ERR_DTYPE);
    REQ(W->dtype==AXON_DTYPE_F32||W->dtype==AXON_DTYPE_TERNARY||W->dtype==AXON_DTYPE_FP4,AXON_ERR_DTYPE);
    const int64_t M=X->shape[0],K=X->shape[1],N=W->shape[1],R=S->shape[1];
    REQ(W->shape[0]==K&&S->shape[0]==K&&R>=1&&R<=N&&C->shape[0]==M&&C->shape[1]==N,AXON_ERR_SHAPE);
    int64_t keep = (keep_in > 0) ? (int64_t)keep_in : ((R >= 2) ? (R / 2) : 1);
    if(keep > R) keep = R;

    const float*x=(const float*)X->data,*s=(const float*)S->data; float*c=(float*)C->data;
    memset(c,0,(size_t)(M*N)*sizeof(float));

    float st_sc[64]; int64_t st_ord[64]; uint8_t st_used[64];
    float*sc=(R<=64)?st_sc:(float*)malloc((size_t)R*sizeof(float));
    int64_t*ord=(R<=64)?st_ord:(int64_t*)malloc((size_t)R*sizeof(int64_t));
    uint8_t*used_reg=(R<=64)?st_used:(uint8_t*)malloc((size_t)R);
    int64_t*row_sel=(int64_t*)malloc((size_t)(M*keep)*sizeof(int64_t));
    if(!sc||!ord||!used_reg||!row_sel){
        if(R>64){free(sc);free(ord);free(used_reg);}
        free(row_sel);
        return AXON_ERR_NOMEM;
    }
    memset(used_reg, 0, (size_t)R);

    for(int64_t i=0;i<M;++i){
        for(int64_t r=0;r<R;++r){
            float acc=0.f; for(int64_t k=0;k<K;++k) acc+=x[i*K+k]*s[k*R+r];
            sc[r]=fabsf(acc); ord[r]=r;
        }
        for(int64_t t=0;t<keep;++t){
            int64_t best=t;
            for(int64_t u=t+1;u<R;++u) if(sc[ord[u]]>sc[ord[best]]||(sc[ord[u]]==sc[ord[best]]&&ord[u]<ord[best])) best=u;
            int64_t tmp=ord[t]; ord[t]=ord[best]; ord[best]=tmp;
            row_sel[i*keep+t]=ord[t];
            used_reg[ord[t]]=1;
        }
    }

    int64_t unique_cnt=0, bytes_active=0, bytes_total=0;
    const int is_mmap = is_in_mmap(W->data);
    for(int64_t r=0;r<R;++r){
        int64_t c0=(r*N)/R, c1=((r+1)*N)/R;
        const uint8_t *rp=NULL; size_t rlen=0;
        region_byte_span(W, c0*K, c1*K, &rp, &rlen);
        bytes_total += (int64_t)rlen;
        if(used_reg[r]){ unique_cnt++; bytes_active += (int64_t)rlen; }
    }
    if(W->dtype==AXON_DTYPE_FP4){
        const uint8_t*p=(const uint8_t*)W->data; uint32_t nb=0,nr=0; memcpy(&nb,p,4); memcpy(&nr,p+4,4);
        int64_t extra = 8 + (int64_t)nb + (int64_t)nr * 8;
        bytes_total += extra;
        bytes_active += extra;
    } else if(W->dtype==AXON_DTYPE_TERNARY){
        bytes_total += 8; bytes_active += 8;
    }

    const int do_cold_paging = (region_blocked && is_mmap && bytes_total >= AXON_PAGING_MIN_BYTES);
    if(do_cold_paging || (region_blocked && is_mmap)){
        for(int64_t r=0;r<R;++r){
            int64_t c0=(r*N)/R, c1=((r+1)*N)/R;
            const uint8_t *rp=NULL; size_t rlen=0;
            region_byte_span(W, c0*K, c1*K, &rp, &rlen);
            if(rlen>0){
                if(used_reg[r]) advise_range(rp, rlen, 1);
                else if(do_cold_paging) advise_range(rp, rlen, 0);
            }
        }
    }

    int64_t active_cols_eval = 0;
    for(int64_t i=0;i<M;++i){
        for(int64_t t=0;t<keep;++t){
            int64_t r=row_sel[i*keep+t], c0=(r*N)/R, c1=((r+1)*N)/R;
            active_cols_eval += (c1 - c0);
        }
    }
    const int par = ((M * K * N) >= AXON_PAR_MIN);

    #pragma omp parallel for collapse(2) schedule(static) if(par)
    for(int64_t i=0;i<M;++i){
        for(int64_t t=0;t<keep;++t){
            int64_t r=row_sel[i*keep+t], c0=(r*N)/R, c1=((r+1)*N)/R, wr=c1-c0;
            if(W->dtype==AXON_DTYPE_F32){
                const float*w=(const float*)W->data;
                if(region_blocked){
                    const float*wreg=w + c0*K;
                    for(int64_t k=0;k<K;++k){
                        const float xik=x[i*K+k]; const float*wk=wreg + k*wr;
                        #pragma omp simd
                        for(int64_t jo=0;jo<wr;++jo) c[i*N+c0+jo]+=xik*wk[jo];
                    }
                } else {
                    for(int64_t k=0;k<K;++k){
                        const float xik=x[i*K+k];
                        #pragma omp simd
                        for(int64_t j=c0;j<c1;++j) c[i*N+j]+=xik*w[k*N+j];
                    }
                }
            } else if(W->dtype==AXON_DTYPE_TERNARY){
                const uint8_t*w=(const uint8_t*)W->data; const float lut[4]={0.f,W->scale,-W->scale,0.f};
                if(region_blocked){
                    int64_t base=c0*K;
                    for(int64_t k=0;k<K;++k){
                        const float xik=x[i*K+k];
                        for(int64_t jo=0;jo<wr;++jo){ int64_t idx=base+k*wr+jo; c[i*N+c0+jo]+=xik*lut[(w[idx>>2]>>((idx&3)*2))&3]; }
                    }
                } else {
                    for(int64_t k=0;k<K;++k){
                        const float xik=x[i*K+k];
                        for(int64_t j=c0;j<c1;++j){ int64_t idx=k*N+j; c[i*N+j]+=xik*lut[(w[idx>>2]>>((idx&3)*2))&3]; }
                    }
                }
            } else {
                const uint8_t*p=(const uint8_t*)W->data; uint32_t nb=0,nr=0; memcpy(&nb,p,4); memcpy(&nr,p+4,4);
                const uint8_t*scales=p+8,*packed=scales+nb,*res=packed+(size_t)nb*16u;
                uint32_t cur_blk=UINT32_MAX; float cur_sc=1.0f;
                if(region_blocked){
                    int64_t base=c0*K;
                    for(int64_t k=0;k<K;++k){
                        const float xik=x[i*K+k];
                        for(int64_t jo=0;jo<wr;++jo){
                            int64_t idx=base+k*wr+jo; uint32_t blk=(uint32_t)(idx>>5);
                            if(blk!=cur_blk){cur_blk=blk;cur_sc=ldexpf(1.0f,(int)scales[blk]-127);}
                            int off=(int)(idx&31); uint8_t byte=packed[(size_t)blk*16u+(size_t)(off>>1)];
                            c[i*N+c0+jo]+=xik*FP4_LUT[(off&1)?((byte>>4)&0x0F):(byte&0x0F)]*cur_sc;
                        }
                    }
                    int64_t rstart=c0*K, rend=c1*K;
                    if((W->reserved&2)!=0 && nr>0){
                        for(uint32_t ri=fp4_res_lb_flat(res,nr,rstart);ri<nr;++ri){
                            uint32_t idx; float diff; memcpy(&idx,res+(size_t)ri*8u,4); memcpy(&diff,res+(size_t)ri*8u+4,4);
                            if((int64_t)idx>=rend) break;
                            int64_t off=(int64_t)idx-rstart, rk=off/wr, jo=off%wr;
                            c[i*N+c0+jo]+=x[i*K+rk]*diff;
                        }
                    } else {
                        for(uint32_t ri=0;ri<nr;++ri){
                            uint32_t idx; float diff; memcpy(&idx,res+(size_t)ri*8u,4); memcpy(&diff,res+(size_t)ri*8u+4,4);
                            if((int64_t)idx>=rstart && (int64_t)idx<rend){
                                int64_t off=(int64_t)idx-rstart, rk=off/wr, jo=off%wr;
                                c[i*N+c0+jo]+=x[i*K+rk]*diff;
                            }
                        }
                    }
                } else {
                    for(int64_t k=0;k<K;++k){
                        const float xik=x[i*K+k];
                        for(int64_t j=c0;j<c1;++j){
                            int64_t idx=k*N+j; uint32_t blk=(uint32_t)(idx>>5); if(blk!=cur_blk){cur_blk=blk;cur_sc=ldexpf(1.0f,(int)scales[blk]-127);}
                            int off=(int)(idx&31); uint8_t byte=packed[(size_t)blk*16u+(size_t)(off>>1)];
                            c[i*N+j]+=xik*FP4_LUT[(off&1)?((byte>>4)&0x0F):(byte&0x0F)]*cur_sc;
                        }
                    }
                    if((W->reserved&1)!=0 && nr>0){
                        for(uint32_t ri=fp4_res_lb(res,nr,N,c0);ri<nr;++ri){
                            uint32_t idx; float diff; memcpy(&idx,res+(size_t)ri*8u,4); memcpy(&diff,res+(size_t)ri*8u+4,4);
                            int64_t rk=(int64_t)idx/N, rj=(int64_t)idx%N; if(rj>=c1)break; c[i*N+rj]+=x[i*K+rk]*diff;
                        }
                    } else {
                        for(uint32_t ri=0;ri<nr;++ri){
                            uint32_t idx; float diff; memcpy(&idx,res+(size_t)ri*8u,4); memcpy(&diff,res+(size_t)ri*8u+4,4);
                            if((int64_t)idx<K*N){int64_t rk=(int64_t)idx/N,rj=(int64_t)idx%N;if(rj>=c0&&rj<c1)c[i*N+rj]+=x[i*K+rk]*diff;}
                        }
                    }
                }
            }
        }
    }
    if(do_cold_paging){
        for(int64_t r=0;r<R;++r){
            if(!used_reg[r]) continue;
            int64_t c0=(r*N)/R, c1=((r+1)*N)/R;
            const uint8_t *rp=NULL; size_t rlen=0;
            region_byte_span(W, c0*K, c1*K, &rp, &rlen);
            if(rlen>0) advise_range(rp, rlen, 0);
        }
    }
    int64_t exec_flops = 2 * M * K * R + 2 * K * active_cols_eval;
    int64_t dense_flops = 2 * M * K * N;
    int64_t f32_bytes = K * N * (int64_t)sizeof(float);
    axon_brain_record_stats(M * keep, M * R, unique_cnt, R, bytes_active, bytes_total, f32_bytes, exec_flops, dense_flops);

    if(R>64){free(sc);free(ord);free(used_reg);}
    free(row_sel);
    return AXON_OK;
}

int axon_neuro_matmul(const AxonTensor*X,const AxonTensor*W,const AxonTensor*S,AxonTensor*C){
    return axon_neuro_matmul_paged(X, W, S, C, 0, 0);
}

static int bcast_strides(const AxonTensor*T,int64_t nd,const int64_t*shape,int64_t*st){
    for(int64_t i=0;i<nd;++i)st[i]=0; int64_t off=nd-T->ndim; if(off<0)return AXON_ERR_SHAPE;
    for(int64_t i=0;i<T->ndim;++i){ int64_t d=T->shape[i],s=shape[off+i]; if(d==s)st[off+i]=T->strides[i]; else if(d==1)st[off+i]=0; else return AXON_ERR_SHAPE; }
    return AXON_OK;
}
static int elementwise(const AxonTensor*A,const AxonTensor*B,AxonTensor*C,int mode){
    REQ(A&&B&&C&&A->data&&B->data&&C->data,AXON_ERR_ARG);
    REQ(A->dtype==AXON_DTYPE_F32&&B->dtype==AXON_DTYPE_F32&&C->dtype==AXON_DTYPE_F32,AXON_ERR_DTYPE);
    REQ(C->ndim>=1&&C->ndim<=AXON_MAX_DIMS,AXON_ERR_SHAPE);
    const float*a=(const float*)A->data,*b=(const float*)B->data; float*c=(float*)C->data; const int64_t n=axon_tensor_numel(C);
    int same=(A->ndim==C->ndim&&B->ndim==C->ndim);
    if(same)for(int64_t i=0;i<C->ndim;++i)if(A->shape[i]!=C->shape[i]||B->shape[i]!=C->shape[i])same=0;
    if(same){
        #pragma omp parallel for schedule(static) if(n>=AXON_PAR_MIN)
        for(int64_t i=0;i<n;++i){float x=a[i],y=b[i]; c[i]=mode==0?x+y:(mode==1?x-y:x*y);} return AXON_OK;
    }
    int64_t sa[AXON_MAX_DIMS],sb[AXON_MAX_DIMS];
    REQ(bcast_strides(A,C->ndim,C->shape,sa)==AXON_OK,AXON_ERR_SHAPE); REQ(bcast_strides(B,C->ndim,C->shape,sb)==AXON_OK,AXON_ERR_SHAPE);
    for(int64_t lin=0;lin<n;++lin){ int64_t rem=lin,oa=0,ob=0;
        for(int64_t d=0;d<C->ndim;++d){ int64_t idx=rem/C->strides[d]; rem-=idx*C->strides[d]; oa+=idx*sa[d]; ob+=idx*sb[d]; }
        float x=a[oa],y=b[ob]; c[lin]=mode==0?x+y:(mode==1?x-y:x*y);
    }
    return AXON_OK;
}
int axon_add(const AxonTensor*A,const AxonTensor*B,AxonTensor*C){return elementwise(A,B,C,0);}
int axon_sub(const AxonTensor*A,const AxonTensor*B,AxonTensor*C){return elementwise(A,B,C,1);}
int axon_mul(const AxonTensor*A,const AxonTensor*B,AxonTensor*C){return elementwise(A,B,C,2);}
static int unary(const AxonTensor*A,AxonTensor*C,int mode){
    REQ(A&&C&&A->data&&C->data,AXON_ERR_ARG); REQ(A->dtype==AXON_DTYPE_F32&&C->dtype==AXON_DTYPE_F32,AXON_ERR_DTYPE);
    REQ(axon_tensor_numel(A)==axon_tensor_numel(C),AXON_ERR_SHAPE);
    const float*a=(const float*)A->data; float*c=(float*)C->data; const int64_t n=axon_tensor_numel(A);
    #pragma omp parallel for schedule(static) if(n>=AXON_PAR_MIN)
    for(int64_t i=0;i<n;++i)c[i]=act_apply(a[i],mode); return AXON_OK;
}
int axon_relu(const AxonTensor*A,AxonTensor*C){return unary(A,C,1);}
int axon_gelu(const AxonTensor*A,AxonTensor*C){return unary(A,C,2);}
int axon_silu(const AxonTensor*A,AxonTensor*C){return unary(A,C,3);}
int axon_sigmoid(const AxonTensor*A,AxonTensor*C){return unary(A,C,4);}
int axon_softmax(const AxonTensor*A,AxonTensor*C){
    REQ(A&&C&&A->data&&C->data,AXON_ERR_ARG); REQ(A->dtype==AXON_DTYPE_F32&&C->dtype==AXON_DTYPE_F32,AXON_ERR_DTYPE);
    REQ(A->ndim>=1&&A->ndim==C->ndim,AXON_ERR_SHAPE);
    for(int64_t i=0;i<A->ndim;++i)REQ(A->shape[i]==C->shape[i],AXON_ERR_SHAPE);
    const int64_t cols=A->shape[A->ndim-1]; REQ(cols>0,AXON_ERR_SHAPE);
    int64_t rows=1; for(int64_t i=0;i<A->ndim-1;++i)rows*=A->shape[i];
    const float*a=(const float*)A->data; float*c=(float*)C->data;
    #pragma omp parallel for schedule(static) if(rows*cols>=AXON_PAR_MIN)
    for(int64_t r=0;r<rows;++r){ const float*x=a+r*cols; float*y=c+r*cols; float mx=x[0];
        for(int64_t j=1;j<cols;++j)if(x[j]>mx)mx=x[j];
        float sum=0.f; for(int64_t j=0;j<cols;++j){float e=expf(x[j]-mx);y[j]=e;sum+=e;}
        float inv=sum>0.f?1.f/sum:0.f; for(int64_t j=0;j<cols;++j)y[j]*=inv;
    }
    return AXON_OK;
}
int axon_layer_norm(const AxonTensor*X,const AxonTensor*G,const AxonTensor*B,AxonTensor*C){
    REQ(X&&C&&X->data&&C->data,AXON_ERR_ARG); REQ(X->dtype==AXON_DTYPE_F32&&C->dtype==AXON_DTYPE_F32,AXON_ERR_DTYPE);
    REQ(X->ndim>=1&&X->ndim==C->ndim,AXON_ERR_SHAPE);
    for(int64_t i=0;i<X->ndim;++i)REQ(X->shape[i]==C->shape[i],AXON_ERR_SHAPE);
    const int64_t cols=X->shape[X->ndim-1]; REQ(cols>0,AXON_ERR_SHAPE);
    int64_t rows=1; for(int64_t i=0;i<X->ndim-1;++i)rows*=X->shape[i];
    const float*g=NULL,*b=NULL;
    if(G){REQ(G->data&&G->ndim==1&&G->shape[0]==cols&&G->dtype==AXON_DTYPE_F32,AXON_ERR_SHAPE);g=(const float*)G->data;}
    if(B){REQ(B->data&&B->ndim==1&&B->shape[0]==cols&&B->dtype==AXON_DTYPE_F32,AXON_ERR_SHAPE);b=(const float*)B->data;}
    const float*x=(const float*)X->data; float*c=(float*)C->data;
    #pragma omp parallel for schedule(static) if(rows*cols>=AXON_PAR_MIN)
    for(int64_t r=0;r<rows;++r){ const float*xr=x+r*cols; float*yr=c+r*cols; float mean=0.f;
        for(int64_t j=0;j<cols;++j)mean+=xr[j]; mean/=(float)cols; float var=0.f;
        for(int64_t j=0;j<cols;++j){float d=xr[j]-mean;var+=d*d;}
        float inv=1.f/sqrtf(var/(float)cols+LN_EPS);
        for(int64_t j=0;j<cols;++j)yr[j]=((xr[j]-mean)*inv)*(g?g[j]:1.f)+(b?b[j]:0.f);
    }
    return AXON_OK;
}
int axon_rms_norm(const AxonTensor*X,const AxonTensor*G,AxonTensor*C){
    REQ(X&&C&&X->data&&C->data,AXON_ERR_ARG); REQ(X->dtype==AXON_DTYPE_F32&&C->dtype==AXON_DTYPE_F32&&X->ndim>=1&&X->ndim==C->ndim,AXON_ERR_DTYPE);
    for(int64_t i=0;i<X->ndim;++i)REQ(X->shape[i]==C->shape[i],AXON_ERR_SHAPE);
    const int64_t cols=X->shape[X->ndim-1]; REQ(cols>0,AXON_ERR_SHAPE);
    int64_t rows=1; for(int64_t i=0;i<X->ndim-1;++i)rows*=X->shape[i];
    const float*g=NULL; if(G){REQ(G->data&&G->ndim==1&&G->shape[0]==cols&&G->dtype==AXON_DTYPE_F32,AXON_ERR_SHAPE);g=(const float*)G->data;}
    const float*x=(const float*)X->data; float*c=(float*)C->data;
    #pragma omp parallel for schedule(static) if(rows*cols>=AXON_PAR_MIN)
    for(int64_t r=0;r<rows;++r){ const float*xr=x+r*cols; float*yr=c+r*cols; float ms=0.f;
        for(int64_t j=0;j<cols;++j)ms+=xr[j]*xr[j];
        float inv=1.f/sqrtf(ms/(float)cols+LN_EPS);
        for(int64_t j=0;j<cols;++j)yr[j]=(xr[j]*inv)*(g?g[j]:1.f);
    }
    return AXON_OK;
}
int axon_rope(const AxonTensor*X,AxonTensor*C){
    REQ(X&&C&&X->data&&C->data,AXON_ERR_ARG); REQ(X->ndim==2&&C->ndim==2&&X->dtype==AXON_DTYPE_F32&&C->dtype==AXON_DTYPE_F32,AXON_ERR_SHAPE);
    const int64_t S=X->shape[0],D=X->shape[1]; REQ(S>0&&D>0&&(D%2)==0&&C->shape[0]==S&&C->shape[1]==D,AXON_ERR_SHAPE);
    const float*x=(const float*)X->data; float*c=(float*)C->data;
    #pragma omp parallel for schedule(static) if(S*D>=AXON_PAR_MIN)
    for(int64_t p=0;p<S;++p)for(int64_t k=0;k<D/2;++k){
        float th=(float)p*powf(10000.f,-2.f*(float)k/(float)D), cs=cosf(th), sn=sinf(th);
        float x0=x[p*D+2*k], x1=x[p*D+2*k+1];
        c[p*D+2*k]=x0*cs-x1*sn; c[p*D+2*k+1]=x0*sn+x1*cs;
    }
    return AXON_OK;
}
int axon_attn_causal(const AxonTensor*Q,const AxonTensor*K,const AxonTensor*V,AxonTensor*C){
    REQ(Q&&K&&V&&C&&Q->data&&K->data&&V->data&&C->data,AXON_ERR_ARG);
    REQ(Q->ndim==2&&K->ndim==2&&V->ndim==2&&C->ndim==2,AXON_ERR_SHAPE);
    REQ(Q->dtype==AXON_DTYPE_F32&&K->dtype==AXON_DTYPE_F32&&V->dtype==AXON_DTYPE_F32&&C->dtype==AXON_DTYPE_F32,AXON_ERR_DTYPE);
    const int64_t S=Q->shape[0],D=Q->shape[1];
    REQ(S>0&&D>0&&K->shape[0]==S&&K->shape[1]==D&&V->shape[0]==S&&V->shape[1]==D&&C->shape[0]==S&&C->shape[1]==D,AXON_ERR_SHAPE);
    const float*q=(const float*)Q->data,*k=(const float*)K->data,*v=(const float*)V->data; float*c=(float*)C->data;
    const float inv=1.f/sqrtf((float)D);
    #pragma omp parallel for schedule(static) if(S*S*D>=AXON_PAR_MIN)
    for(int64_t i=0;i<S;++i){
        const float*qi=q+i*D; float*ci=c+i*D; for(int64_t d=0;d<D;++d)ci[d]=0.f;
        float m=-INFINITY, l=0.f;
        for(int64_t j=0;j<=i;++j){
            const float*kj=k+j*D,*vj=v+j*D; float dot=0.f; for(int64_t d=0;d<D;++d)dot+=qi[d]*kj[d];
            float sc=dot*inv, mn=(sc>m)?sc:m, alpha=(m==-INFINITY)?0.f:expf(m-mn), beta=expf(sc-mn);
            for(int64_t d=0;d<D;++d)ci[d]=ci[d]*alpha+beta*vj[d];
            l=l*alpha+beta; m=mn;
        }
        float il=(l>0.f)?(1.f/l):0.f; for(int64_t d=0;d<D;++d)ci[d]*=il;
    }
    return AXON_OK;
}
int axon_transpose(const AxonTensor*A,AxonTensor*C){
    REQ(A&&C&&A->data&&C->data,AXON_ERR_ARG); REQ(A->ndim==2&&C->ndim==2,AXON_ERR_SHAPE);
    REQ(A->dtype==AXON_DTYPE_F32&&C->dtype==AXON_DTYPE_F32,AXON_ERR_DTYPE);
    const int64_t M=A->shape[0],N=A->shape[1]; REQ(C->shape[0]==N&&C->shape[1]==M,AXON_ERR_SHAPE);
    const float*a=(const float*)A->data; float*c=(float*)C->data;
    #pragma omp parallel for collapse(2) schedule(static) if(M*N>=AXON_PAR_MIN)
    for(int64_t i=0;i<M;++i)for(int64_t j=0;j<N;++j)c[j*M+i]=a[i*N+j];
    return AXON_OK;
}
int axon_equilibrium(const AxonTensor*X,const AxonTensor*W,const AxonTensor*B,AxonTensor*C){
    REQ(X&&W&&C&&X->data&&W->data&&C->data,AXON_ERR_ARG);
    REQ(X->dtype==AXON_DTYPE_F32&&W->dtype==AXON_DTYPE_F32&&C->dtype==AXON_DTYPE_F32,AXON_ERR_DTYPE);
    REQ(X->ndim>=1&&X->ndim==C->ndim,AXON_ERR_SHAPE);
    for(int64_t i=0;i<X->ndim;++i)REQ(X->shape[i]==C->shape[i],AXON_ERR_SHAPE);
    const int64_t D=X->shape[X->ndim-1]; REQ(D>0&&W->ndim==2&&W->shape[0]==D&&W->shape[1]==D,AXON_ERR_SHAPE);
    const float*b=NULL; if(B){REQ(B->data&&B->ndim==1&&B->shape[0]==D&&B->dtype==AXON_DTYPE_F32,AXON_ERR_SHAPE);b=(const float*)B->data;}
    int64_t M=1; for(int64_t i=0;i<X->ndim-1;++i)M*=X->shape[i];
    const float*x=(const float*)X->data,*w=(const float*)W->data; float*s=(float*)C->data;
    memcpy(s,x,(size_t)(M*D)*sizeof(float));
    AxonTensor*tmp=axon_tensor_alloc(X->ndim,X->shape,AXON_DTYPE_F32); REQ(tmp&&tmp->data,AXON_ERR_NOMEM);
    float*sn=(float*)tmp->data;
    for(int step=0;step<16;++step){ float max_diff=0.f;
        for(int64_t i=0;i<M;++i)for(int64_t j=0;j<D;++j){
            float acc=x[i*D+j]+(b?b[j]:0.f); for(int64_t k=0;k<D;++k)acc+=s[i*D+k]*w[k*D+j];
            float v=0.7f*s[i*D+j]+0.3f*tanhf(acc); float d=fabsf(v-s[i*D+j]); if(d>max_diff)max_diff=d; sn[i*D+j]=v;
        }
        memcpy(s,sn,(size_t)(M*D)*sizeof(float)); if(max_diff<1e-4f)break;
    }
    axon_tensor_free(tmp); return AXON_OK;
}
int axon_bench_matmul(int n,double*out_ms,float*out_c){
    REQ(n>0&&out_ms&&out_c,AXON_ERR_ARG); int64_t shp[2]={n,n};
    AxonTensor*A=axon_tensor_alloc_persistent(2,shp,AXON_DTYPE_F32); AxonTensor*B=axon_tensor_alloc_persistent(2,shp,AXON_DTYPE_F32); AxonTensor*C=axon_tensor_alloc_persistent(2,shp,AXON_DTYPE_F32);
    if(!A||!B||!C){axon_tensor_free(A);axon_tensor_free(B);axon_tensor_free(C);return AXON_ERR_NOMEM;}
    float*a=(float*)A->data,*b=(float*)B->data;
    for(int64_t i=0;i<n;++i)for(int64_t j=0;j<n;++j){a[i*n+j]=(float)(i%10)*0.01f;b[i*n+j]=(float)(j%10)*0.01f;}
    double t0 = axon_now_ms(); int rc=axon_matmul(A,B,C); *out_ms = axon_now_ms() - t0;
    if(rc==AXON_OK)memcpy(out_c,C->data,(size_t)n*n*sizeof(float));
    axon_tensor_free(A);axon_tensor_free(B);axon_tensor_free(C); return rc;
}
#define LSH_BITS 6
#define LSH_BUCKETS (1<<LSH_BITS)
static inline uint32_t xs32(uint32_t*s){uint32_t x=*s;x^=x<<13;x^=x>>17;x^=x<<5;*s=x;return x;}
void axon_lsh_benchmark(int dim){
    if(dim<8){printf("dim must be >=8\n");return;}
    const int K=dim,N=dim,REPS=50;
    float*A=(float*)malloc((size_t)K*4),*B=(float*)malloc((size_t)K*N*4),*Cd=(float*)calloc((size_t)N,4),*Cl=(float*)calloc((size_t)N,4);
    int*cnt=(int*)calloc(LSH_BUCKETS,sizeof(int)),*cols=(int*)malloc((size_t)LSH_BUCKETS*N*sizeof(int));
    if(!A||!B||!Cd||!Cl||!cnt||!cols)goto done;
    for(int i=0;i<K;++i)A[i]=((i%7)-3)*0.1f;
    for(int i=0;i<K*N;++i)B[i]=((i%11)-5)*0.05f;
    for(int j=0;j<N;++j){ uint8_t h=0;
        for(int b=0;b<LSH_BITS;++b){ uint32_t rng=1337u+(uint32_t)b*7919u; float proj=0.f; for(int k=0;k<K;++k)proj+=B[(size_t)k*N+j]*((xs32(&rng)&1)?1.f:-1.f); if(proj>=0.f)h|=(uint8_t)(1u<<b); }
        cols[(size_t)h*N+cnt[h]++]=j;
    }
    double t0 = axon_now_ms();
    for(int r=0;r<REPS;++r)for(int j=0;j<N;++j){float s=0.f;for(int k=0;k<K;++k)s+=A[k]*B[(size_t)k*N+j];Cd[j]=s;}
    double t1 = axon_now_ms();
    uint8_t hq=0; for(int b=0;b<LSH_BITS;++b){uint32_t rng=1337u+(uint32_t)b*7919u;float proj=0.f;for(int k=0;k<K;++k)proj+=A[k]*((xs32(&rng)&1)?1.f:-1.f);if(proj>=0.f)hq|=(uint8_t)(1u<<b);}
    double t2 = axon_now_ms();
    for(int r=0;r<REPS;++r){ memset(Cl,0,(size_t)N*4);
        for(int nb=0;nb<=LSH_BITS;++nb){ uint8_t hb=nb==0?hq:(uint8_t)(hq^(1u<<(nb-1))); int c=cnt[hb];
            for(int t=0;t<c;++t){int j=cols[(size_t)hb*N+t];float s=0.f;for(int k=0;k<K;++k)s+=A[k]*B[(size_t)k*N+j];Cl[j]=s;}
        }
    }
    double t3 = axon_now_ms();
    int hit=0; double num=0.0,den=0.0;
    for(int j=0;j<N;++j){double d=Cd[j],l=Cl[j]; if(fabs(d)>1e-6&&fabs(l-d)<=1e-4*fabs(d))hit++; num+=(l-d)*(l-d);den+=d*d;}
    printf("=== AXON LSH (%dx%d) === dense: %.3fms | lsh: %.3fms | recall: %.1f%% | rel_L2: %.4f\n",K,N,
           t1-t0, t3-t2, 100.0*hit/N, den>0?sqrt(num/den):0.0);
done: free(A);free(B);free(Cd);free(Cl);free(cnt);free(cols);
}
'''


V11_MAIN_C = r'''#ifndef _WIN32
#  define _POSIX_C_SOURCE 200809L
#endif
#include "axon_runtime.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
static double main_now_us(void){
    static LARGE_INTEGER freq; static int init=0;
    if(!init){ QueryPerformanceFrequency(&freq); init=1; }
    LARGE_INTEGER now; QueryPerformanceCounter(&now);
    return ((double)now.QuadPart * 1e6) / (double)freq.QuadPart;
}
#else
static double main_now_us(void){
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e6 + (double)t.tv_nsec / 1e3;
}
#endif
static void usage(const char*p){ printf("usage: %s [--input \"a,b,c\"] [--weights file.axn3] [--iters N] [--generate N] [--arena-mb N] [--max-print N] [--quiet]\n",p); }
static float* parse_input(const char*text,int64_t*count){
    size_t cap=64,n=0; float*v=(float*)malloc(cap*sizeof(float)); if(!v)return NULL;
    const char*p=text;
    while(*p){ char*end=NULL; float f=strtof(p,&end); if(end==p){free(v);return NULL;}
        if(n==cap){cap*=2;float*nv=(float*)realloc(v,cap*sizeof(float));if(!nv){free(v);return NULL;}v=nv;}
        v[n++]=f;p=end; while(*p==','||*p==' '||*p=='\t')++p;
    }
    *count=(int64_t)n; return v;
}
int main(int argc,char**argv){
    const char*input_text=NULL,*weights_path=NULL; int iters=1,gen_steps=0,quiet=0; size_t arena_mb=64; int64_t max_print=16;
    for(int i=1;i<argc;++i){
        if(!strcmp(argv[i],"--help")||!strcmp(argv[i],"-h")){usage(argv[0]);return 0;}
        else if(!strcmp(argv[i],"--input")&&i+1<argc)input_text=argv[++i];
        else if(!strcmp(argv[i],"--weights")&&i+1<argc)weights_path=argv[++i];
        else if(!strcmp(argv[i],"--iters")&&i+1<argc){iters=atoi(argv[++i]);if(iters<1)iters=1;}
        else if(!strcmp(argv[i],"--generate")&&i+1<argc){gen_steps=atoi(argv[++i]);if(gen_steps<0)gen_steps=0;}
        else if(!strcmp(argv[i],"--arena-mb")&&i+1<argc){long v=atol(argv[++i]);if(v>=0)arena_mb=(size_t)v;}
        else if(!strcmp(argv[i],"--max-print")&&i+1<argc){long v=atol(argv[++i]);if(v>0)max_print=(int64_t)v;}
        else if(!strcmp(argv[i],"--quiet"))quiet=1;
        else{fprintf(stderr,"unknown argument: %s\n",argv[i]);usage(argv[0]);return 2;}
    }
    if(weights_path){
        if(axon_load_weights_file(weights_path)!=AXON_OK){fprintf(stderr,"failed to mmap weights: %s\n",weights_path);return 2;}
    } else {
        const void*bp=NULL; int64_t bn=0; axon_weights_get(&bp,&bn);
        if(!bp){
            int loaded = 0;
            size_t len=strlen(argv[0])+8; char*ap=(char*)malloc(len);
            if(ap){
                snprintf(ap,len,"%s.axn3",argv[0]);
                if(axon_load_weights_file(ap)==AXON_OK) loaded=1;
                free(ap);
            }
#ifdef _WIN32
            if(!loaded){
                char exe_p[4096]; DWORD elen = GetModuleFileNameA(NULL, exe_p, (DWORD)(sizeof(exe_p)-8));
                if(elen > 0 && elen < sizeof(exe_p)-8){
                    snprintf(exe_p + elen, sizeof(exe_p) - elen, ".axn3");
                    if(axon_load_weights_file(exe_p)==AXON_OK) loaded=1;
                }
            }
#endif
            if(!loaded){fprintf(stderr,"missing weights (tried %s.axn3)\n",argv[0]);return 2;}
        }
    }
    const int64_t ndim=axon_get_input_ndim(); int64_t shape[AXON_MAX_DIMS]={0};
    if(ndim<1||ndim>AXON_MAX_DIMS){fprintf(stderr,"bad input rank\n");axon_unload_weights_file();return 1;}
    axon_get_input_shape(shape); const int64_t need=axon_get_input_numel(); float*vals=NULL; int64_t got=0;
    if(input_text){ vals=parse_input(input_text,&got); if(!vals){fprintf(stderr,"bad --input\n");axon_unload_weights_file();return 2;}
        if(got!=need){fprintf(stderr,"input size mismatch: need %lld got %lld\n",(long long)need,(long long)got);free(vals);axon_unload_weights_file();return 2;}
    } else { got=need; vals=(float*)malloc((size_t)need*sizeof(float)); if(!vals){axon_unload_weights_file();return 1;}
        const float seed[4]={1.f,0.5f,-1.f,2.f}; for(int64_t i=0;i<need;++i)vals[i]=seed[i%4];
    }
    AxonTensor*inp=axon_tensor_alloc_persistent(ndim,shape,AXON_DTYPE_F32);
    if(!inp||axon_tensor_fill_f32(inp,vals,got)!=AXON_OK){fprintf(stderr,"input alloc failed\n");axon_tensor_free(inp);free(vals);axon_unload_weights_file();return 1;}
    free(vals); AxonTensor*inputs[1]={inp};
    if(arena_mb>0) axon_arena_init(arena_mb*1024u*1024u);
    const int temp_out=(axon_output_is_temp()!=0);

    if(gen_steps > 0){
        const int64_t out_need = axon_get_output_numel();
        const int64_t seq_len = (ndim == 2) ? shape[0] : 1;
        const int64_t in_dim = (ndim == 2) ? shape[1] : need;
        const int64_t out_dim = (ndim == 2 && seq_len > 0 && out_need % seq_len == 0) ? (out_need / seq_len) : out_need;
        const int64_t kv_dim = (out_dim > 0) ? out_dim : 1;
        AxonKVCache *kv = axon_kv_cache_alloc((int64_t)gen_steps + 1, kv_dim);
        float *ctx = (float*)malloc((size_t)kv_dim * sizeof(float));
        if(!kv || !ctx){
            free(ctx); axon_kv_cache_free(kv); axon_shutdown(); axon_tensor_free(inp); axon_arena_destroy(); axon_unload_weights_file();
            fprintf(stderr,"kv cache alloc failed\n"); return 1;
        }
        const size_t kv_mark = axon_arena_mark();
        double gt0 = main_now_us();
        for(int step = 0; step < gen_steps; ++step){
            if(arena_mb > 0) axon_arena_rewind(kv_mark);
            AxonTensor *gout = axon_forward(inputs);
            if(!gout || !gout->data){
                if(gout && temp_out) axon_tensor_free(gout);
                free(ctx); axon_kv_cache_free(kv); axon_shutdown(); axon_tensor_free(inp); axon_arena_destroy(); axon_unload_weights_file();
                fprintf(stderr,"generate failed at step %d\n", step); return 1;
            }
            const int64_t offset = (out_need >= kv_dim) ? (out_need - kv_dim) : 0;
            const float *last_h = ((const float*)gout->data) + offset;
            axon_kv_cache_append(kv, last_h, last_h);
            axon_kv_cache_attend(kv, last_h, ctx);
            int best_tok = 0; float best_score = -1e30f;
            for(int64_t d = 0; d < kv_dim; ++d){
                float blended = 0.5f * last_h[d] + 0.5f * ctx[d];
                if(blended > best_score){ best_score = blended; best_tok = (int)d; }
            }
            if(!quiet){
                printf("[generate step %d/%d] token=%d score=%.6f kv_len=%lld/%lld arena=%zu/%zu bytes\n",
                       step + 1, gen_steps, best_tok, (double)best_score,
                       (long long)kv->len, (long long)kv->max_steps, axon_arena_used(), axon_arena_capacity());
            }
            float *in_d = (float*)inp->data;
            const int64_t copy_d = (in_dim < kv_dim) ? in_dim : kv_dim;
            if(ndim == 2 && seq_len > 1 && need == seq_len * in_dim){
                memmove(in_d, in_d + in_dim, (size_t)((seq_len - 1) * in_dim) * sizeof(float));
                float *dst_row = in_d + (seq_len - 1) * in_dim;
                for(int64_t d = 0; d < in_dim; ++d) dst_row[d] = (d < copy_d) ? (0.5f * tanhf(last_h[d]) + 0.5f * tanhf(ctx[d])) : 0.0f;
            } else {
                const int64_t lim = (need < kv_dim) ? need : kv_dim;
                for(int64_t d = 0; d < lim; ++d) in_d[d] = 0.5f * tanhf(last_h[d]) + 0.5f * tanhf(ctx[d]);
            }
            if(temp_out) axon_tensor_free(gout);
        }
        double gus = (main_now_us() - gt0) / gen_steps;
        if(!quiet) printf("[generate] %d steps avg %.2f us/token | kv_cache_in_arena=%d\n", gen_steps, gus, kv->in_arena ? 1 : 0);
        free(ctx);
        axon_kv_cache_free(kv);
    } else {
        AxonTensor*out=axon_forward(inputs);
        if(!out){fprintf(stderr,"inference failed\n");axon_shutdown();axon_tensor_free(inp);axon_arena_destroy();axon_unload_weights_file();return 1;}
        double t0 = main_now_us();
        for(int it=0;it<iters;++it){
            if(out&&temp_out) axon_tensor_free(out);
            if(arena_mb>0) axon_arena_reset();
            out=axon_forward(inputs);
            if(!out){fprintf(stderr,"inference failed at %d\n",it);axon_shutdown();axon_tensor_free(inp);axon_arena_destroy();axon_unload_weights_file();return 1;}
        }
        double us = (main_now_us() - t0) / iters;
        int64_t flops=axon_get_total_flops(); double gflops=(us>0.0&&flops>0)?(double)flops/(us*1e3):0.0;
        AxonBrainStats bs; axon_get_brain_stats(&bs);
        if(!quiet){
            printf("[axon] iters=%d arena=%zu/%zu bytes\n",iters,axon_arena_used(),axon_arena_capacity());
            if(bs.total_regions > 0){
                double ram_saved = (bs.dense_f32_bytes > 0) ? 100.0 * (1.0 - (double)bs.paged_bytes_active / (double)bs.dense_f32_bytes) : 0.0;
                double fl_skip = (bs.dense_flops > 0) ? 100.0 * (double)(bs.dense_flops - bs.executed_flops) / (double)bs.dense_flops : 0.0;
                if(fl_skip < 0.0) fl_skip = 0.0;
                printf("[brain] active_regions=%lld/%lld (paged_regions=%lld/%lld) | ram_saved=%.1f%% | flops_skipped=%.1f%%\n",
                       (long long)bs.active_regions,(long long)bs.total_regions,(long long)bs.paged_regions,(long long)bs.unique_regions,ram_saved,fl_skip);
            }
            axon_print_tensor_n(out,max_print);
            printf("[axon] avg %.2f us | est flops %lld | ~%.4f GFLOP/s (estimate)\n",us,(long long)flops,gflops);
        }
        if(out&&temp_out) axon_tensor_free(out);
    }
    axon_shutdown(); axon_tensor_free(inp); axon_arena_destroy(); axon_unload_weights_file();
    int64_t live=axon_live_allocs();
    if(!quiet)printf("[axon] live heap tensors at exit: %lld %s\n",(long long)live,live==0?"(no leak)":"(LEAK!)");
    return live==0?0:1;
}
'''


V11_GPU_OPENCL_H = r'''#ifndef AXON_GPU_OPENCL_H
#define AXON_GPU_OPENCL_H
#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#  include <OpenCL/opencl.h>
#else
#  include <CL/cl.h>
#endif
#include "axon_runtime.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct AxonCLContext AxonCLContext;

AxonCLContext *axon_cl_create(const char *kernel_src, char *err_buf, size_t err_sz);
void axon_cl_destroy(AxonCLContext *ctx);
const char *axon_cl_device_name(const AxonCLContext *ctx);
cl_mem axon_cl_alloc_buf(AxonCLContext *ctx, size_t bytes, const void *host_init, int read_only);
int axon_cl_upload(AxonCLContext *ctx, cl_mem buf, size_t bytes, const void *src);
int axon_cl_download(AxonCLContext *ctx, cl_mem buf, size_t bytes, void *dst);
int axon_cl_upload_tensor(AxonCLContext *ctx, cl_mem buf, const AxonTensor *t);

int axon_cl_run_gemm(AxonCLContext *ctx, cl_mem A, cl_mem B, cl_mem bias, cl_mem C,
                     int64_t M, int64_t K, int64_t N, int act);
int axon_cl_run_neuro_mm(AxonCLContext *ctx, cl_mem X, cl_mem W, cl_mem S, cl_mem C,
                         int64_t M, int64_t K, int64_t N, int64_t R, int keep, int region_blocked);
int axon_cl_run_ew(AxonCLContext *ctx, cl_mem A, cl_mem B, cl_mem C,
                   int ndim, int64_t total, int mode,
                   int64_t cs0, int64_t cs1, int64_t cs2, int64_t cs3,
                   int64_t sa0, int64_t sa1, int64_t sa2, int64_t sa3,
                   int64_t sb0, int64_t sb1, int64_t sb2, int64_t sb3);
int axon_cl_run_unary(AxonCLContext *ctx, cl_mem A, cl_mem C, int64_t total, int act);
int axon_cl_run_softmax(AxonCLContext *ctx, cl_mem A, cl_mem C, int64_t rows, int64_t cols);
int axon_cl_run_layernorm(AxonCLContext *ctx, cl_mem X, cl_mem G, cl_mem B, cl_mem C, int64_t rows, int64_t cols);
int axon_cl_run_rmsnorm(AxonCLContext *ctx, cl_mem X, cl_mem G, cl_mem C, int64_t rows, int64_t cols);
int axon_cl_run_rope(AxonCLContext *ctx, cl_mem X, cl_mem C, int64_t S, int64_t D);
int axon_cl_run_attn_causal(AxonCLContext *ctx, cl_mem Q, cl_mem K, cl_mem V, cl_mem C, int64_t S, int64_t D);
int axon_cl_run_transpose(AxonCLContext *ctx, cl_mem A, cl_mem C, int64_t M, int64_t N);
int axon_cl_run_equilibrium(AxonCLContext *ctx, cl_mem X, cl_mem W, cl_mem B, cl_mem C, int64_t M, int64_t D);
int axon_cl_bench_matmul_src(const char *kernel_src, int M, int K, int N,
                             const float *A, const float *B, float *C,
                             int iters, double *out_ms, char *dev_name, size_t dev_name_sz);

#ifdef __cplusplus
}
#endif
#endif
'''


V11_RUNTIME_PATHS = r'''import os, platform, shutil, subprocess, sys
from dataclasses import dataclass
from pathlib import Path
from .errors import CodegenError
PKG = Path(__file__).resolve().parent

def _shared_lib_name():
    if os.name == "nt":
        return "axon_runtime.dll"
    if sys.platform == "darwin":
        return "libaxon_runtime.dylib"
    return "libaxon_runtime.so"

@dataclass(frozen=True)
class RuntimePaths:
    dir: Path; static_lib: Path; shared_lib: Path; main_obj: Path; core_obj: Path; header: Path

def runtime_dir():
    return Path(os.environ.get("AXON_RUNTIME_DIR", str(PKG / "runtime"))).resolve()

def tuning_header_path():
    rt = runtime_dir() / "axon_tuning.h"
    if rt.exists():
        return rt
    c = Path(os.environ.get("AXON_CACHE_DIR", str(Path.home() / ".cache" / "axon"))) / "axon_tuning.h"
    return c if c.exists() else None

def runtime_paths():
    d = runtime_dir()
    return RuntimePaths(d, d / "libaxon_runtime.a", d / _shared_lib_name(), d / "axon_main.o", d / "axon_runtime.o", d / "axon_runtime.h")

def _stamp(rt):
    return rt.dir / ".axon_tuning_stamp"

def _want_stamp():
    th = tuning_header_path()
    return f"{str(th) if th is not None else ''}|san={int(os.environ.get('AXON_SANITIZE','0')=='1')}|os={os.name}:{platform.machine().lower()}"

def _stale(rt):
    outs = [rt.static_lib, rt.shared_lib, rt.main_obj, rt.core_obj]
    if not all(p.exists() for p in outs):
        return True
    sp = _stamp(rt)
    have = sp.read_text().strip() if sp.exists() else ""
    if _want_stamp() != have:
        return True
    oldest = min(p.stat().st_mtime for p in outs)
    th = tuning_header_path()
    sources = list(rt.dir.glob("*.c")) + list(rt.dir.glob("*.h")) + ([th] if th is not None else [])
    return any(s.exists() and s.stat().st_mtime > oldest for s in sources)

def _run(cmd, cwd=None):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise CodegenError(f"command failed ({' '.join(cmd)}):\n{(r.stderr or r.stdout).strip()}")

def ensure_runtime(verbose=False, native=None, extra=None):
    from .tools import Inspector
    from .backend import _tool
    rt = runtime_paths()
    if not rt.dir.exists():
        raise CodegenError(f"runtime dir missing: {rt.dir}")
    if extra is None and not _stale(rt):
        return rt
    insp = Inspector.shared()
    if native is None:
        native = os.environ.get("AXON_NATIVE", "0") == "1"
    san = os.environ.get("AXON_SANITIZE", "0") == "1"
    cc = _tool("cc")
    ar = _tool("ar")
    cflags = ["-O1" if san else "-O3", "-std=c11", "-Wall", "-Wextra", "-Wno-unknown-pragmas", "-fno-math-errno"]
    if os.name != "nt":
        cflags += ["-fPIC", "-pthread"]
    if san:
        cflags += ["-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
    if native and platform.machine().lower() in ("x86_64", "amd64", "aarch64", "arm64"):
        cflags.append("-march=native")
    cflags.extend(insp.omp_cflags)
    th = tuning_header_path()
    if th is not None:
        cflags.append(f"-I{th.parent}")
    cflags.append(f"-I{rt.dir}")
    if extra:
        cflags.extend(extra.split())

    if verbose:
        print(f"[axon] building runtime ({os.name}/{platform.machine()}) in {rt.dir} (san={int(san)})", file=sys.stderr)

    _run([cc, *cflags, "-c", str(rt.dir / "axon_runtime.c"), "-o", str(rt.core_obj)])
    _run([cc, *cflags, "-c", str(rt.dir / "axon_main.c"), "-o", str(rt.main_obj)])
    if rt.static_lib.exists():
        rt.static_lib.unlink()
    _run([ar, "rcs", str(rt.static_lib), str(rt.core_obj)])

    sh_cmd = [cc, *cflags, "-shared", str(rt.core_obj)]
    if san:
        sh_cmd.append("-fsanitize=address,undefined")
    if os.name == "nt":
        sh_cmd.extend(["-Wl,--export-all-symbols", "-static-libgcc"])
    sh_cmd.extend(insp.omp_ldflags)
    sh_cmd.extend(["-lm", "-o", str(rt.shared_lib)])
    _run(sh_cmd)

    if extra is None:
        try:
            _stamp(rt).write_text(_want_stamp() + "\n")
        except OSError:
            pass
    return rt
'''


V11_AUTOCONFIG = r'''"""Cross-OS hardware inspector, auto-tuner, and backend selector for Axon v11.0."""
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from .runtime_paths import ensure_runtime, runtime_dir
from .tools import Evolver, Inspector


def detect_hardware():
    insp = Inspector.shared()
    from .backend.opencl_backend import OpenCLBackend
    cores = os.cpu_count() or 2
    opencl_info = OpenCLBackend.probe_device()
    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang") or ""
    has_iverilog = bool(shutil.which("iverilog") and shutil.which("vvp"))
    arch = platform.machine().lower()
    os_name = platform.system().lower()
    if opencl_info.get("is_gpu"):
        recommended_target = "gpu"
    else:
        recommended_target = "cpu"
    return {
        "os": os_name,
        "python_os": os.name,
        "arch": arch,
        "cores": cores,
        "memory_gb": insp.memory_gb,
        "simd": list(insp.features),
        "cc": cc,
        "omp_cflags": list(insp.omp_cflags),
        "omp_ldflags": list(insp.omp_ldflags),
        "llvmlite": insp.llvmlite_version,
        "opencl": opencl_info,
        "has_iverilog": has_iverilog,
        "recommended_target": recommended_target,
    }


def auto_configure(tune=True, dim=128, verbose=True):
    hw = detect_hardware()
    th = runtime_dir() / "axon_tuning.h"
    tuned = None
    if tune and not th.exists():
        if hw["cc"]:
            try:
                tuned = Evolver.evolve(generations=2, pop_size=4, mat_dim=dim, write_header=True)
            except Exception:
                tuned = None
        if tuned is None:
            tile = 64 if hw["cores"] >= 4 else 32
            par_min = 32768 if hw["cores"] >= 4 else 65536
            try:
                th.write_text(f"#define AXON_TILE {tile}\n#define AXON_PAR_MIN {par_min}\n")
                tuned = {"best_tile": tile, "best_par_min": par_min, "header": str(th)}
            except OSError:
                pass
    rt = ensure_runtime(verbose=verbose)
    profile = {
        "axon_version": "11.0.0",
        "hardware": hw,
        "tuning": tuned,
        "runtime_dir": str(rt.dir),
        "shared_lib": str(rt.shared_lib),
    }
    prof_path = runtime_dir() / "axon_profile.json"
    try:
        prof_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    except OSError:
        pass
    if verbose:
        print(f"[autoconfig] OS={hw['os']}/{hw['arch']} cores={hw['cores']} SIMD={hw['simd'] or ['generic']} OMP={bool(hw['omp_cflags'])} OpenCL={hw['opencl'].get('device','none')} -> target={hw['recommended_target']}")
    return profile
'''


V11_DEPLOY = r'''"""Enterprise single-command model optimizer, cross-OS builder, and accuracy verifier."""
import json
import os
import subprocess
import sys
from pathlib import Path
from .autoconfig import auto_configure
from .backend import Baker, LLVMBackend, OpenCLBackend, TPUBackend
from .blob import BlobPacker
from .errors import AxonError
from .frontend import Lexer, Parser, Semantic
from .ir import IRBuilder, OpKind
from .passes import Pipeline
from .reference import T, evaluate
from .tools import Converter


def _exe_ext():
    return ".exe" if os.name == "nt" else ".bin"


def _shared_ext():
    if os.name == "nt":
        return ".dll"
    if sys.platform == "darwin":
        return ".dylib"
    return ".so"


def load_any_model(source, quant="fp4", brain=True, brain_ratio=0.5, transpose=False, fusion=True):
    src_path = Path(source)
    if not src_path.exists():
        raise FileNotFoundError(str(source))
    use_ternary = (quant == "ternary")
    use_quant = (quant == "fp4")

    if src_path.suffix == ".safetensors":
        g = Converter.graph_from_safetensors(str(src_path), transpose=transpose)
    elif src_path.suffix == ".json":
        with open(src_path, encoding="utf-8") as fh:
            model = json.load(fh)
        ax_src = Converter.generate(model)
        ast = Parser(Lexer(ax_src, str(src_path)).tokenize(), str(src_path)).parse()
        Semantic(str(src_path)).analyze(ast)
        g = IRBuilder().build(ast)
    else:
        with open(src_path, encoding="utf-8") as fh:
            ax_src = fh.read()
        ast = Parser(Lexer(ax_src, str(src_path)).tokenize(), str(src_path)).parse()
        Semantic(str(src_path)).analyze(ast)
        g = IRBuilder().build(ast)

    raw_f32_bytes = sum(len(b.data) * 4 for b in g.weights.values())
    g = Pipeline(fusion=fusion, quant=use_quant, ternary=use_ternary, brain=brain, brain_ratio=brain_ratio).run(g)
    return g, raw_f32_bytes


def deploy_model(source, out_dir, quant="fp4", brain=True, brain_ratio=0.5, transpose=False, emit_gpu="auto", emit_tpu=True, verify=True):
    profile = auto_configure(tune=True, verbose=True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(source).stem

    g, raw_f32_bytes = load_any_model(source, quant=quant, brain=brain, brain_ratio=brain_ratio, transpose=transpose)
    axn_bytes = len(BlobPacker.pack(g.weights))

    exe_path = str(out / (stem + _exe_ext()))
    obj_path = str(out / (stem + ".tmp.o"))
    try:
        LLVMBackend().compile(g, obj_path)
        Baker().bake([obj_path], g.weights, exe_path, external_weights=True)
    finally:
        if os.path.exists(obj_path):
            os.remove(obj_path)

    so_name = ("lib" + stem if os.name != "nt" else stem) + _shared_ext()
    so_path = str(out / so_name)
    obj_so = str(out / (stem + ".so.tmp.o"))
    try:
        LLVMBackend().compile(g, obj_so)
        Baker().bake([obj_so], g.weights, so_path, emit_shared=True, external_weights=True)
    finally:
        if os.path.exists(obj_so):
            os.remove(obj_so)

    artifacts = {
        "executable": exe_path,
        "weights_mmap": exe_path + ".axn3",
        "shared_library": so_path,
        "shared_weights": so_path + ".axn3",
        "python_client": str(out / (Path(so_path).stem + "_client.py")),
    }

    if emit_tpu:
        g_tpu, _ = load_any_model(source, quant="none", brain=False, transpose=transpose)
        mlir_path = str(out / (stem + ".mlir"))
        TPUBackend().compile(g_tpu, mlir_path)
        artifacts["tpu_mlir"] = mlir_path

    do_gpu = (emit_gpu is True) or (emit_gpu == "auto" and OpenCLBackend.is_available())
    if do_gpu:
        gpu_path = str(out / (stem + "_gpu" + _exe_ext()))
        OpenCLBackend().build_binary(g, gpu_path, external_weights=True)
        artifacts["gpu_executable"] = gpu_path
        artifacts["gpu_opencl_kernels"] = gpu_path + ".cl"

    verify_info = {"verified": False}
    if verify:
        in_node = g.get(g.inputs[0])
        in_shape = tuple(in_node.shape)
        need = 1
        for d in in_shape:
            need *= d
        seed = [1.0, 0.5, -1.0, 2.0]
        in_tensor = T(in_shape, [seed[i % 4] for i in range(need)])
        ref_out = evaluate(g, [in_tensor]).data

        r = subprocess.run([exe_path, "--iters", "2", "--max-print", "1000000"], capture_output=True, text=True)
        if r.returncode != 0:
            raise AxonError(f"deployed binary verification failed:\n{r.stdout}\n{r.stderr}")
        if "no leak" not in r.stdout:
            raise AxonError("deployed binary reported memory leak")
        got_vals = []
        inside = False
        for line in r.stdout.splitlines():
            s = line.strip()
            if s.endswith("= ["):
                inside = True
                continue
            if inside and s == "]":
                break
            if inside:
                try:
                    got_vals.append(float(s))
                except ValueError:
                    pass
        if len(got_vals) != len(ref_out):
            raise AxonError(f"deployed output length {len(got_vals)} != {len(ref_out)}")
        max_err = max(abs(a - b) for a, b in zip(got_vals, ref_out)) if got_vals else 0.0
        verify_info = {"verified": True, "max_abs_err": max_err, "no_leak": True}

    manifest = {
        "axon_version": "11.0.0",
        "source": str(source),
        "quantization": quant,
        "brain_routing": bool(brain and any(n.op == OpKind.NEURO_MATMUL for n in g.nodes)),
        "raw_f32_weight_bytes": raw_f32_bytes,
        "axn3_compressed_bytes": axn_bytes,
        "compression_ratio_pct": round((1.0 - axn_bytes / raw_f32_bytes) * 100.0, 2) if raw_f32_bytes else 0.0,
        "hardware": profile["hardware"],
        "artifacts": artifacts,
        "verification": verify_info,
    }
    man_path = out / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
'''


def patch_tools(s):
    s = s.replace(
        'f.write("#include <omp.h>\\nint main(void){return omp_get_max_threads()<1;}\\n")',
        'f.write("#include <omp.h>\\nint main(void){return omp_get_max_threads()<1;}\\n")',
    )
    old_probe = (
        '            for cf,lf in cands:\n'
        '                try:\n'
        '                    if subprocess.run([cc,*cf,src,*lf,"-o",out],capture_output=True).returncode==0: return cf,lf\n'
        '                except (FileNotFoundError,OSError): continue'
    )
    new_probe = (
        '            for cf,lf in cands:\n'
        '                try:\n'
        '                    extra_win = ["-static-libgcc"] if os.name == "nt" else []\n'
        '                    if subprocess.run([cc,*cf,src,*lf,*extra_win,"-o",out],capture_output=True).returncode==0:\n'
        '                        if subprocess.run([out],capture_output=True).returncode==0: return cf,lf\n'
        '                except (FileNotFoundError,OSError): continue'
    )
    if old_probe in s:
        s = s.replace(old_probe, new_probe)
    s = s.replace(
        'base="-O3 -fPIC -std=c11 -Wall -Wextra -Wno-unknown-pragmas -fno-math-errno -pthread"',
        'base="-O3 -std=c11 -Wall -Wextra -Wno-unknown-pragmas -fno-math-errno" + ("" if os.name=="nt" else " -fPIC -pthread")',
    )
    old_ev_head = (
        'EVOLVE_SRC=r"""#define _POSIX_C_SOURCE 200809L\n'
        '#include <stdio.h>\n'
        '#include <stdlib.h>\n'
        '#include <time.h>\n'
    )
    new_ev_head = (
        'EVOLVE_SRC=r"""#ifndef _WIN32\n'
        '#define _POSIX_C_SOURCE 200809L\n'
        '#endif\n'
        '#include <stdio.h>\n'
        '#include <stdlib.h>\n'
        '#include <time.h>\n'
        '#ifdef _WIN32\n'
        '#define WIN32_LEAN_AND_MEAN\n'
        '#include <windows.h>\n'
        'static double ev_ms(void){ LARGE_INTEGER f,n; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&n); return (double)n.QuadPart*1000.0/(double)f.QuadPart; }\n'
        '#else\n'
        'static double ev_ms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (double)t.tv_sec*1e3+(double)t.tv_nsec/1e6; }\n'
        '#endif\n'
    )
    if old_ev_head in s and "static double ev_ms" not in s:
        s = s.replace(old_ev_head, new_ev_head)
        s = s.replace('struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);', 'double t0 = ev_ms();')
        s = s.replace(
            'clock_gettime(CLOCK_MONOTONIC,&t1);\n        double ms=(t1.tv_sec-t0.tv_sec)*1e3+(t1.tv_nsec-t0.tv_nsec)/1e6;',
            'double ms = ev_ms() - t0;',
        )
    s = s.replace(
        'f=["-O3","-std=c11","-fno-math-errno","-pthread",*insp.omp_cflags]',
        'f=["-O3","-std=c11","-fno-math-errno",*([] if os.name=="nt" else ["-pthread"]),*insp.omp_cflags]',
    )
    return s


def patch_backend_init(s):
    if "_default_llvm_triple" not in s:
        helper = (
            'def _default_llvm_triple():\n'
            '    t=llvm.get_default_triple()\n'
            '    if os.name=="nt":\n'
            '        cc=_tool("cc")\n'
            '        try:\n'
            '            out=subprocess.check_output([cc,"-dumpmachine"],text=True,stderr=subprocess.DEVNULL).strip().lower()\n'
            '            if "mingw" in out: return out\n'
            '        except Exception: pass\n'
            '    return t\n'
        )
        s = s.replace("def _tool(name,target=\"\"):", helper + "\ndef _tool(name,target=\"\"):")
        s = s.replace("self.triple=target_triple or llvm.get_default_triple();", "self.triple=target_triple or _default_llvm_triple();")
    s = s.replace(
        'if target and target not in ("native","gpu","tpu"):',
        'if target and target not in ("native","gpu","tpu","auto"):',
    )
    s = s.replace(
        'f=["-O1" if _san_flags() else "-O3","-fPIC","-std=c11","-Wall","-Wextra","-Wno-unknown-pragmas","-fno-math-errno","-pthread"]',
        'f=["-O1" if _san_flags() else "-O3","-std=c11","-Wall","-Wextra","-Wno-unknown-pragmas","-fno-math-errno"] + ([] if os.name=="nt" else ["-fPIC","-pthread"])',
    )
    s = s.replace(
        'cmd=[_tool("cc",target),"-O1" if _san_flags() else "-O3","-pthread",*_san_flags(),*omp,*objects]',
        'win_f=["-static-libgcc"] if os.name=="nt" else ["-pthread"]\n        cmd=[_tool("cc",target),"-O1" if _san_flags() else "-O3",*win_f,*_san_flags(),*omp,*objects]',
    )
    s = s.replace(
        'cmd=[_tool("cc",target),"-O1" if _san_flags() else "-O3","-shared","-fPIC","-pthread",*_san_flags(),*omp,*objects]',
        'win_f=["-Wl,--export-all-symbols","-static-libgcc"] if os.name=="nt" else ["-fPIC","-pthread"]\n        cmd=[_tool("cc",target),"-O1" if _san_flags() else "-O3","-shared",*win_f,*_san_flags(),*omp,*objects]',
    )
    if "os.add_dll_directory" not in s:
        s = s.replace(
            '        \'        here=os.path.dirname(os.path.abspath(__file__)); self.so_path=so_path or os.path.join(here,"%s"); self.lib=ctypes.CDLL(os.path.abspath(self.so_path))\\n\'',
            '        \'        here=os.path.dirname(os.path.abspath(__file__)); self.so_path=so_path or os.path.join(here,"%s")\\n\'\n'
            '        \'        if os.name=="nt" and hasattr(os,"add_dll_directory"): os.add_dll_directory(os.path.dirname(os.path.abspath(self.so_path)))\\n\'\n'
            '        \'        self.lib=ctypes.CDLL(os.path.abspath(self.so_path))\\n\'',
        )
    old_sh = (
        '            if emit_shared:\n'
        '                so=output if output.endswith(".so") else output+".so"\n'
        '                Linker.link_shared([*objects,wobj,str(rt.core_obj)],so,self.target,self.native,extra_libs=extra_libs)\n'
        '                open(so[:-3]+"_client.py","w").write(self.PY_CLIENT%os.path.basename(so))\n'
        '                return so'
    )
    new_sh = (
        '            if emit_shared:\n'
        '                sh_ext = ".dll" if os.name=="nt" else (".dylib" if os.sys.platform=="darwin" else ".so")\n'
        '                so = output if output.endswith((".so",".dll",".dylib")) else output + sh_ext\n'
        '                Linker.link_shared([*objects,wobj,str(rt.core_obj)],so,self.target,self.native,extra_libs=extra_libs)\n'
        '                stem = os.path.splitext(so)[0]\n'
        '                open(stem+"_client.py","w").write(self.PY_CLIENT%os.path.basename(so))\n'
        '                return so'
    )
    if old_sh in s:
        s = s.replace(old_sh, new_sh)
    s = s.replace('if os.sys.platform=="darwin"', 'if sys.platform=="darwin"')
    if "import os, platform, shutil, subprocess, tempfile" in s and "sys" not in s.splitlines()[0]:
        s = s.replace("import os, platform, shutil, subprocess, tempfile", "import os, platform, shutil, subprocess, sys, tempfile")
    return s


def patch_opencl_backend(s):
    if "import os, subprocess, tempfile" in s:
        s = s.replace("import os, subprocess, tempfile", "import os, subprocess, sys, tempfile")
    old_chk = '''    @staticmethod
    def is_available():
        from . import _tool
        rt = runtime_paths()
        src = rt.dir / "axon_gpu_opencl.c"
        if not src.exists():
            return False
        with tempfile.TemporaryDirectory(prefix="axon_cl_chk_") as tmp:
            exe = os.path.join(tmp, "chk")
            cfile = os.path.join(tmp, "chk.c")
            with open(cfile, "w") as f:
                f.write(
                    '#define CL_TARGET_OPENCL_VERSION 120\\n'
                    '#define CL_USE_DEPRECATED_OPENCL_1_2_APIS\\n'
                    '#include <CL/cl.h>\\n'
                    'int main(void){ cl_uint n=0; if(clGetPlatformIDs(0,NULL,&n)!=CL_SUCCESS || n==0) return 1; return 0; }\\n'
                )
            r = subprocess.run([_tool("cc"), cfile, "-lOpenCL", "-o", exe], capture_output=True)
            if r.returncode != 0:
                return False
            return subprocess.run([exe], capture_output=True).returncode == 0'''
    new_chk = '''    @staticmethod
    def opencl_libs():
        if sys.platform == "darwin":
            return ["-framework", "OpenCL"]
        return ["-lOpenCL"]

    @staticmethod
    def probe_device():
        from . import _tool
        rt = runtime_paths()
        if not (rt.dir / "axon_gpu_opencl.c").exists():
            return {"available": False, "is_gpu": False, "device": "none"}
        with tempfile.TemporaryDirectory(prefix="axon_cl_chk_") as tmp:
            exe = os.path.join(tmp, "chk.exe" if os.name == "nt" else "chk")
            cfile = os.path.join(tmp, "chk.c")
            with open(cfile, "w") as f:
                f.write(
                    '#define CL_TARGET_OPENCL_VERSION 120\\n'
                    '#define CL_USE_DEPRECATED_OPENCL_1_2_APIS\\n'
                    '#include <stdio.h>\\n'
                    '#ifdef __APPLE__\\n#include <OpenCL/opencl.h>\\n#else\\n#include <CL/cl.h>\\n#endif\\n'
                    'int main(void){\\n'
                    '  cl_uint np=0; if(clGetPlatformIDs(0,NULL,&np)!=CL_SUCCESS||np==0) return 1;\\n'
                    '  cl_platform_id p; clGetPlatformIDs(1,&p,NULL); cl_device_id d; cl_uint nd=0;\\n'
                    '  int is_gpu = (clGetDeviceIDs(p,CL_DEVICE_TYPE_GPU,1,&d,&nd)==CL_SUCCESS && nd>0);\\n'
                    '  if(!is_gpu && (clGetDeviceIDs(p,CL_DEVICE_TYPE_ALL,1,&d,&nd)!=CL_SUCCESS || nd==0)) return 1;\\n'
                    '  char nm[256]="OpenCL"; clGetDeviceInfo(d,CL_DEVICE_NAME,255,nm,NULL);\\n'
                    '  printf("%d|%s\\n", is_gpu, nm); return 0;\\n'
                    '}\\n'
                )
            extra = ["-Wno-deprecated-declarations"] if sys.platform == "darwin" else []
            r = subprocess.run([_tool("cc"), *extra, cfile, *OpenCLBackend.opencl_libs(), "-o", exe], capture_output=True)
            if r.returncode != 0:
                return {"available": False, "is_gpu": False, "device": "none"}
            rr = subprocess.run([exe], capture_output=True, text=True)
            if rr.returncode != 0:
                return {"available": False, "is_gpu": False, "device": "none"}
            parts = rr.stdout.strip().split("|", 1)
            return {"available": True, "is_gpu": parts[0] == "1", "device": parts[1] if len(parts) > 1 else "OpenCL"}

    @staticmethod
    def is_available():
        return OpenCLBackend.probe_device()["available"]'''
    if old_chk in s:
        s = s.replace(old_chk, new_chk)
    s = s.replace('extra_libs=["-lOpenCL"]', 'extra_libs=OpenCLBackend.opencl_libs()')
    return s


def patch_gpu_opencl_c(s):
    if "axon_cl_now_ms" not in s:
        s = s.replace("#include <time.h>", "#include <time.h>\n#ifdef _WIN32\n#define WIN32_LEAN_AND_MEAN\n#include <windows.h>\nstatic double axon_cl_now_ms(void){ LARGE_INTEGER f,n; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&n); return (double)n.QuadPart*1000.0/(double)f.QuadPart; }\n#else\nstatic double axon_cl_now_ms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (double)t.tv_sec*1e3+(double)t.tv_nsec/1e6; }\n#endif")
        s = s.replace(
            "        struct timespec t0, t1;\n        clock_gettime(CLOCK_MONOTONIC, &t0);\n        rc = axon_cl_run_gemm(ctx, dA, dB, NULL, dC, M, K, N, 0);\n        if (rc == AXON_OK) clFinish(ctx->queue);\n        clock_gettime(CLOCK_MONOTONIC, &t1);\n        double ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;",
            "        double t0 = axon_cl_now_ms();\n        rc = axon_cl_run_gemm(ctx, dA, dB, NULL, dC, M, K, N, 0);\n        if (rc == AXON_OK) clFinish(ctx->queue);\n        double ms = axon_cl_now_ms() - t0;",
        )
    return s


def patch_tpu_backend(s):
    old_fn = (
        '    def _bcast(self, var, src_shape, dst_shape, elem="f32"):\n'
        '        src_shape = tuple(src_shape)\n'
        '        dst_shape = tuple(dst_shape)\n'
        '        if src_shape == dst_shape:\n'
        '            return var\n'
        '        out = self._v("bc")\n'
        '        if len(src_shape) == 0:\n'
        '            self._emit(f"{out} = stablehlo.broadcast_in_dim {var}, dims = [] : ({_ty(src_shape, elem)}) -> {_ty(dst_shape, elem)}")\n'
        '            return out\n'
        '        off = len(dst_shape) - len(src_shape)\n'
        '        dims = ", ".join(str(off + i) for i in range(len(src_shape)))\n'
        '        self._emit(f"{out} = stablehlo.broadcast_in_dim {var}, dims = [{dims}] : ({_ty(src_shape, elem)}) -> {_ty(dst_shape, elem)}")\n'
        '        return out'
    )
    new_fn = (
        '    def _bcast(self, var, src_shape, dst_shape, elem="f32", dims=None):\n'
        '        src_shape = tuple(src_shape)\n'
        '        dst_shape = tuple(dst_shape)\n'
        '        if src_shape == dst_shape:\n'
        '            return var\n'
        '        out = self._v("bc")\n'
        '        if len(src_shape) == 0:\n'
        '            self._emit(f"{out} = stablehlo.broadcast_in_dim {var}, dims = [] : ({_ty(src_shape, elem)}) -> {_ty(dst_shape, elem)}")\n'
        '            return out\n'
        '        if dims is None:\n'
        '            off = len(dst_shape) - len(src_shape)\n'
        '            dims = [off + i for i in range(len(src_shape))]\n'
        '        dims_str = ", ".join(str(d) for d in dims)\n'
        '        self._emit(f"{out} = stablehlo.broadcast_in_dim {var}, dims = [{dims_str}] : ({_ty(src_shape, elem)}) -> {_ty(dst_shape, elem)}")\n'
        '        return out'
    )
    if old_fn in s:
        s = s.replace(old_fn, new_fn)
    for call_old, call_new in (
        ("mxb = self._bcast(mx, rsh, shape)", "mxb = self._bcast(mx, rsh, shape, dims=list(range(len(rsh))))"),
        ("smb = self._bcast(sm, rsh, shape)", "smb = self._bcast(sm, rsh, shape, dims=list(range(len(rsh))))"),
        ("meanb = self._bcast(mean, rsh, shape)", "meanb = self._bcast(mean, rsh, shape, dims=list(range(len(rsh))))"),
        ("rsb = self._bcast(rs, rsh, shape)", "rsb = self._bcast(rs, rsh, shape, dims=list(range(len(rsh))))"),
    ):
        s = s.replace(call_old, call_new)
    return s


def patch_cli(s):
    if "from .autoconfig import auto_configure" not in s:
        s = s.replace(
            "from .errors import AxonError",
            "from .errors import AxonError\nfrom .autoconfig import auto_configure\nfrom .deploy import deploy_model",
        )
    if "def cmd_autoconfig" not in s:
        extra_cmds = textwrap.dedent('''\
            def cmd_autoconfig(args):
                prof = auto_configure(tune=not args.no_tune, dim=args.dim, verbose=True)
                print(json.dumps(prof, indent=2))
                return 0
            def cmd_deploy(args):
                q = "ternary" if args.ternary else ("none" if args.no_quant else "fp4")
                man = deploy_model(
                    args.source, args.output, quant=q, brain=not args.no_brain,
                    brain_ratio=args.brain_ratio, transpose=args.transpose,
                    emit_gpu="auto" if args.gpu else False, emit_tpu=not args.no_tpu, verify=not args.no_verify,
                )
                print("=== AXON v11.0 DEPLOYMENT COMPLETE ===")
                print(f"OS / Arch   : {man['hardware']['os']} / {man['hardware']['arch']}")
                print(f"Weights     : {man['raw_f32_weight_bytes']} B -> {man['axn3_compressed_bytes']} B ({man['compression_ratio_pct']:.1f}% smaller, quant={q}, brain={man['brain_routing']})")
                print(f"Verified    : {man['verification']}")
                print(f"Artifacts   : {args.output}")
                return 0
        ''')
        s = s.replace("def cmd_doctor(args):", extra_cmds + "def cmd_doctor(args):")
    s = s.replace(
        '    if args.target=="tpu":',
        '    if args.target=="auto":\n        args.target = auto_configure(tune=False, verbose=False)["hardware"]["recommended_target"]\n        if args.target=="cpu": args.target=""\n    if args.target=="tpu":',
    )
    s = s.replace(
        'pb.add_argument("-o","--output",default="model.bin");',
        'pb.add_argument("-o","--output",default=("model.exe" if os.name=="nt" else "model.bin"));',
    )
    s = s.replace(
        'pc.add_argument("-o","--output",default="model.bin");',
        'pc.add_argument("-o","--output",default=("model.exe" if os.name=="nt" else "model.bin"));',
    )
    if 'sp.add_parser("autoconfig")' not in s:
        add_sp = (
            '    pa=sp.add_parser("autoconfig"); pa.add_argument("--dim",type=int,default=128); pa.add_argument("--no-tune",action="store_true"); pa.set_defaults(func=cmd_autoconfig)\n'
            '    pd=sp.add_parser("deploy"); pd.add_argument("source"); pd.add_argument("-o","--output",default="dist_axon")\n'
            '    pd.add_argument("--ternary",action="store_true"); pd.add_argument("--no-quant",action="store_true")\n'
            '    pd.add_argument("--no-brain",action="store_true"); pd.add_argument("--brain-ratio",type=float,default=0.5)\n'
            '    pd.add_argument("--transpose",action="store_true"); pd.add_argument("--gpu",action="store_true")\n'
            '    pd.add_argument("--no-tpu",action="store_true"); pd.add_argument("--no-verify",action="store_true"); pd.set_defaults(func=cmd_deploy)\n'
        )
        s = s.replace('    sp.add_parser("doctor")', add_sp + '    sp.add_parser("doctor")')
    return s


def patch_tests(s):
    s = s.replace('import ctypes, importlib.util, json, os, shutil, struct, subprocess, tempfile, unittest',
                  'import ctypes, importlib.util, json, os, shutil, struct, subprocess, sys, tempfile, unittest')
    s = s.replace('exe=os.path.join(self.tmp,name+".bin")',
                  'exe=os.path.join(self.tmp,name+(".exe" if os.name=="nt" else ".bin"))')
    s = s.replace('exe=os.path.join(self.tmp,f"gpu_{fname}.bin")',
                  'exe=os.path.join(self.tmp,f"gpu_{fname}"+(".exe" if os.name=="nt" else ".bin"))')
    s = s.replace('["python3","-m","axon.cli"', '[sys.executable,"-m","axon.cli"')
    s = s.replace(
        'so=os.path.join(self.tmp,"llama_lib.so"); obj=os.path.join(self.tmp,"llama_lib.o")',
        'sh_ext=".dll" if os.name=="nt" else (".dylib" if sys.platform=="darwin" else ".so")\n'
        '        so=os.path.join(self.tmp,"llama_lib"+sh_ext); obj=os.path.join(self.tmp,"llama_lib.o")',
    )
    if "test_autoconfig_and_enterprise_deploy" not in s:
        new_test = textwrap.indent(textwrap.dedent('''\
            def test_autoconfig_and_enterprise_deploy(self):
                if os.environ.get("AXON_SANITIZE","0")=="1": return
                from axon.autoconfig import detect_hardware
                from axon.deploy import deploy_model
                hw = detect_hardware()
                self.assertIn(hw["recommended_target"], ("cpu", "gpu"))
                out_dir = os.path.join(self.tmp, "enterprise_dist")
                man = deploy_model(os.path.join(EX, "mini_llm.axon"), out_dir, quant="fp4", brain=True, emit_gpu=False)
                self.assertTrue(man["verification"]["verified"])
                self.assertTrue(man["verification"]["no_leak"])
                self.assertLessEqual(man["verification"]["max_abs_err"], 5e-4)
                self.assertTrue(os.path.exists(man["artifacts"]["executable"]))
                self.assertTrue(os.path.exists(man["artifacts"]["weights_mmap"]))
                self.assertTrue(os.path.exists(man["artifacts"]["shared_library"]))
                self.assertTrue(os.path.exists(man["artifacts"]["tpu_mlir"]))
                cli_py = man["artifacts"]["python_client"]
                spec = importlib.util.spec_from_file_location("mini_llm_client", cli_py)
                mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
                with mod.AxonModel(man["artifacts"]["shared_library"], man["artifacts"]["shared_weights"]) as m:
                    seed = [1.0, 0.5, -1.0, 2.0]
                    res = m.infer([seed[i % 4] for i in range(32)])
                    self.assertEqual(len(res), 32)
        '''), "    ")
        s = s.replace('if __name__=="__main__": unittest.main()', new_test + '\nif __name__=="__main__": unittest.main()')
    return s


def patch_bench_chart(s):
    s = s.replace(
        '    finally:\n        if os.path.exists(axn_path):\n            os.remove(axn_path)',
        '    finally:\n        lib.axon_unload_weights_file()\n        if os.path.exists(axn_path):\n            os.remove(axn_path)',
    )
    s = s.replace(
        '    cl_so = os.path.join(cl_so_dir, "libaxon_bench_cl.so")',
        '    sh_ext = ".dll" if os.name == "nt" else (".dylib" if sys.platform == "darwin" else ".so")\n'
        '    cl_so = os.path.join(cl_so_dir, "libaxon_bench_cl" + sh_ext)',
    )
    return s


def apply_v11():
    unpack_v10_if_needed()
    print("[v11] Upgrading codebase to Axon v11.0 (Cross-OS + AutoConfig + Enterprise Deploy) ...")
    write_file("axon/__init__.py", '__version__ = "11.0.0"\n')
    patch_file("pyproject.toml", lambda s: s.replace('version = "10.0.0"', 'version = "11.0.0"'))
    write_file("axon/runtime/axon_runtime.h", V11_RUNTIME_H)
    write_file("axon/runtime/axon_runtime.c", V11_RUNTIME_C)
    write_file("axon/runtime/axon_main.c", V11_MAIN_C)
    write_file("axon/runtime/axon_gpu_opencl.h", V11_GPU_OPENCL_H)
    write_file("axon/runtime_paths.py", V11_RUNTIME_PATHS)
    write_file("axon/autoconfig.py", V11_AUTOCONFIG)
    write_file("axon/deploy.py", V11_DEPLOY)
    patch_file("axon/tools.py", patch_tools)
    patch_file("axon/backend/__init__.py", patch_backend_init)
    patch_file("axon/backend/opencl_backend.py", patch_opencl_backend)
    patch_file("axon/runtime/axon_gpu_opencl.c", patch_gpu_opencl_c)
    patch_file("axon/backend/tpu_backend.py", patch_tpu_backend)
    patch_file("axon/cli.py", patch_cli)
    patch_file("tests/test_axon.py", patch_tests)
    patch_file("tests/bench_chart.py", patch_bench_chart)


def main():
    apply_v11()
    print("[v11] Auto-configuring hardware & compiling C runtime ...")
    if os.path.exists("axon/runtime/.axon_tuning_stamp"):
        os.remove("axon/runtime/.axon_tuning_stamp")
    from axon.autoconfig import auto_configure
    auto_configure(tune=False, verbose=True)
    print("[v11] Running Cross-OS unit & TPU MLIR test suite ...")
    r = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"])
    if r.returncode != 0:
        sys.exit(r.returncode)
    print("\n[COMPLETED] Axon v11.0 ready on", sys.platform)


if __name__ == "__main__":
    main()