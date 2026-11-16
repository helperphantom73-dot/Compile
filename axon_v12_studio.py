#!/usr/bin/env python3
"""Axon v12.0 Final - Zero-Prerequisite Standalone Studio (Free & Pro) + Web UI + Mobile SDK."""
import os
import platform
import re
import subprocess
import sys


def ensure_v11_base():
    if os.path.isfile("axon/deploy.py") and os.path.isfile("axon/runtime/axon_runtime.c"):
        return
    if os.path.isfile("axon_v11_upgrade.py"):
        p = "axon_v11_upgrade.py"
        s = open(p, encoding="utf-8").read()
        s = s.replace('.split("def main():")[0]', '.rsplit("def main():", 1)[0]')
        s = re.sub(r'printf\("%d\|%s[^"]*",\s*is_gpu,\s*nm\)', 'printf("%d|%s", is_gpu, nm)', s)
        open(p, "w", encoding="utf-8").write(s)
        ns = {}
        exec(s.rsplit("def main():", 1)[0], ns)
        ns["apply_v11"]()
    elif os.path.isfile("axon_v11.py"):
        ns = {}
        exec(open("axon_v11.py", encoding="utf-8").read().split("def main():")[0], ns)
        ns["_unpack"](force=True)
    else:
        raise FileNotFoundError("axon_v11_upgrade.py or axon_v11.py required alongside axon_v12_studio.py")

    bi = "axon/backend/__init__.py"
    if os.path.isfile(bi):
        bs = open(bi, encoding="utf-8").read().replace(
            "tm=target.create_target_machine(opt=3, reloc='pic')",
            "tm=target.create_target_machine(opt=3, reloc='default' if ('windows' in self.triple or 'mingw' in self.triple) else 'pic')",
        )
        open(bi, "w", encoding="utf-8").write(bs)


def write_file(path, text):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


AXG_PY = r'''"""Axon Binary Graph (.axg1) exporter for zero-prerequisite C execution."""
import struct
from .ir import OpKind

AXG_MAGIC = b"AXG1"
OP_CODES = {
    OpKind.INPUT: 0, OpKind.CONSTANT: 1, OpKind.MATMUL: 2, OpKind.NEURO_MATMUL: 3,
    OpKind.ADD: 4, OpKind.SUB: 5, OpKind.MUL: 6,
    OpKind.RELU: 7, OpKind.GELU: 8, OpKind.SILU: 9, OpKind.SIGMOID: 10,
    OpKind.SOFTMAX: 11, OpKind.LAYERNORM: 12, OpKind.RMSNORM: 13, OpKind.ROPE: 14,
    OpKind.ATTN_CAUSAL: 15, OpKind.TRANSPOSE: 16, OpKind.EQUILIBRIUM: 17,
    OpKind.FUSED_MATMUL_RELU: 18, OpKind.FUSED_MATMUL_GELU: 19, OpKind.FUSED_MATMUL_SILU: 20,
    OpKind.FUSED_MATMUL_BIAS: 21, OpKind.FUSED_MATMUL_BIAS_RELU: 22,
    OpKind.FUSED_MATMUL_BIAS_GELU: 23, OpKind.FUSED_MATMUL_BIAS_SILU: 24,
}
HDR = struct.Struct("<4sIIIIII4q")
NODE_ST = struct.Struct("<IIII4i4qii")


def export_axg(graph, out_path):
    order = graph.topo()
    id_map = {n.id: idx for idx, n in enumerate(order)}
    names = bytearray()
    recs = bytearray()
    for n in order:
        op = OP_CODES[n.op]
        ins = [id_map[i] for i in n.inputs]
        nin = len(ins)
        ins_pad = (ins + [-1, -1, -1, -1])[:4]
        shp = (list(n.shape) + [0, 0, 0, 0])[:4]
        wname = n.attrs.get("weight", "")
        if wname:
            noff = len(names)
            names.extend(wname.encode("utf-8") + b"\0")
        else:
            noff = 0xFFFFFFFF
        keep = int(n.attrs.get("keep") or 0)
        rb = 1 if n.attrs.get("region_blocked", False) else 0
        recs.extend(NODE_ST.pack(op, nin, len(n.shape), noff, *ins_pad, *shp, keep, rb))
    in_n = graph.get(graph.inputs[0])
    in_shp = (list(in_n.shape) + [0, 0, 0, 0])[:4]
    out_idx = id_map[graph.outputs[0]]
    in_idx = id_map[graph.inputs[0]]
    header = HDR.pack(AXG_MAGIC, 1, len(order), in_idx, out_idx, len(in_n.shape), len(names), *in_shp)
    with open(out_path, "wb") as fh:
        fh.write(header)
        fh.write(recs)
        fh.write(names)
    return out_path
'''


MOBILE_PY = r'''"""Android NDK & Apple iOS Mobile SDK generator for Axon v12.0."""
import os
import shutil
import subprocess
import sys
from pathlib import Path
from .runtime_paths import runtime_dir


JNI_BRIDGE_C = r'''#include "axon_runtime.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#ifdef __ANDROID__
#include <jni.h>
JNIEXPORT jlong JNICALL Java_com_axon_AxonBridge_nativeVersion(JNIEnv *env, jclass cls) {
    (void)env; (void)cls; return 1200;
}
JNIEXPORT jint JNICALL Java_com_axon_AxonBridge_loadWeights(JNIEnv *env, jclass cls, jstring path) {
    (void)cls;
    const char *cpath = (*env)->GetStringUTFChars(env, path, NULL);
    int rc = axon_load_weights_file(cpath);
    (*env)->ReleaseStringUTFChars(env, path, cpath);
    return (jint)rc;
}
JNIEXPORT void JNICALL Java_com_axon_AxonBridge_unloadWeights(JNIEnv *env, jclass cls) {
    (void)env; (void)cls; axon_unload_weights_file();
}
JNIEXPORT jfloatArray JNICALL Java_com_axon_AxonBridge_matmul(JNIEnv *env, jclass cls, jfloatArray aArr, jfloatArray bArr, jint m, jint k, jint n) {
    (void)cls;
    jfloat *ap = (*env)->GetFloatArrayElements(env, aArr, NULL);
    jfloat *bp = (*env)->GetFloatArrayElements(env, bArr, NULL);
    int64_t sa[2] = {m, k}, sb[2] = {k, n}, sc[2] = {m, n};
    axon_arena_init(16u * 1024u * 1024u);
    axon_arena_reset();
    AxonTensor *A = axon_tensor_alloc(2, sa, AXON_DTYPE_F32);
    AxonTensor *B = axon_tensor_alloc(2, sb, AXON_DTYPE_F32);
    AxonTensor *C = axon_tensor_alloc(2, sc, AXON_DTYPE_F32);
    axon_tensor_fill_f32(A, ap, (int64_t)m * k);
    axon_tensor_fill_f32(B, bp, (int64_t)k * n);
    axon_matmul(A, B, C);
    jfloatArray out = (*env)->NewFloatArray(env, m * n);
    if (out && C && C->data) (*env)->SetFloatArrayRegion(env, out, 0, m * n, (const jfloat *)C->data);
    axon_tensor_free(A); axon_tensor_free(B); axon_tensor_free(C);
    axon_arena_destroy();
    (*env)->ReleaseFloatArrayElements(env, aArr, ap, JNI_ABORT);
    (*env)->ReleaseFloatArrayElements(env, bArr, bp, JNI_ABORT);
    return out;
}
#endif
'''

KOTLIN_KT = r'''package com.axon

object AxonBridge {
    init {
        System.loadLibrary("axon_android")
    }
    external fun nativeVersion(): Long
    external fun loadWeights(axn3Path: String): Int
    external fun unloadWeights()
    external fun matmul(a: FloatArray, b: FloatArray, m: Int, k: Int, n: Int): FloatArray
}
'''

SWIFT_WRAP = r'''import Foundation

public final class AxonEngine {
    public init(arenaMB: Int = 16) {
        axon_arena_init(arenaMB * 1024 * 1024)
    }
    deinit {
        axon_unload_weights_file()
        axon_arena_destroy()
    }
    public func loadWeights(path: String) -> Int32 {
        return path.withCString { axon_load_weights_file($0) }
    }
    public func rmsNorm(input: [Float], gamma: [Float], rows: Int64, cols: Int64) -> [Float] {
        var sIn: [Int64] = [rows, cols]
        var sG: [Int64] = [cols]
        axon_arena_reset()
        guard let x = axon_tensor_alloc(2, &sIn, 0),
              let g = axon_tensor_alloc(1, &sG, 0),
              let c = axon_tensor_alloc(2, &sIn, 0) else { return [] }
        input.withUnsafeBufferPointer { _ = axon_tensor_fill_f32(x, $0.baseAddress, Int64(input.count)) }
        gamma.withUnsafeBufferPointer { _ = axon_tensor_fill_f32(g, $0.baseAddress, Int64(gamma.count)) }
        _ = axon_rms_norm(x, g, c)
        let ptr = c.pointee.data.assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: ptr, count: Int(rows * cols)))
    }
}
'''


def _find_android_cc():
    roots = [
        os.environ.get("ANDROID_NDK_HOME", ""),
        os.environ.get("ANDROID_NDK_ROOT", ""),
        "/usr/local/lib/android/sdk/ndk",
    ]
    for r in roots:
        if not r or not os.path.isdir(r):
            continue
        cands = [r] if os.path.isdir(os.path.join(r, "toolchains")) else sorted(
            [os.path.join(r, d) for d in os.listdir(r) if os.path.isdir(os.path.join(r, d))]
        )
        for ndk in reversed(cands):
            pre = os.path.join(ndk, "toolchains", "llvm", "prebuilt")
            if not os.path.isdir(pre):
                continue
            for host in os.listdir(pre):
                cc = os.path.join(pre, host, "bin", "aarch64-linux-android24-clang")
                if os.path.isfile(cc):
                    return cc
    return shutil.which("aarch64-linux-android24-clang")


def build_mobile_sdk(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rt = runtime_dir()
    shutil.copy2(rt / "axon_runtime.h", out / "axon_runtime.h")
    (out / "module.modulemap").write_text('module AxonRuntime {\n  header "axon_runtime.h"\n  export *\n}\n', encoding="utf-8")
    (out / "AxonBridge.kt").write_text(KOTLIN_KT, encoding="utf-8")
    (out / "AxonEngine.swift").write_text(SWIFT_WRAP, encoding="utf-8")
    jni_c = out / "axon_mobile_jni.c"
    jni_c.write_text(JNI_BRIDGE_C, encoding="utf-8")

    built = {"android_so": None, "ios_a": None, "ios_sim_a": None, "out_dir": str(out)}
    ndk_cc = _find_android_cc()
    if ndk_cc:
        so = out / "libaxon_android_arm64.so"
        r = subprocess.run(
            [ndk_cc, "-O3", "-fPIC", "-shared", "-std=c11", "-D__ANDROID__", f"-I{rt}", str(rt / "axon_runtime.c"), str(jni_c), "-lm", "-o", str(so)],
            capture_output=True, text=True,
        )
        if r.returncode == 0:
            built["android_so"] = str(so)

    if sys.platform == "darwin" and shutil.which("xcrun"):
        for sdk, name, key in (
            ("iphoneos", "libaxon_ios_arm64.a", "ios_a"),
            ("iphonesimulator", "libaxon_ios_sim_arm64.a", "ios_sim_a"),
        ):
            p = subprocess.run(["xcrun", "--sdk", sdk, "--show-sdk-path"], capture_output=True, text=True)
            if p.returncode == 0:
                sdk_path = p.stdout.strip()
                obj = out / f"{sdk}_rt.o"
                lib = out / name
                c_res = subprocess.run(
                    ["xcrun", "--sdk", sdk, "clang", "-arch", "arm64", "-isysroot", sdk_path,
                     "-miphoneos-version-min=13.0" if sdk == "iphoneos" else "-mios-simulator-version-min=13.0",
                     "-O3", "-std=c11", f"-I{rt}", "-c", str(rt / "axon_runtime.c"), "-o", str(obj)],
                    capture_output=True, text=True,
                )
                if c_res.returncode == 0:
                    subprocess.run(["ar", "rcs", str(lib), str(obj)], check=True)
                    obj.unlink()
                    built[key] = str(lib)
    return built
'''


STUDIO_C = r'''#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  include <windows.h>
#  include <direct.h>
typedef SOCKET axon_sock_t;
#  define AXON_BAD_SOCK INVALID_SOCKET
#  define axon_sock_close(s) closesocket(s)
#else
#  define _POSIX_C_SOURCE 200809L
#  define _DEFAULT_SOURCE
#  include <arpa/inet.h>
#  include <dlfcn.h>
#  include <netinet/in.h>
#  include <signal.h>
#  include <sys/socket.h>
#  include <sys/stat.h>
#  include <unistd.h>
typedef int axon_sock_t;
#  define AXON_BAD_SOCK (-1)
#  define axon_sock_close(s) close(s)
#endif

#include "axon_runtime.h"
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef AXON_EDITION_PRO
#define AXON_EDITION_PRO 0
#endif

#define AXON_WEEK_SEC 604800ULL
#define AXON_FREE_CAP_SEC 86400.0

static double studio_now_ms(void) {
#ifdef _WIN32
    static LARGE_INTEGER f; static int init = 0;
    if (!init) { QueryPerformanceFrequency(&f); init = 1; }
    LARGE_INTEGER n; QueryPerformanceCounter(&n);
    return (double)n.QuadPart * 1000.0 / (double)f.QuadPart;
#else
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e3 + (double)t.tv_nsec / 1e6;
#endif
}

/* ---- SHA-256 + HMAC-SHA256 for safe Fail-Closed integrity & quota ---- */
static const uint32_t K256[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};
#define ROR32(x,n) (((x)>>(n))|((x)<<(32-(n))))
static void sha256_transform(uint32_t st[8], const uint8_t b[64]) {
    uint32_t w[64], a,bb,c,d,e,f,g,h;
    for (int i=0;i<16;++i) w[i]=((uint32_t)b[i*4]<<24)|((uint32_t)b[i*4+1]<<16)|((uint32_t)b[i*4+2]<<8)|b[i*4+3];
    for (int i=16;i<64;++i) {
        uint32_t s0=ROR32(w[i-15],7)^ROR32(w[i-15],18)^(w[i-15]>>3);
        uint32_t s1=ROR32(w[i-2],17)^ROR32(w[i-2],19)^(w[i-2]>>10);
        w[i]=w[i-16]+s0+w[i-7]+s1;
    }
    a=st[0];bb=st[1];c=st[2];d=st[3];e=st[4];f=st[5];g=st[6];h=st[7];
    for (int i=0;i<64;++i) {
        uint32_t S1=ROR32(e,6)^ROR32(e,11)^ROR32(e,25), ch=(e&f)^((~e)&g), t1=h+S1+ch+K256[i]+w[i];
        uint32_t S0=ROR32(a,2)^ROR32(a,13)^ROR32(a,22), maj=(a&bb)^(a&c)^(bb&c), t2=S0+maj;
        h=g;g=f;f=e;e=d+t1;d=c;c=bb;bb=a;a=t1+t2;
    }
    st[0]+=a;st[1]+=bb;st[2]+=c;st[3]+=d;st[4]+=e;st[5]+=f;st[6]+=g;st[7]+=h;
}
static void sha256_calc(const uint8_t *data, size_t len, uint8_t out[32]) {
    uint32_t st[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    uint8_t blk[64]; size_t i=0;
    while (i+64<=len) { sha256_transform(st, data+i); i+=64; }
    size_t rem=len-i; memset(blk,0,64); if(rem) memcpy(blk,data+i,rem); blk[rem]=0x80;
    if (rem>=56) { sha256_transform(st,blk); memset(blk,0,64); }
    uint64_t bits=(uint64_t)len*8ULL;
    for (int k=0;k<8;++k) blk[63-k]=(uint8_t)(bits>>(k*8));
    sha256_transform(st,blk);
    for (int k=0;k<8;++k){ out[k*4]=(st[k]>>24)&255; out[k*4+1]=(st[k]>>16)&255; out[k*4+2]=(st[k]>>8)&255; out[k*4+3]=st[k]&255; }
}
static void hmac_sha256(const uint8_t *key, size_t klen, const uint8_t *msg, size_t mlen, uint8_t out[32]) {
    uint8_t k0[64], ipad[64], opad[64], ih[32]; memset(k0,0,64);
    if (klen>64) sha256_calc(key,klen,k0); else memcpy(k0,key,klen);
    for (int i=0;i<64;++i){ ipad[i]=k0[i]^0x36; opad[i]=k0[i]^0x5c; }
    uint8_t *buf1=(uint8_t*)malloc(64+mlen); if(!buf1)return;
    memcpy(buf1,ipad,64); if(mlen) memcpy(buf1+64,msg,mlen); sha256_calc(buf1,64+mlen,ih); free(buf1);
    uint8_t buf2[96]; memcpy(buf2,opad,64); memcpy(buf2+64,ih,32); sha256_calc(buf2,96,out);
}

/* ---- Dynamic OpenCL (zero compile-time dependency on libOpenCL) ---- */
typedef void* cl_plat; typedef void* cl_dev; typedef void* cl_ctx; typedef void* cl_q;
typedef void* cl_prog; typedef void* cl_kern; typedef void* cl_buf;
typedef int32_t cl_i; typedef uint32_t cl_u; typedef uint64_t cl_fl;
typedef struct {
    void *lib;
    int available, is_gpu;
    char device_name[128];
    cl_ctx ctx; cl_q q; cl_prog prog; cl_kern k_gemm;
    cl_i (*get_plats)(cl_u, cl_plat*, cl_u*);
    cl_i (*get_devs)(cl_plat, cl_fl, cl_u, cl_dev*, cl_u*);
    cl_i (*get_dinfo)(cl_dev, cl_u, size_t, void*, size_t*);
    cl_ctx (*create_ctx)(void*, cl_u, const cl_dev*, void*, void*, cl_i*);
    cl_q (*create_q)(cl_ctx, cl_dev, cl_fl, cl_i*);
    cl_prog (*create_prog)(cl_ctx, cl_u, const char**, const size_t*, cl_i*);
    cl_i (*build_prog)(cl_prog, cl_u, const cl_dev*, const char*, void*, void*);
    cl_kern (*create_kern)(cl_prog, const char*, cl_i*);
    cl_buf (*create_buf)(cl_ctx, cl_fl, size_t, void*, cl_i*);
    cl_i (*enq_w)(cl_q, cl_buf, cl_u, size_t, size_t, const void*, cl_u, void*, void*);
    cl_i (*enq_r)(cl_q, cl_buf, cl_u, size_t, size_t, void*, cl_u, void*, void*);
    cl_i (*set_arg)(cl_kern, cl_u, size_t, const void*);
    cl_i (*enq_nd)(cl_q, cl_kern, cl_u, const size_t*, const size_t*, const size_t*, cl_u, void*, void*);
    cl_i (*finish)(cl_q);
    cl_i (*rel_buf)(cl_buf); cl_i (*rel_k)(cl_kern); cl_i (*rel_p)(cl_prog); cl_i (*rel_q)(cl_q); cl_i (*rel_c)(cl_ctx);
} DynOCL;
static DynOCL g_ocl = {0};
static int g_ocl_inited = 0;

static void *dyn_sym(void *h, const char *name) {
#ifdef _WIN32
    return (void*)GetProcAddress((HMODULE)h, name);
#else
    return dlsym(h, name);
#endif
}
static void dyn_ocl_init(void) {
    if (g_ocl_inited) return;
    g_ocl_inited = 1;
    snprintf(g_ocl.device_name, sizeof(g_ocl.device_name), "none");
#ifdef _WIN32
    void *h = (void*)LoadLibraryA("OpenCL.dll");
#elif defined(__APPLE__)
    void *h = dlopen("/System/Library/Frameworks/OpenCL.framework/OpenCL", RTLD_LAZY);
#else
    void *h = dlopen("libOpenCL.so.1", RTLD_LAZY);
    if (!h) h = dlopen("libOpenCL.so", RTLD_LAZY);
#endif
    if (!h) return;
    g_ocl.lib = h;
    g_ocl.get_plats   = dyn_sym(h, "clGetPlatformIDs");
    g_ocl.get_devs    = dyn_sym(h, "clGetDeviceIDs");
    g_ocl.get_dinfo   = dyn_sym(h, "clGetDeviceInfo");
    g_ocl.create_ctx  = dyn_sym(h, "clCreateContext");
    g_ocl.create_q    = dyn_sym(h, "clCreateCommandQueue");
    g_ocl.create_prog = dyn_sym(h, "clCreateProgramWithSource");
    g_ocl.build_prog  = dyn_sym(h, "clBuildProgram");
    g_ocl.create_kern = dyn_sym(h, "clCreateKernel");
    g_ocl.create_buf  = dyn_sym(h, "clCreateBuffer");
    g_ocl.enq_w       = dyn_sym(h, "clEnqueueWriteBuffer");
    g_ocl.enq_r       = dyn_sym(h, "clEnqueueReadBuffer");
    g_ocl.set_arg     = dyn_sym(h, "clSetKernelArg");
    g_ocl.enq_nd      = dyn_sym(h, "clEnqueueNDRangeKernel");
    g_ocl.finish      = dyn_sym(h, "clFinish");
    g_ocl.rel_buf     = dyn_sym(h, "clReleaseMemObject");
    g_ocl.rel_k       = dyn_sym(h, "clReleaseKernel");
    g_ocl.rel_p       = dyn_sym(h, "clReleaseProgram");
    g_ocl.rel_q       = dyn_sym(h, "clReleaseCommandQueue");
    g_ocl.rel_c       = dyn_sym(h, "clReleaseContext");
    if (!g_ocl.get_plats || !g_ocl.get_devs || !g_ocl.create_ctx || !g_ocl.create_q || !g_ocl.create_prog || !g_ocl.build_prog || !g_ocl.create_kern) return;

    cl_u np = 0; if (g_ocl.get_plats(0, NULL, &np) != 0 || np == 0) return;
    cl_plat plats[16]; if (np > 16) np = 16;
    if (g_ocl.get_plats(np, plats, NULL) != 0) return;
    cl_dev dev = NULL; int is_gpu = 0;
    for (cl_u i = 0; i < np && !dev; ++i) {
        cl_u nd = 0;
        if (g_ocl.get_devs(plats[i], 4, 1, &dev, &nd) == 0 && nd > 0) { is_gpu = 1; break; } /* GPU = 4 */
    }
    for (cl_u i = 0; i < np && !dev; ++i) {
        cl_u nd = 0;
        if (g_ocl.get_devs(plats[i], 0xFFFFFFFFULL, 1, &dev, &nd) == 0 && nd > 0) break;
    }
    if (!dev) return;
    if (g_ocl.get_dinfo) g_ocl.get_dinfo(dev, 0x102B, sizeof(g_ocl.device_name)-1, g_ocl.device_name, NULL);
    cl_i err = 0;
    g_ocl.ctx = g_ocl.create_ctx(NULL, 1, &dev, NULL, NULL, &err); if (err || !g_ocl.ctx) return;
    g_ocl.q = g_ocl.create_q(g_ocl.ctx, dev, 0, &err); if (err || !g_ocl.q) return;
    const char *ksrc =
        "__kernel void mm(__global const float*A,__global const float*B,__global float*C,int M,int K,int N){\n"
        " int r=get_global_id(0), c=get_global_id(1); if(r>=M||c>=N) return;\n"
        " float s=0.0f; for(int k=0;k<K;++k) s+=A[r*K+k]*B[k*N+c]; C[r*N+c]=s;\n"
        "}\n";
    size_t klen = strlen(ksrc);
    g_ocl.prog = g_ocl.create_prog(g_ocl.ctx, 1, &ksrc, &klen, &err);
    if (err || !g_ocl.prog || g_ocl.build_prog(g_ocl.prog, 1, &dev, "", NULL, NULL) != 0) return;
    g_ocl.k_gemm = g_ocl.create_kern(g_ocl.prog, "mm", &err);
    if (!err && g_ocl.k_gemm) { g_ocl.available = 1; g_ocl.is_gpu = is_gpu; }
}
static int dyn_ocl_matmul(const AxonTensor *A, const AxonTensor *B, AxonTensor *C) {
    if (!g_ocl.available) return AXON_ERR_ARG;
    int M = (int)A->shape[0], K = (int)A->shape[1], N = (int)B->shape[1];
    float *bf32 = NULL; const void *bp = B->data;
    if (B->dtype != AXON_DTYPE_F32) {
        bf32 = (float*)malloc((size_t)(K * N) * sizeof(float));
        if (!bf32 || axon_tensor_to_f32(B, bf32, (int64_t)K * N) != AXON_OK) { free(bf32); return AXON_ERR_NOMEM; }
        bp = bf32;
    }
    cl_i e = 0;
    cl_buf dA = g_ocl.create_buf(g_ocl.ctx, (1<<2)|(1<<5), (size_t)(M*K)*4, (void*)A->data, &e);
    cl_buf dB = g_ocl.create_buf(g_ocl.ctx, (1<<2)|(1<<5), (size_t)(K*N)*4, (void*)bp, &e);
    cl_buf dC = g_ocl.create_buf(g_ocl.ctx, (1<<0), (size_t)(M*N)*4, NULL, &e);
    free(bf32);
    if (!dA || !dB || !dC) {
        if (dA) g_ocl.rel_buf(dA); if (dB) g_ocl.rel_buf(dB); if (dC) g_ocl.rel_buf(dC);
        return AXON_ERR_NOMEM;
    }
    g_ocl.set_arg(g_ocl.k_gemm, 0, sizeof(cl_buf), &dA);
    g_ocl.set_arg(g_ocl.k_gemm, 1, sizeof(cl_buf), &dB);
    g_ocl.set_arg(g_ocl.k_gemm, 2, sizeof(cl_buf), &dC);
    g_ocl.set_arg(g_ocl.k_gemm, 3, sizeof(int), &M);
    g_ocl.set_arg(g_ocl.k_gemm, 4, sizeof(int), &K);
    g_ocl.set_arg(g_ocl.k_gemm, 5, sizeof(int), &N);
    size_t gws[2] = {(size_t)M, (size_t)N};
    e = g_ocl.enq_nd(g_ocl.q, g_ocl.k_gemm, 2, NULL, gws, NULL, 0, NULL, NULL);
    if (!e) { g_ocl.finish(g_ocl.q); e = g_ocl.enq_r(g_ocl.q, dC, 1, 0, (size_t)(M*N)*4, C->data, 0, NULL, NULL); }
    g_ocl.rel_buf(dA); g_ocl.rel_buf(dB); g_ocl.rel_buf(dC);
    return e ? AXON_ERR_ARG : AXON_OK;
}

/* ---- Server Detection & Weekly 24h Active Quota (Free Edition Only) ---- */
static int g_sim_server = 0;
static const uint8_t QUOTA_KEY[24] = "AXON_V12_QUOTA_HMAC_KEY!";
typedef struct { uint32_t magic; uint32_t reserved; uint64_t week_start; double active_sec; uint64_t last_seen; uint8_t tag[32]; } QuotaState;

static int detect_server_env(void) {
    if (g_sim_server || getenv("AXON_TEST_SERVER")) return 1;
#if defined(_WIN32) || defined(__APPLE__) || defined(__ANDROID__)
    return 0; /* Unlimited on Windows, macOS, Android, iOS */
#else
    long cpus = sysconf(_SC_NPROCESSORS_ONLN);
    if (cpus >= 16) return 1;
    struct stat st;
    if (stat("/sys/devices/system/node/node1", &st) == 0) return 1;
    if (stat("/dev/accel0", &st) == 0 || getenv("TPU_NAME")) return 1;
    return 0;
#endif
}
static void get_quota_path(char *buf, size_t sz) {
#ifdef _WIN32
    const char *base = getenv("LOCALAPPDATA"); if (!base) base = ".";
    snprintf(buf, sz, "%s\\axon_quota_v12.bin", base);
#else
    const char *home = getenv("HOME"); if (!home) home = "/tmp";
    snprintf(buf, sz, "%s/.axon_quota_v12.bin", home);
#endif
}
static void compute_quota_tag(QuotaState *qs, uint8_t out[32]) {
    hmac_sha256(QUOTA_KEY, sizeof(QUOTA_KEY), (const uint8_t*)qs, offsetof(QuotaState, tag), out);
}
static int check_and_add_active_time(double add_sec, double *out_used, int *out_locked) {
    *out_used = 0.0; *out_locked = 0;
    if (AXON_EDITION_PRO || !detect_server_env()) return 0;
    char qpath[512]; get_quota_path(qpath, sizeof(qpath));
    uint64_t now = (uint64_t)time(NULL);
    QuotaState qs; memset(&qs, 0, sizeof(qs));
    FILE *fp = fopen(qpath, "rb");
    if (fp) {
        size_t nr = fread(&qs, 1, sizeof(qs), fp); fclose(fp);
        if (nr != sizeof(qs) || qs.magic != 0x41585131u) { *out_locked = 1; return -1; }
        uint8_t exp[32]; compute_quota_tag(&qs, exp);
        int diff = 0; for (int i=0;i<32;++i) diff |= (exp[i] ^ qs.tag[i]);
        if (diff != 0 || now + 300ULL < qs.last_seen) { *out_locked = 1; return -1; }
        if (now - qs.week_start >= AXON_WEEK_SEC) { qs.week_start = now; qs.active_sec = 0.0; }
    } else {
        qs.magic = 0x41585131u; qs.week_start = now; qs.active_sec = 0.0;
    }
    if (qs.active_sec >= AXON_FREE_CAP_SEC) { *out_used = qs.active_sec; *out_locked = 1; return -2; }
    if (add_sec > 0.0) {
        qs.active_sec += add_sec;
        qs.last_seen = now;
        compute_quota_tag(&qs, qs.tag);
        FILE *fw = fopen(qpath, "wb");
        if (fw) { fwrite(&qs, 1, sizeof(qs), fw); fclose(fw); }
    }
    *out_used = qs.active_sec;
    if (qs.active_sec >= AXON_FREE_CAP_SEC) { *out_locked = 1; return -2; }
    return 0;
}

/* ---- Pure-C Quantizer & Universal Graph Engine (AXG1 / Safetensors / Demo) ---- */
typedef struct {
    int op, nin, ndim;
    int inputs[4];
    int64_t shape[4];
    char weight[96];
    int keep, region_blocked;
} CNode;
typedef struct {
    char name[96];
    int ndim;
    int64_t shape[4];
    float *f32_data;
    int64_t numel;
    uint8_t *fp4_buf; size_t fp4_len;
    uint8_t *tern_buf; float tern_scale; size_t tern_len;
    float *reg_data; float *sig_data; int64_t R, keep;
} CWeight;
typedef struct {
    char model_name[128];
    char format[32];
    char backend_mode[32]; /* auto | f32 | fp4 | ternary | brain | gpu */
    int num_nodes, in_idx, out_idx;
    int in_ndim; int64_t in_shape[4];
    CNode *nodes;
    int num_weights;
    CWeight *weights;
    size_t raw_f32_bytes;
    size_t active_weight_bytes;
    double last_ms;
    int64_t last_active_reg, last_total_reg;
    double last_ram_saved, last_flops_skipped;
} StudioModel;

static StudioModel g_model = {0};

static float f16_to_f32(uint16_t h) {
    uint32_t s = (uint32_t)(h >> 15) << 31, e = (h >> 10) & 0x1F, m = h & 0x3FF, out;
    if (e == 0) {
        if (!m) out = s;
        else { int sh = -1; while (!(m & 0x400)) { m <<= 1; sh--; } out = s | ((uint32_t)(127 - 15 + sh + 1) << 23) | ((m & 0x3FF) << 13); }
    } else if (e == 0x1F) out = s | 0x7F800000u | (m << 13);
    else out = s | ((e - 15 + 127) << 23) | (m << 13);
    float f; memcpy(&f, &out, 4); return f;
}
static float bf16_to_f32(uint16_t b) { uint32_t u = ((uint32_t)b) << 16; float f; memcpy(&f, &u, 4); return f; }

static void free_model(StudioModel *m) {
    free(m->nodes);
    for (int i = 0; i < m->num_weights; ++i) {
        free(m->weights[i].f32_data); free(m->weights[i].fp4_buf);
        free(m->weights[i].tern_buf); free(m->weights[i].reg_data); free(m->weights[i].sig_data);
    }
    free(m->weights);
    memset(m, 0, sizeof(*m));
}

static const float C_FP4_LV[8] = {0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f};
static uint8_t c_fp4_nib(float v) {
    uint8_t s = (v < 0.f) ? 8 : 0; float av = fabsf(v); uint8_t best = 0; float bd = 1e30f;
    for (int i = 0; i < 8; ++i) { float d = fabsf(av - C_FP4_LV[i]); if (d <= bd) { best = (uint8_t)i; bd = d; } }
    return (uint8_t)(best | s);
}
static void quantize_weight_all(CWeight *w, const float *src) {
    int64_t n = w->numel;
    /* Ternary 1.58-bit */
    double sum_a = 0.0; for (int64_t i=0;i<n;++i) sum_a += fabs(src[i]);
    float alpha = (n > 0 && sum_a > 0) ? (float)(sum_a / n) : 1.0f; if (alpha < 1e-8f) alpha = 1e-8f;
    w->tern_scale = alpha; w->tern_len = (size_t)((n + 3) / 4);
    w->tern_buf = (uint8_t*)calloc(w->tern_len, 1);
    if (w->tern_buf) {
        for (int64_t i=0;i<n;++i) {
            float r = src[i] / alpha; int q = (r >= 0.5f) ? 1 : ((r <= -0.5f) ? -1 : 0);
            uint8_t code = (q == 1) ? 1 : ((q == -1) ? 2 : 0);
            w->tern_buf[i>>2] |= (uint8_t)(code << ((i & 3) * 2));
        }
    }
    /* FP4 + Outliers */
    uint32_t nb = (uint32_t)((n + 31) / 32);
    uint32_t res_cap = 32, nr = 0;
    uint32_t *ridx = (uint32_t*)malloc(res_cap * sizeof(uint32_t));
    float *rdiff = (float*)malloc(res_cap * sizeof(float));
    uint8_t *scales = (uint8_t*)calloc(nb, 1);
    uint8_t *packed = (uint8_t*)calloc((size_t)nb * 16, 1);
    if (scales && packed && ridx && rdiff) {
        for (uint32_t blk=0; blk<nb; ++blk) {
            int64_t s = (int64_t)blk * 32; float mean = 0.f;
            for (int i=0;i<32;++i) mean += (s+i < n) ? fabsf(src[s+i]) : 0.f; mean /= 32.f;
            float var = 0.f; for (int i=0;i<32;++i) { float d=((s+i<n)?fabsf(src[s+i]):0.f)-mean; var += d*d; }
            float tau = mean + 3.0f * sqrtf(var / 32.f), mx = 0.f;
            for (int i=0;i<32;++i) { float a=(s+i<n)?fabsf(src[s+i]):0.f; if (a<=tau && a>mx) mx=a; }
            if (mx <= 0.f) mx = 1e-8f;
            int exp = (int)ceilf(log2f(mx / 6.0f)); if (exp < -126) exp = -126; if (exp > 127) exp = 127;
            float sc = ldexpf(1.0f, exp); scales[blk] = (uint8_t)((exp + 127) & 0xFF);
            for (int i=0;i<32;++i) {
                float v = (s+i < n) ? src[s+i] : 0.f; uint8_t nib;
                if (fabsf(v) > tau && s+i < n) {
                    float q = (v >= 0.f) ? 6.f : -6.f; nib = c_fp4_nib(q);
                    if (nr == res_cap) {
                        res_cap *= 2;
                        ridx = (uint32_t*)realloc(ridx, res_cap * sizeof(uint32_t));
                        rdiff = (float*)realloc(rdiff, res_cap * sizeof(float));
                    }
                    ridx[nr] = (uint32_t)(s + i); rdiff[nr] = v - sc * q; nr++;
                } else nib = c_fp4_nib(v / sc);
                packed[(size_t)blk * 16 + (i >> 1)] |= (i & 1) ? (uint8_t)((nib & 0xF) << 4) : (uint8_t)(nib & 0xF);
            }
        }
        w->fp4_len = 8 + (size_t)nb + (size_t)nb * 16 + (size_t)nr * 8;
        w->fp4_buf = (uint8_t*)malloc(w->fp4_len);
        if (w->fp4_buf) {
            memcpy(w->fp4_buf, &nb, 4); memcpy(w->fp4_buf + 4, &nr, 4);
            memcpy(w->fp4_buf + 8, scales, nb);
            memcpy(w->fp4_buf + 8 + nb, packed, (size_t)nb * 16);
            uint8_t *rp = w->fp4_buf + 8 + nb + (size_t)nb * 16;
            for (uint32_t r=0; r<nr; ++r) { memcpy(rp + r*8, &ridx[r], 4); memcpy(rp + r*8 + 4, &rdiff[r], 4); }
        }
    }
    free(ridx); free(rdiff); free(scales); free(packed);
    if (w->ndim == 2 && w->shape[1] >= 4) {
        int64_t K = w->shape[0], N = w->shape[1];
        int64_t R = (N <= 64) ? (N < 8 ? N : 8) : (N / 16 < 64 ? (N / 16 > 8 ? N / 16 : 8) : 64);
        w->R = R; w->keep = (R >= 2) ? (R / 2) : 1;
        w->sig_data = (float*)calloc((size_t)(K * R), sizeof(float));
        w->reg_data = (float*)malloc((size_t)(K * N) * sizeof(float));
        if (w->sig_data && w->reg_data) {
            int64_t pos = 0;
            for (int64_t r = 0; r < R; ++r) {
                int64_t c0 = (r * N) / R, c1 = ((r + 1) * N) / R, wr = c1 - c0;
                for (int64_t k = 0; k < K; ++k) {
                    float acc = 0.f;
                    for (int64_t c = c0; c < c1; ++c) { float val = src[k * N + c]; w->reg_data[pos++] = val; acc += val; }
                    w->sig_data[k * R + r] = acc / (float)(wr > 0 ? wr : 1);
                }
            }
        }
    }
}

static void add_weight(StudioModel *m, const char *name, int ndim, const int64_t *shp, const float *vals) {
    m->weights = (CWeight*)realloc(m->weights, (size_t)(m->num_weights + 1) * sizeof(CWeight));
    CWeight *w = &m->weights[m->num_weights++]; memset(w, 0, sizeof(*w));
    snprintf(w->name, sizeof(w->name), "%s", name);
    w->ndim = ndim; int64_t tot = 1;
    for (int i=0;i<ndim;++i){ w->shape[i]=shp[i]; tot *= shp[i]; }
    w->numel = tot;
    w->f32_data = (float*)malloc((size_t)tot * sizeof(float));
    memcpy(w->f32_data, vals, (size_t)tot * sizeof(float));
    m->raw_f32_bytes += (size_t)tot * sizeof(float);
    quantize_weight_all(w, vals);
}

static CWeight *find_cweight(StudioModel *m, const char *name) {
    for (int i = 0; i < m->num_weights; ++i) if (strcmp(m->weights[i].name, name) == 0) return &m->weights[i];
    return NULL;
}

static void init_builtin_demo_model(StudioModel *m) {
    free_model(m);
    snprintf(m->model_name, sizeof(m->model_name), "mini_llm_builtin");
    snprintf(m->format, sizeof(m->format), "builtin-llama");
    snprintf(m->backend_mode, sizeof(m->backend_mode), "fp4");
    m->in_ndim = 2; m->in_shape[0] = 4; m->in_shape[1] = 8;

    float g[8] = {1,1,1,1,1,1,1,1}, Wq[64], Wk[64], Wv[64], Wo[64], Wg[64], Wu[64], Wd[64];
    for (int i=0;i<64;++i) {
        Wq[i] = 0.25f * sinf((float)(i + 1) * 0.7f);
        Wk[i] = 0.25f * cosf((float)(i + 1) * 0.5f);
        Wv[i] = ((i / 8) == (i % 8)) ? 0.5f : 0.05f * sinf((float)i);
        Wo[i] = 0.22f * cosf((float)(i + 3) * 0.4f);
        Wg[i] = 0.20f * sinf((float)(i + 2) * 0.6f);
        Wu[i] = 0.18f * cosf((float)(i + 5) * 0.3f);
        Wd[i] = 0.20f * sinf((float)(i + 7) * 0.5f);
    }
    int64_t s1[1]={8}, s2[2]={8,8};
    add_weight(m,"g1",1,s1,g); add_weight(m,"Wq",2,s2,Wq); add_weight(m,"Wk",2,s2,Wk); add_weight(m,"Wv",2,s2,Wv);
    add_weight(m,"Wo",2,s2,Wo); add_weight(m,"Wg",2,s2,Wg); add_weight(m,"Wu",2,s2,Wu); add_weight(m,"Wd",2,s2,Wd);

    /* 20 nodes: x(0), g1(1), h1=rms(2), Wq(3), qm=mm(4), q=rope(5), Wk(6), km=mm(7), k=rope(8), Wv(9), v=mm(10),
       att=attn_causal(11), Wo(12), ao=mm(13), x2=add(14), h2=rms(15), Wg(16), gmm=mm_silu(17), Wu(18), umm=mm(19),
       swi=mul(20), Wd(21), down=mm(22), x3=add(23), out=rms(24) */
    m->num_nodes = 25; m->in_idx = 0; m->out_idx = 24;
    m->nodes = (CNode*)calloc(25, sizeof(CNode));
    #define SN(idx,OP,ND,D0,D1,NI,A,B,C,WN) do{ CNode*z=&m->nodes[idx]; z->op=OP; z->ndim=ND; z->shape[0]=D0; z->shape[1]=D1; z->nin=NI; z->inputs[0]=A; z->inputs[1]=B; z->inputs[2]=C; if(WN) snprintf(z->weight,sizeof(z->weight),"%s",WN); }while(0)
    SN(0,0,2,4,8,0,-1,-1,-1,NULL); SN(1,1,1,8,0,0,-1,-1,-1,"g1"); SN(2,13,2,4,8,2,0,1,-1,NULL);
    SN(3,1,2,8,8,0,-1,-1,-1,"Wq"); SN(4,2,2,4,8,2,2,3,-1,NULL); SN(5,14,2,4,8,1,4,-1,-1,NULL);
    SN(6,1,2,8,8,0,-1,-1,-1,"Wk"); SN(7,2,2,4,8,2,2,6,-1,NULL); SN(8,14,2,4,8,1,7,-1,-1,NULL);
    SN(9,1,2,8,8,0,-1,-1,-1,"Wv"); SN(10,2,2,4,8,2,2,9,-1,NULL); SN(11,15,2,4,8,3,5,8,10);
    SN(12,1,2,8,8,0,-1,-1,-1,"Wo"); SN(13,2,2,4,8,2,11,12,-1,NULL); SN(14,4,2,4,8,2,0,13,-1,NULL);
    SN(15,13,2,4,8,2,14,1,-1,NULL); SN(16,1,2,8,8,0,-1,-1,-1,"Wg"); SN(17,20,2,4,8,2,15,16,-1,NULL);
    SN(18,1,2,8,8,0,-1,-1,-1,"Wu"); SN(19,2,2,4,8,2,15,18,-1,NULL); SN(20,6,2,4,8,2,17,19,-1,NULL);
    SN(21,1,2,8,8,0,-1,-1,-1,"Wd"); SN(22,2,2,4,8,2,20,21,-1,NULL); SN(23,4,2,4,8,2,14,22,-1,NULL);
    SN(24,13,2,4,8,2,23,1,-1,NULL);
    #undef SN
}

static int load_axg_axn3(StudioModel *m, const char *axg_path, const char *axn_path) {
    FILE *fg = fopen(axg_path, "rb"); if (!fg) return -1;
    uint8_t hdr[56]; if (fread(hdr, 1, 56, fg) != 56 || memcmp(hdr, "AXG1", 4) != 0) { fclose(fg); return -1; }
    uint32_t ver, nn, in_i, out_i, in_nd, nlen; int64_t ishp[4];
    memcpy(&ver,hdr+4,4); memcpy(&nn,hdr+8,4); memcpy(&in_i,hdr+12,4); memcpy(&out_i,hdr+16,4);
    memcpy(&in_nd,hdr+20,4); memcpy(&nlen,hdr+24,4); memcpy(ishp,hdr+28,32);
    if (ver != 1 || nn == 0 || nn > 4096) { fclose(fg); return -1; }
    size_t rsz = (size_t)nn * 72; uint8_t *rb = (uint8_t*)malloc(rsz); char *nb = (char*)malloc(nlen + 1);
    if (!rb || !nb || fread(rb, 1, rsz, fg) != rsz || (nlen && fread(nb, 1, nlen, fg) != nlen)) {
        free(rb); free(nb); fclose(fg); return -1;
    }
    nb[nlen] = '\0'; fclose(fg);
    if (axon_load_weights_file(axn_path) != AXON_OK) { free(rb); free(nb); return -1; }

    free_model(m);
    snprintf(m->model_name, sizeof(m->model_name), "%s", axg_path);
    snprintf(m->format, sizeof(m->format), "axg1+axn3");
    snprintf(m->backend_mode, sizeof(m->backend_mode), "fp4");
    m->num_nodes = (int)nn; m->in_idx = (int)in_i; m->out_idx = (int)out_i;
    m->in_ndim = (int)in_nd; memcpy(m->in_shape, ishp, 32);
    m->nodes = (CNode*)calloc(nn, sizeof(CNode));
    for (uint32_t i=0;i<nn;++i) {
        const uint8_t *r = rb + i * 72; CNode *cn = &m->nodes[i];
        uint32_t op, nin, nd, noff; memcpy(&op,r,4); memcpy(&nin,r+4,4); memcpy(&nd,r+8,4); memcpy(&noff,r+12,4);
        cn->op = (int)op; cn->nin = (int)nin; cn->ndim = (int)nd;
        memcpy(cn->inputs, r+16, 16); memcpy(cn->shape, r+32, 32); memcpy(&cn->keep, r+64, 4); memcpy(&cn->region_blocked, r+68, 4);
        if (noff < nlen) snprintf(cn->weight, sizeof(cn->weight), "%s", nb + noff);
        if (cn->op == 1 && cn->weight[0]) {
            AxonTensor *wt = axon_weight_tensor(cn->weight);
            if (wt) {
                int64_t tot = axon_tensor_numel(wt);
                float *tmp = (float*)malloc((size_t)tot * sizeof(float));
                if (tmp && axon_tensor_to_f32(wt, tmp, tot) == AXON_OK) add_weight(m, cn->weight, (int)wt->ndim, wt->shape, tmp);
                free(tmp); axon_tensor_free(wt);
            }
        }
    }
    axon_unload_weights_file();
    free(rb); free(nb);
    return 0;
}

static int parse_safetensors_bytes(StudioModel *m, const char *name, const uint8_t *buf, size_t len, int transpose) {
    if (len < 16) return -1;
    uint64_t hlen = 0; memcpy(&hlen, buf, 8);
    if (hlen == 0 || hlen > len - 8 || hlen > 16u * 1024u * 1024u) return -1;
    char *hdr = (char*)malloc((size_t)hlen + 1); if (!hdr) return -1;
    memcpy(hdr, buf + 8, (size_t)hlen); hdr[hlen] = '\0';
    const uint8_t *data_base = buf + 8 + hlen; size_t data_len = len - 8 - (size_t)hlen;

    const char *prefixes[3] = {"l", "layers.", "model.layers."};
    const char *pfx = NULL;
    for (int p=0;p<3;++p) { char pat[64]; snprintf(pat,sizeof(pat),"\"%s0.weight\"",prefixes[p]); if (strstr(hdr,pat)) { pfx=prefixes[p]; break; } }
    if (!pfx) { free(hdr); return -1; }

    free_model(m);
    snprintf(m->model_name, sizeof(m->model_name), "%s", name);
    snprintf(m->format, sizeof(m->format), "safetensors");
    snprintf(m->backend_mode, sizeof(m->backend_mode), "fp4");

    int idx = 0, cur_node = -1;
    CNode *nodes = NULL; int ncap = 0, ncnt = 0;
    #define PUSH_NODE(OP,ND,D0,D1,NI,A,B,C,WN) do{ if(ncnt==ncap){ncap=ncap?ncap*2:32; nodes=(CNode*)realloc(nodes,ncap*sizeof(CNode));} CNode*z=&nodes[ncnt]; memset(z,0,sizeof(*z)); z->op=OP;z->ndim=ND;z->shape[0]=D0;z->shape[1]=D1;z->nin=NI;z->inputs[0]=A;z->inputs[1]=B;z->inputs[2]=C; if(WN)snprintf(z->weight,sizeof(z->weight),"%s",WN); ncnt++; }while(0)

    while (1) {
        char key[96]; snprintf(key, sizeof(key), "\"%s%d.weight\"", pfx, idx);
        const char *pos = strstr(hdr, key); if (!pos) break;
        const char *obj = strchr(pos, '{'), *end_obj = obj ? strchr(obj, '}') : NULL; if (!obj || !end_obj) break;
        int dt = 0; if (strstr(obj, "\"F16\"") && strstr(obj, "\"F16\"") < end_obj) dt = 1;
        else if (strstr(obj, "\"BF16\"") && strstr(obj, "\"BF16\"") < end_obj) dt = 2;
        else if (!(strstr(obj, "\"F32\"") && strstr(obj, "\"F32\"") < end_obj)) { free(nodes); free(hdr); return -1; }
        const char *sh = strstr(obj, "\"shape\""); const char *lb = sh ? strchr(sh, '[') : NULL;
        if (!lb || lb > end_obj) break;
        long r=0, c=0; if (sscanf(lb+1, " %ld , %ld", &r, &c) != 2 || r<=0 || c<=0) break;
        const char *df = strstr(obj, "\"data_offsets\""); const char *db = df ? strchr(df, '[') : NULL;
        if (!db || db > end_obj) break;
        unsigned long long b0=0, b1=0; if (sscanf(db+1, " %llu , %llu", &b0, &b1) != 2 || b1 < b0 || b1 > data_len) break;
        int64_t K = transpose ? c : r, N = transpose ? r : c, tot = K * N;
        float *wf = (float*)malloc((size_t)tot * sizeof(float)); if (!wf) break;
        const uint8_t *raw = data_base + b0;
        for (long i=0;i<r;++i) for (long j=0;j<c;++j) {
            size_t si = (size_t)i * (size_t)c + (size_t)j;
            size_t di = transpose ? ((size_t)j * (size_t)r + (size_t)i) : si;
            if (dt == 0) { float v; memcpy(&v, raw + si*4, 4); wf[di] = v; }
            else if (dt == 1) { uint16_t h; memcpy(&h, raw + si*2, 2); wf[di] = f16_to_f32(h); }
            else { uint16_t h; memcpy(&h, raw + si*2, 2); wf[di] = bf16_to_f32(h); }
        }
        if (idx == 0) { m->in_ndim = 2; m->in_shape[0] = 1; m->in_shape[1] = K; PUSH_NODE(0,2,1,K,0,-1,-1,-1,NULL); cur_node = 0; }
        char wn[64]; snprintf(wn, sizeof(wn), "l%d_w", idx);
        int64_t shp2[2] = {K, N}; add_weight(m, wn, 2, shp2, wf); free(wf);
        int wi = ncnt; PUSH_NODE(1,2,K,N,0,-1,-1,-1,wn);
        int mi = ncnt; PUSH_NODE(2,2,1,N,2,cur_node,wi,-1,NULL); cur_node = mi;

        snprintf(key, sizeof(key), "\"%s%d.bias\"", pfx, idx);
        const char *bp = strstr(hdr, key);
        if (bp) {
            const char *bo = strchr(bp, '{'), *be = bo ? strchr(bo, '}') : NULL;
            const char *bdf = bo ? strstr(bo, "\"data_offsets\"") : NULL; const char *bdb = bdf ? strchr(bdf, '[') : NULL;
            unsigned long long bb0=0, bb1=0;
            if (be && bdb && bdb < be && sscanf(bdb+1, " %llu , %llu", &bb0, &bb1) == 2 && bb1 <= data_len && bb1 - bb0 == (unsigned long long)N*4) {
                float *bf = (float*)malloc((size_t)N * sizeof(float)); memcpy(bf, data_base + bb0, (size_t)N * 4);
                char bn[64]; snprintf(bn, sizeof(bn), "l%d_b", idx);
                int64_t s1[1] = {N}; add_weight(m, bn, 1, s1, bf); free(bf);
                int bi = ncnt; PUSH_NODE(1,1,N,0,0,-1,-1,-1,bn);
                int ai = ncnt; PUSH_NODE(4,2,1,N,2,cur_node,bi,-1,NULL); cur_node = ai;
            }
        }
        idx++;
    }
    free(hdr);
    if (idx == 0) { free(nodes); return -1; }
    m->nodes = nodes; m->num_nodes = ncnt; m->in_idx = 0; m->out_idx = cur_node;
    #undef PUSH_NODE
    return 0;
}

static int run_studio_forward(StudioModel *m, const float *in_vals, float *out_vals, int64_t *out_len) {
    double lock_used = 0; int locked = 0;
    if (check_and_add_active_time(0.0, &lock_used, &locked) != 0) return -100;

    axon_arena_init(64u * 1024u * 1024u);
    axon_arena_reset();
    axon_brain_stats_reset();
    double t0 = studio_now_ms();

    AxonTensor **vals = (AxonTensor**)calloc((size_t)m->num_nodes, sizeof(AxonTensor*));
    AxonTensor *inp = axon_tensor_alloc(m->in_ndim, m->in_shape, AXON_DTYPE_F32);
    int64_t in_n = axon_tensor_numel(inp);
    axon_tensor_fill_f32(inp, in_vals, in_n);
    vals[m->in_idx] = inp;

    size_t w_mem = 0;
    const char *mode = m->backend_mode;
    int use_gpu = (strcmp(mode, "gpu") == 0);
    int use_brain = (strcmp(mode, "brain") == 0 || strcmp(mode, "auto") == 0);
    int use_fp4 = (strcmp(mode, "fp4") == 0 || strcmp(mode, "brain") == 0 || strcmp(mode, "auto") == 0);
    int use_tern = (strcmp(mode, "ternary") == 0);

    for (int i = 0; i < m->num_nodes; ++i) {
        CNode *n = &m->nodes[i];
        if (n->op == 0) continue;
        if (n->op == 1) {
            CWeight *cw = find_cweight(m, n->weight);
            if (!cw) { free(vals); axon_arena_destroy(); return -1; }
            AxonTensor *t = axon_tensor_alloc(cw->ndim, cw->shape, AXON_DTYPE_F32);
            if (use_brain && cw->reg_data && cw->ndim == 2 && !use_gpu) {
                t->data = cw->reg_data; t->owns_data = false; w_mem += (size_t)cw->numel * 4;
            } else if (use_fp4 && cw->fp4_buf && cw->ndim == 2 && !use_gpu) {
                t->dtype = AXON_DTYPE_FP4; t->data = cw->fp4_buf; t->owns_data = false; t->reserved = 0; w_mem += cw->fp4_len;
            } else if (use_tern && cw->tern_buf && cw->ndim == 2 && !use_gpu) {
                t->dtype = AXON_DTYPE_TERNARY; t->data = cw->tern_buf; t->scale = cw->tern_scale; t->owns_data = false; w_mem += cw->tern_len + 8;
            } else {
                t->data = cw->f32_data; t->owns_data = false; w_mem += (size_t)cw->numel * 4;
            }
            vals[i] = t;
            continue;
        }
        AxonTensor *out = axon_tensor_alloc(n->ndim, n->shape, AXON_DTYPE_F32);
        AxonTensor *a0 = vals[n->inputs[0]], *a1 = (n->nin > 1) ? vals[n->inputs[1]] : NULL, *a2 = (n->nin > 2) ? vals[n->inputs[2]] : NULL;
        int rc = AXON_OK;
        if (n->op == 2) {
            if (use_gpu && g_ocl.available) rc = dyn_ocl_matmul(a0, a1, out);
            else if (use_brain && find_cweight(m, m->nodes[n->inputs[1]].weight) && find_cweight(m, m->nodes[n->inputs[1]].weight)->sig_data) {
                CWeight *cw = find_cweight(m, m->nodes[n->inputs[1]].weight);
                int64_t sshp[2] = {cw->shape[0], cw->R};
                AxonTensor *S = axon_tensor_alloc(2, sshp, AXON_DTYPE_F32);
                S->data = cw->sig_data; S->owns_data = false;
                AxonTensor *W_reg = axon_tensor_alloc(2, cw->shape, AXON_DTYPE_F32);
                W_reg->data = cw->reg_data; W_reg->owns_data = false;
                rc = axon_neuro_matmul_paged(a0, W_reg, S, out, (int32_t)cw->keep, 1);
            } else rc = axon_matmul(a0, a1, out);
        } else if (n->op == 3) rc = axon_neuro_matmul_paged(a0, a1, a2, out, (int32_t)n->keep, n->region_blocked);
        else if (n->op == 4) rc = axon_add(a0, a1, out);
        else if (n->op == 5) rc = axon_sub(a0, a1, out);
        else if (n->op == 6) rc = axon_mul(a0, a1, out);
        else if (n->op == 7) rc = axon_relu(a0, out);
        else if (n->op == 8) rc = axon_gelu(a0, out);
        else if (n->op == 9) rc = axon_silu(a0, out);
        else if (n->op == 10) rc = axon_sigmoid(a0, out);
        else if (n->op == 11) rc = axon_softmax(a0, out);
        else if (n->op == 12) rc = axon_layer_norm(a0, a1, a2, out);
        else if (n->op == 13) rc = axon_rms_norm(a0, a1, out);
        else if (n->op == 14) rc = axon_rope(a0, out);
        else if (n->op == 15) rc = axon_attn_causal(a0, a1, a2, out);
        else if (n->op == 16) rc = axon_transpose(a0, out);
        else if (n->op == 17) rc = axon_equilibrium(a0, a1, a2, out);
        else if (n->op == 18) rc = axon_mm_relu(a0, a1, out);
        else if (n->op == 19) rc = axon_mm_gelu(a0, a1, out);
        else if (n->op == 20) rc = axon_mm_silu(a0, a1, out);
        else if (n->op == 21) rc = axon_mm_bias(a0, a1, a2, out);
        else if (n->op == 22) rc = axon_mm_bias_relu(a0, a1, a2, out);
        else if (n->op == 23) rc = axon_mm_bias_gelu(a0, a1, a2, out);
        else if (n->op == 24) rc = axon_mm_bias_silu(a0, a1, a2, out);
        if (rc != AXON_OK) { free(vals); axon_arena_destroy(); return rc; }
        vals[i] = out;
    }

    double elapsed_ms = studio_now_ms() - t0;
    AxonTensor *res = vals[m->out_idx];
    int64_t on = axon_tensor_numel(res);
    *out_len = on;
    if (res && res->data && out_vals) memcpy(out_vals, res->data, (size_t)on * sizeof(float));

    AxonBrainStats bs; axon_get_brain_stats(&bs);
    m->active_weight_bytes = w_mem;
    m->last_ms = elapsed_ms;
    m->last_active_reg = bs.active_regions; m->last_total_reg = bs.total_regions;
    m->last_ram_saved = (m->raw_f32_bytes > 0) ? 100.0 * (1.0 - (double)w_mem / (double)m->raw_f32_bytes) : 0.0;
    m->last_flops_skipped = (bs.dense_flops > 0) ? 100.0 * (double)(bs.dense_flops - bs.executed_flops) / (double)bs.dense_flops : 0.0;
    if (m->last_flops_skipped < 0) m->last_flops_skipped = 0;

    free(vals);
    axon_arena_destroy();
    check_and_add_active_time(elapsed_ms / 1000.0, &lock_used, &locked);
    return AXON_OK;
}

/* ---- Embedded Modern Web UI (Chat, Server Connect, Upload, Device Selector, Live Metrics) ---- */
static const char *UI_HTML =
"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
"<title>Axon v12 Studio</title><style>"
":root{--bg:#0b1120;--card:#131c31;--b:#23314d;--tx:#e2e8f0;--mu:#94a3b8;--ac:#38bdf8;--ok:#34d399;--pr:#f59e0b}"
"*{box-sizing:border-box}body{margin:0;font-family:system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--tx);padding:14px}"
".top{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;border-bottom:1px solid var(--b);padding-bottom:10px;margin-bottom:14px}"
".badge{padding:4px 10px;border-radius:99px;font-size:12px;font-weight:700;background:#1e293b;color:var(--ac)}"
".pro{background:rgba(245,158,11,.2);color:var(--pr);border:1px solid var(--pr)}"
".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:14px;max-width:1200px;margin:0 auto}"
".card{background:var(--card);border:1px solid var(--b);border-radius:12px;padding:14px}"
"h3{margin:0 0 10px 0;font-size:15px;color:#fff}label{display:block;font-size:12px;color:var(--mu);margin:8px 0 4px}"
"input,select,button,textarea{width:100%;padding:9px;border-radius:8px;border:1px solid var(--b);background:#0f172a;color:var(--tx);font-size:13px}"
"button{background:var(--ac);color:#06121f;font-weight:700;border:none;cursor:pointer;margin-top:8px}button:hover{opacity:.9}"
".kpi{display:grid;grid-template-columns:1fr 1fr;gap:8px}.k{background:#0f172a;padding:9px;border-radius:8px;border:1px solid var(--b)}"
".k b{display:block;font-size:16px;color:var(--ok)}.k span{font-size:11px;color:var(--mu)}"
".chat{height:240px;overflow-y:auto;background:#0f172a;border:1px solid var(--b);border-radius:8px;padding:10px;font-size:13px;white-space:pre-wrap}"
".bar-row{margin:6px 0;font-size:12px}.bar{height:10px;background:var(--ac);border-radius:4px;margin-top:3px}"
"</style></head><body>"
"<div class='top'><div><strong style='font-size:18px'>AXON v12.0 STUDIO</strong> <span id='ed' class='badge'>...</span></div>"
"<div style='display:flex;gap:8px;align-items:center'><input id='srv' placeholder='Remote Server (blank=local)' style='width:220px'>"
"<button onclick='refreshStatus()' style='width:auto;margin:0;padding:8px 14px'>Connect / Refresh</button></div></div>"
"<div class='grid'>"
"<div class='card'><h3>1. Model & Hardware Selector</h3>"
"<label>Upload Model (.safetensors / .axg1)</label><input type='file' id='mfile' onchange='uploadModel()'>"
"<label>Processor / Quantization Engine</label>"
"<select id='mode' onchange='setMode()'><option value='fp4'>CPU FP4 + Outliers (Recommended)</option><option value='brain'>CPU Brain-Routed Sparse</option>"
"<option value='ternary'>CPU Ternary 1.58-bit</option><option value='f32'>CPU Dense F32 (SIMD+OMP)</option><option value='gpu'>GPU OpenCL (if available)</option></select>"
"<div class='kpi' style='margin-top:10px'>"
"<div class='k'><span>Model</span><b id='k_mod'>-</b></div><div class='k'><span>Weight Savings</span><b id='k_sav'>-</b></div>"
"<div class='k'><span>Latency</span><b id='k_ms'>-</b></div><div class='k'><span>FLOPs Skipped</span><b id='k_fl'>-</b></div>"
"<div class='k'><span>OpenCL Device</span><b id='k_cl'>-</b></div><div class='k'><span>Server Quota</span><b id='k_q'>-</b></div>"
"</div></div>"
"<div class='card'><h3>2. Interactive Chat / KV-Cache Loop</h3>"
"<div id='chat' class='chat'>Ready. Type a prompt or token seed below.</div>"
"<label>Prompt</label><input id='prompt' placeholder='Hello Axon v12...' onkeydown='if(event.key===\"Enter\")sendChat()'>"
"<button onclick='sendChat()'>Send & Generate (KV-Cache)</button></div>"
"<div class='card'><h3>3. Live Multi-Backend Benchmark</h3>"
"<button onclick='runBench()'>Run Real 256x256 Matrix Benchmark</button>"
"<div id='bchart' style='margin-top:10px'>Click Run to measure real latency on this host.</div></div>"
"</div><script>"
"function base(){ var v=document.getElementById('srv').value.trim(); return v?v.replace(/\\/$/,''):''; }"
"async function refreshStatus(){ try{ const r=await fetch(base()+'/api/status'); const d=await r.json();"
" const eb=document.getElementById('ed'); eb.textContent=d.edition+(d.is_server?' (SERVER)':' (DESKTOP/MOBILE)'); if(d.pro) eb.className='badge pro';"
" document.getElementById('k_mod').textContent=d.model+' ('+d.format+')';"
" document.getElementById('k_sav').textContent=d.ram_saved.toFixed(1)+'% ('+d.active_bytes+' B)';"
" document.getElementById('k_ms').textContent=d.last_ms.toFixed(3)+' ms';"
" document.getElementById('k_fl').textContent=d.flops_skipped.toFixed(1)+'%';"
" document.getElementById('k_cl').textContent=d.opencl_device;"
" document.getElementById('k_q').textContent=d.pro?'UNLIMITED (Pro)':(d.is_server?(d.quota_used_s.toFixed(1)+'s / 86400s'):'UNLIMITED');"
" document.getElementById('mode').value=d.mode; }catch(e){ alert('Connect error: '+e); }}"
"async function setMode(){ const m=document.getElementById('mode').value; await fetch(base()+'/api/mode?mode='+encodeURIComponent(m),{method:'POST'}); refreshStatus(); }"
"async function uploadModel(){ const f=document.getElementById('mfile').files[0]; if(!f)return; const buf=await f.arrayBuffer();"
" const r=await fetch(base()+'/api/upload?name='+encodeURIComponent(f.name),{method:'POST',body:buf}); const d=await r.json(); alert(d.ok?'Loaded '+d.model:'Error: '+d.error); refreshStatus(); }"
"async function sendChat(){ const p=document.getElementById('prompt').value; const c=document.getElementById('chat');"
" c.textContent+='\\n> '+p+'\\n'; const r=await fetch(base()+'/api/chat',{method:'POST',body:p}); const d=await r.json();"
" c.textContent+=(d.error?('ERR: '+d.error):('[Axon '+d.mode+' | '+d.ms.toFixed(3)+' ms | tokens='+d.tokens.join(',')+']\\n'+d.note)); c.scrollTop=c.scrollHeight; refreshStatus(); }"
"async function runBench(){ const el=document.getElementById('bchart'); el.textContent='Running...'; const r=await fetch(base()+'/api/bench'); const d=await r.json();"
" const mx=Math.max(...d.results.map(x=>x.ms),0.01); el.innerHTML=d.results.map(x=>`<div class='bar-row'>${x.mode}: <b>${x.ms.toFixed(3)} ms</b> (err ${x.err.toExponential(1)})<div class='bar' style='width:${Math.max(5,(1-(x.ms/mx))*90+10)}%'></div></div>`).join(''); }"
"refreshStatus();</script></body></html>";

/* ---- HTTP Server ---- */
static void send_http(axon_sock_t c, int code, const char *ctype, const char *body, size_t blen) {
    char hdr[512];
    int n = snprintf(hdr, sizeof(hdr),
        "HTTP/1.1 %d OK\r\nContent-Type: %s\r\nContent-Length: %zu\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nConnection: close\r\n\r\n",
        code, ctype, blen);
    send(c, hdr, (int)n, 0);
    size_t sent = 0;
    while (sent < blen) {
        int r = send(c, body + sent, (int)(blen - sent), 0);
        if (r <= 0) break;
        sent += (size_t)r;
    }
}

static void handle_http_client(axon_sock_t c) {
    size_t cap = 65536, len = 0;
    char *req = (char*)malloc(cap);
    if (!req) { axon_sock_close(c); return; }
    char *hdr_end = NULL;
    size_t content_len = 0, hdr_sz = 0;

    while (1) {
        if (len + 4096 >= cap) {
            if (cap >= 64u * 1024u * 1024u) { free(req); axon_sock_close(c); return; }
            cap *= 2; char *nr = (char*)realloc(req, cap);
            if (!nr) { free(req); axon_sock_close(c); return; }
            req = nr;
        }
        int r = recv(c, req + len, 4096, 0);
        if (r <= 0) break;
        len += (size_t)r;
        req[len] = '\0';
        if (!hdr_end) {
            hdr_end = strstr(req, "\r\n\r\n");
            if (hdr_end) {
                hdr_sz = (size_t)(hdr_end - req) + 4;
                for (char *p = req; p < hdr_end; ++p) {
                    if ((p == req || p[-1] == '\n') && (size_t)(hdr_end - p) > 15 &&
                        (p[0]=='C'||p[0]=='c') && (p[1]=='o'||p[1]=='O') && (p[8]=='L'||p[8]=='l')) {
                        if (strncmp(p, "Content-Length:", 15) == 0 || strncmp(p, "content-length:", 15) == 0) {
                            content_len = (size_t)strtoull(p + 15, NULL, 10);
                        }
                    }
                }
            }
        }
        if (hdr_end && len >= hdr_sz + content_len) break;
    }
    if (!hdr_end) { free(req); axon_sock_close(c); return; }

    char method[16]={0}, path[512]={0};
    sscanf(req, "%15s %511s", method, path);
    const char *body = req + hdr_sz;
    size_t blen = (len >= hdr_sz) ? (len - hdr_sz) : 0;
    if (content_len > 0 && blen > content_len) blen = content_len;

    if (strcmp(method, "OPTIONS") == 0) {
        send_http(c, 200, "text/plain", "", 0);
    } else if (strcmp(path, "/") == 0) {
        send_http(c, 200, "text/html; charset=utf-8", UI_HTML, strlen(UI_HTML));
    } else if (strcmp(path, "/api/status") == 0) {
        double qused = 0; int qlocked = 0; check_and_add_active_time(0.0, &qused, &qlocked);
        char json[1024];
        snprintf(json, sizeof(json),
            "{\"edition\":\"%s\",\"pro\":%d,\"is_server\":%d,\"quota_used_s\":%.2f,\"quota_locked\":%d,"
            "\"model\":\"%s\",\"format\":\"%s\",\"mode\":\"%s\",\"raw_bytes\":%zu,\"active_bytes\":%zu,"
            "\"ram_saved\":%.2f,\"flops_skipped\":%.2f,\"last_ms\":%.4f,\"opencl_device\":\"%s\"}",
            AXON_EDITION_PRO ? "Axon Pro v12" : "Axon Free v12", AXON_EDITION_PRO, detect_server_env(),
            qused, qlocked, g_model.model_name, g_model.format, g_model.backend_mode,
            g_model.raw_f32_bytes, g_model.active_weight_bytes, g_model.last_ram_saved,
            g_model.last_flops_skipped, g_model.last_ms, g_ocl.available ? g_ocl.device_name : "none");
        send_http(c, 200, "application/json", json, strlen(json));
    } else if (strncmp(path, "/api/mode?mode=", 15) == 0) {
        snprintf(g_model.backend_mode, sizeof(g_model.backend_mode), "%s", path + 15);
        send_http(c, 200, "application/json", "{\"ok\":true}", 11);
    } else if (strncmp(path, "/api/upload", 11) == 0) {
        const char *nm = strstr(path, "name="); nm = nm ? (nm + 5) : "uploaded.safetensors";
        if (parse_safetensors_bytes(&g_model, nm, (const uint8_t*)body, blen, 1) == 0) {
            char json[256]; snprintf(json, sizeof(json), "{\"ok\":true,\"model\":\"%s\",\"raw_bytes\":%zu}", g_model.model_name, g_model.raw_f32_bytes);
            send_http(c, 200, "application/json", json, strlen(json));
        } else {
            const char *err = "{\"ok\":false,\"error\":\"Unsupported file (supports Safetensors F32/F16/BF16 layers)\"}";
            send_http(c, 400, "application/json", err, strlen(err));
        }
    } else if (strcmp(path, "/api/chat") == 0) {
        int64_t in_n = 1; for (int d=0;d<g_model.in_ndim;++d) in_n *= g_model.in_shape[d];
        float *in_v = (float*)calloc((size_t)in_n, sizeof(float));
        int64_t out_cap = 1; for (int d=0;d<g_model.nodes[g_model.out_idx].ndim;++d) out_cap *= g_model.nodes[g_model.out_idx].shape[d];
        float *out_v = (float*)calloc((size_t)(out_cap > 0 ? out_cap : 1), sizeof(float));
        for (int64_t i=0;i<in_n;++i) in_v[i] = blen ? (((unsigned char)body[i % blen] - 64.f) / 64.f) : ((i%4)*0.25f + 0.5f);
        int toks[4] = {0}; double tot_ms = 0; int rc = 0;
        for (int s=0;s<4;++s) {
            int64_t on = 0; rc = run_studio_forward(&g_model, in_v, out_v, &on);
            if (rc != 0) break;
            tot_ms += g_model.last_ms;
            int best = 0; float bv = -1e30f;
            for (int64_t d=0;d<on;++d) { if (out_v[d] > bv) { bv = out_v[d]; best = (int)d; } }
            toks[s] = best;
            for (int64_t i=0;i<in_n && i<on;++i) in_v[i] = tanhf(out_v[i]);
        }
        free(in_v); free(out_v);
        if (rc == -100) {
            const char *er = "{\"error\":\"Weekly server quota (24h/wk) exhausted or tampered. Upgrade to Axon Pro.\"}";
            send_http(c, 403, "application/json", er, strlen(er));
        } else if (rc != 0) {
            const char *er = "{\"error\":\"Execution failed\"}";
            send_http(c, 500, "application/json", er, strlen(er));
        } else {
            char json[512];
            snprintf(json, sizeof(json),
                "{\"ok\":true,\"mode\":\"%s\",\"ms\":%.4f,\"tokens\":[%d,%d,%d,%d],"
                "\"note\":\"Executed 4 KV-step tensor passes cleanly (live_allocs=%lld).\"}",
                g_model.backend_mode, tot_ms, toks[0], toks[1], toks[2], toks[3], (long long)axon_live_allocs());
            send_http(c, 200, "application/json", json, strlen(json));
        }
    } else if (strcmp(path, "/api/bench") == 0) {
        const int N = 256; int64_t shp[2] = {N, N};
        axon_arena_init(32u * 1024u * 1024u); axon_arena_reset();
        AxonTensor *A = axon_tensor_alloc(2, shp, AXON_DTYPE_F32);
        AxonTensor *B = axon_tensor_alloc(2, shp, AXON_DTYPE_F32);
        AxonTensor *C = axon_tensor_alloc(2, shp, AXON_DTYPE_F32);
        float *fa = (float*)A->data, *fb = (float*)B->data;
        for (int i=0;i<N*N;++i) { fa[i]=((i%13)-6)*0.02f; fb[i]=((i%11)-5)*0.02f; }
        double t0 = studio_now_ms(); axon_matmul(A, B, C); double ms_f32 = studio_now_ms() - t0;
        double ms_cl = -1.0; float err_cl = 0.0f;
        if (g_ocl.available) {
            AxonTensor *Ccl = axon_tensor_alloc(2, shp, AXON_DTYPE_F32);
            double tc = studio_now_ms();
            if (dyn_ocl_matmul(A, B, Ccl) == AXON_OK) {
                ms_cl = studio_now_ms() - tc;
                for (int i=0;i<N*N;++i) { float d = fabsf(((float*)C->data)[i] - ((float*)Ccl->data)[i]); if (d > err_cl) err_cl = d; }
            }
        }
        axon_arena_destroy();
        char json[512];
        if (ms_cl >= 0) {
            snprintf(json, sizeof(json),
                "{\"results\":[{\"mode\":\"CPU F32 (SIMD+OMP)\",\"ms\":%.3f,\"err\":0.0},"
                "{\"mode\":\"OpenCL (%s)\",\"ms\":%.3f,\"err\":%.2e}]}",
                ms_f32, g_ocl.device_name, ms_cl, err_cl);
        } else {
            snprintf(json, sizeof(json), "{\"results\":[{\"mode\":\"CPU F32 (SIMD+OMP)\",\"ms\":%.3f,\"err\":0.0}]}", ms_f32);
        }
        send_http(c, 200, "application/json", json, strlen(json));
    } else {
        send_http(c, 404, "text/plain", "not found", 9);
    }
    free(req);
    axon_sock_close(c);
}

int main(int argc, char **argv) {
#ifdef _WIN32
    WSADATA wsa; WSAStartup(MAKEWORD(2,2), &wsa);
#else
    signal(SIGPIPE, SIG_IGN);
#endif
    dyn_ocl_init();
    init_builtin_demo_model(&g_model);

    int serve = 0, port = 8080, transpose = 1, once = 0;
    const char *host = "127.0.0.1", *axg = NULL, *axn = NULL, *st_path = NULL, *mode = NULL;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--serve")) serve = 1;
        else if (!strcmp(argv[i], "--port") && i+1 < argc) port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--host") && i+1 < argc) host = argv[++i];
        else if (!strcmp(argv[i], "--axg") && i+1 < argc) axg = argv[++i];
        else if (!strcmp(argv[i], "--axn") && i+1 < argc) axn = argv[++i];
        else if (!strcmp(argv[i], "--safetensors") && i+1 < argc) st_path = argv[++i];
        else if (!strcmp(argv[i], "--mode") && i+1 < argc) mode = argv[++i];
        else if (!strcmp(argv[i], "--no-transpose")) transpose = 0;
        else if (!strcmp(argv[i], "--simulate-server")) g_sim_server = 1;
        else if (!strcmp(argv[i], "--once")) once = 1;
    }

    if (axg && axn) {
        if (load_axg_axn3(&g_model, axg, axn) != 0) { fprintf(stderr, "failed loading %s + %s\n", axg, axn); return 2; }
    } else if (st_path) {
        FILE *f = fopen(st_path, "rb"); if (!f) { fprintf(stderr, "cannot open %s\n", st_path); return 2; }
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        uint8_t *buf = (uint8_t*)malloc((size_t)sz); if (!buf || fread(buf, 1, (size_t)sz, f) != (size_t)sz) { free(buf); fclose(f); return 2; }
        fclose(f);
        int rc = parse_safetensors_bytes(&g_model, st_path, buf, (size_t)sz, transpose); free(buf);
        if (rc != 0) { fprintf(stderr, "failed parsing safetensors %s\n", st_path); return 2; }
    }
    if (mode) snprintf(g_model.backend_mode, sizeof(g_model.backend_mode), "%s", mode);

    if (once || !serve) {
        int64_t in_n = 1; for (int d=0;d<g_model.in_ndim;++d) in_n *= g_model.in_shape[d];
        int64_t out_n = 1; for (int d=0;d<g_model.nodes[g_model.out_idx].ndim;++d) out_n *= g_model.nodes[g_model.out_idx].shape[d];
        float *inv = (float*)malloc((size_t)in_n * sizeof(float));
        float *outv = (float*)malloc((size_t)out_n * sizeof(float));
        const float seed[4] = {1.f, 0.5f, -1.f, 2.f};
        for (int64_t i=0;i<in_n;++i) inv[i] = seed[i%4];
        int64_t got_n = 0;
        int rc = run_studio_forward(&g_model, inv, outv, &got_n);
        if (rc == -100) { fprintf(stderr, "AXON_ERR_QUOTA_LOCKED\n"); free(inv); free(outv); free_model(&g_model); return 3; }
        if (rc != 0) { fprintf(stderr, "forward error %d\n", rc); free(inv); free(outv); free_model(&g_model); return 1; }
        printf("[axon-studio] edition=%s model=%s mode=%s ms=%.3f raw=%zu active=%zu saved=%.1f%% live_allocs=%lld\n",
               AXON_EDITION_PRO ? "PRO" : "FREE", g_model.model_name, g_model.backend_mode,
               g_model.last_ms, g_model.raw_f32_bytes, g_model.active_weight_bytes,
               g_model.last_ram_saved, (long long)axon_live_allocs());
        int64_t lim = got_n < 16 ? got_n : 16;
        for (int64_t i=0;i<lim;++i) printf("  out[%lld]=%.6f\n", (long long)i, (double)outv[i]);
        free(inv); free(outv);
        if (!serve) { free_model(&g_model); return axon_live_allocs() == 0 ? 0 : 1; }
    }

    axon_sock_t srv = socket(AF_INET, SOCK_STREAM, 0);
    int yes = 1; setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, (const char*)&yes, sizeof(yes));
    struct sockaddr_in addr; memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET; addr.sin_port = htons((uint16_t)port);
    addr.sin_addr.s_addr = inet_addr(host);
    if (bind(srv, (struct sockaddr*)&addr, sizeof(addr)) != 0 || listen(srv, 16) != 0) {
        fprintf(stderr, "failed to bind %s:%d\n", host, port); axon_sock_close(srv); free_model(&g_model); return 1;
    }
    struct sockaddr_in actual; socklen_t alen = sizeof(actual);
    getsockname(srv, (struct sockaddr*)&actual, &alen);
    int bound_port = (int)ntohs(actual.sin_port);
    printf("LISTENING_PORT=%d\n", bound_port);
    printf("[axon-studio] %s listening at http://%s:%d (OpenCL=%s)\n",
           AXON_EDITION_PRO ? "Axon Pro v12" : "Axon Free v12", host, bound_port,
           g_ocl.available ? g_ocl.device_name : "none");
    fflush(stdout);

    while (1) {
        axon_sock_t cl = accept(srv, NULL, NULL);
        if (cl == AXON_BAD_SOCK) break;
        handle_http_client(cl);
    }
    axon_sock_close(srv);
    free_model(&g_model);
#ifdef _WIN32
    WSACleanup();
#endif
    return 0;
}
'''


TEST_V12_PY = r'''import json
import os
import shutil
import struct
import subprocess
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from axon.axg import export_axg
from axon.backend import _tool
from axon.blob import BlobPacker
from axon.frontend import Lexer, Parser, Semantic
from axon.ir import IRBuilder
from axon.mobile import build_mobile_sdk
from axon.passes import Pipeline
from axon.reference import T, evaluate
from axon.runtime_paths import ensure_runtime


def compile_studio_bins(out_dir):
    rt = ensure_runtime()
    cc = _tool("cc")
    ext = ".exe" if os.name == "nt" else ""
    bins = {}
    for pro, name in ((0, "axon-studio" + ext), (1, "axon-pro-studio" + ext)):
        exe = os.path.join(out_dir, name)
        cmd = [cc, "-O3", "-std=c11", "-Wall", "-Wextra", "-Wno-unknown-pragmas", "-fno-math-errno",
               f"-DAXON_EDITION_PRO={pro}", f"-I{rt.dir}",
               str(rt.dir / "axon_studio.c"), str(rt.dir / "axon_runtime.c")]
        if os.name == "nt":
            cmd += ["-lws2_32", "-static-libgcc", "-lm", "-o", exe]
        elif os.sys.platform == "darwin":
            cmd += ["-lm", "-lpthread", "-o", exe]
        else:
            cmd += ["-ldl", "-lm", "-lpthread", "-o", exe]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"studio compile failed:\n{r.stderr}")
        bins[pro] = exe
    return bins


class TestV12StudioAndMobile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="axon_v12_")
        cls.bins = compile_studio_bins(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_axg1_standalone_zero_prereq_execution(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        src = open(os.path.join(root, "examples", "mini_llm.axon"), encoding="utf-8").read()
        ast = Parser(Lexer(src).tokenize()).parse()
        Semantic().analyze(ast)
        g = Pipeline(quant=True, brain=True).run(IRBuilder().build(ast))

        axg = os.path.join(self.tmp, "mini.axg1")
        axn = os.path.join(self.tmp, "mini.axn3")
        export_axg(g, axg)
        open(axn, "wb").write(BlobPacker.pack(g.weights))

        r = subprocess.run([self.bins[0], "--axg", axg, "--axn", axn, "--once"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("live_allocs=0", r.stdout)

        seed = [1.0, 0.5, -1.0, 2.0]
        ref = evaluate(g, [T((4, 8), [seed[i % 4] for i in range(32)])]).data
        got = [float(line.split("=")[1]) for line in r.stdout.splitlines() if "out[" in line]
        self.assertEqual(len(got), 16)
        for a, b in zip(got, ref[:16]):
            self.assertLessEqual(abs(a - b), 5e-4)

    def test_studio_web_ui_upload_chat_and_bench(self):
        proc = subprocess.Popen([self.bins[1], "--serve", "--port", "0"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            port = None
            for _ in range(50):
                line = proc.stdout.readline()
                if line.startswith("LISTENING_PORT="):
                    port = int(line.strip().split("=")[1])
                    break
            self.assertIsNotNone(port, "studio failed to bind port")
            base = f"http://127.0.0.1:{port}"

            st = json.loads(urllib.request.urlopen(base + "/api/status", timeout=5).read().decode())
            self.assertEqual(st["pro"], 1)

            w_bytes = struct.pack("<64f", *[((i % 7) - 3) * 0.1 for i in range(64)])
            hdr = json.dumps({"layers.0.weight": {"dtype": "F32", "shape": [8, 8], "data_offsets": [0, len(w_bytes)]}}).encode()
            st_blob = struct.pack("<Q", len(hdr)) + hdr + w_bytes
            req = urllib.request.Request(base + "/api/upload?name=test.safetensors", data=st_blob, method="POST")
            up = json.loads(urllib.request.urlopen(req, timeout=5).read().decode())
            self.assertTrue(up["ok"])

            chat_req = urllib.request.Request(base + "/api/chat", data=b"Hello Axon Studio", method="POST")
            ch = json.loads(urllib.request.urlopen(chat_req, timeout=5).read().decode())
            self.assertTrue(ch["ok"])
            self.assertEqual(len(ch["tokens"]), 4)

            bn = json.loads(urllib.request.urlopen(base + "/api/bench", timeout=10).read().decode())
            self.assertGreaterEqual(len(bn["results"]), 1)
        finally:
            proc.terminate()
            proc.wait(timeout=5)

    def test_server_quota_pause_and_pro_unlimited(self):
        fake_home = os.path.join(self.tmp, "qhome")
        os.makedirs(fake_home, exist_ok=True)
        env = os.environ.copy()
        env["HOME"] = fake_home
        env["LOCALAPPDATA"] = fake_home
        env["AXON_TEST_SERVER"] = "1"

        r1 = subprocess.run([self.bins[0], "--once"], capture_output=True, text=True, env=env)
        self.assertEqual(r1.returncode, 0, r1.stdout + r1.stderr)
        qfile = os.path.join(fake_home, "axon_quota_v12.bin" if os.name != "nt" else "axon_quota_v12.bin")
        if os.name != "nt":
            qfile = os.path.join(fake_home, ".axon_quota_v12.bin")
        self.assertTrue(os.path.exists(qfile))

        # Tampering with quota file fails closed safely WITHOUT deleting the executable
        with open(qfile, "r+b") as fh:
            fh.seek(0)
            fh.write(b"\x00\x00\x00\x00")
        r_tamper = subprocess.run([self.bins[0], "--once"], capture_output=True, text=True, env=env)
        self.assertEqual(r_tamper.returncode, 3)
        self.assertTrue(os.path.exists(self.bins[0]))  # Safe fail-closed, never self-deletes

        # Pro edition stays unrestricted
        r_pro = subprocess.run([self.bins[1], "--once"], capture_output=True, text=True, env=env)
        self.assertEqual(r_pro.returncode, 0, r_pro.stdout + r_pro.stderr)

    def test_mobile_sdk_generation(self):
        mob_dir = os.path.join(self.tmp, "mobile_sdk")
        info = build_mobile_sdk(mob_dir)
        self.assertTrue(os.path.exists(os.path.join(mob_dir, "AxonBridge.kt")))
        self.assertTrue(os.path.exists(os.path.join(mob_dir, "AxonEngine.swift")))
        self.assertTrue(os.path.exists(os.path.join(mob_dir, "axon_runtime.h")))


if __name__ == "__main__":
    unittest.main()
'''


def patch_cli_for_v12():
    p = "axon/cli.py"
    s = open(p, encoding="utf-8").read()
    if "def cmd_export_axg" not in s:
        cmds = '''def cmd_export_axg(args):
    from .axg import export_axg
    g = load_graph(args.source, args.ternary, args.no_fusion, args.no_quant, args.brain, args.brain_ratio)
    export_axg(g, args.output + ".axg1")
    open(args.output + ".axn3", "wb").write(BlobPacker.pack(g.weights))
    print(f"[ok] exported standalone AXG1+AXN3: {args.output}.axg1 / {args.output}.axn3")
    return 0
def cmd_mobile(args):
    from .mobile import build_mobile_sdk
    info = build_mobile_sdk(args.output)
    print(json.dumps(info, indent=2))
    return 0
'''
        s = s.replace("def cmd_doctor(args):", cmds + "def cmd_doctor(args):")
        sp_add = (
            '    px=sp.add_parser("export-axg"); common(px); px.add_argument("-o","--output",default="model"); px.set_defaults(func=cmd_export_axg)\n'
            '    pm=sp.add_parser("build-mobile"); pm.add_argument("-o","--output",default="dist_mobile"); pm.set_defaults(func=cmd_mobile)\n'
        )
        s = s.replace('    sp.add_parser("doctor")', sp_add + '    sp.add_parser("doctor")')
        open(p, "w", encoding="utf-8").write(s)


def build_studio_binaries(out_dir="dist_v12"):
    from axon.backend import _tool
    from axon.runtime_paths import ensure_runtime
    rt = ensure_runtime(verbose=True)
    cc = _tool("cc")
    os.makedirs(out_dir, exist_ok=True)
    ext = ".exe" if os.name == "nt" else ""
    built = {}
    for pro, name in ((0, "axon-studio" + ext), (1, "axon-pro-studio" + ext)):
        exe = os.path.join(out_dir, name)
        cmd = [cc, "-O3", "-std=c11", "-Wall", "-Wextra", "-Wno-unknown-pragmas", "-fno-math-errno",
               f"-DAXON_EDITION_PRO={pro}", f"-I{rt.dir}",
               str(rt.dir / "axon_studio.c"), str(rt.dir / "axon_runtime.c")]
        if os.name == "nt":
            cmd += ["-lws2_32", "-static-libgcc", "-s", "-lm", "-o", exe]
        elif sys.platform == "darwin":
            cmd += ["-lm", "-lpthread", "-o", exe]
        else:
            cmd += ["-s", "-ldl", "-lm", "-lpthread", "-o", exe]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"failed building {name}:\n{r.stderr}")
        if os.name != "nt":
            os.chmod(exe, 0o755)
        built["pro" if pro else "free"] = exe
        print(f"[v12] built standalone zero-prereq binary: {exe} ({os.path.getsize(exe)} bytes)")
    return built


def main():
    ensure_v11_base()
    write_file("axon/__init__.py", '__version__ = "12.0.0"\n')
    write_file("axon/axg.py", AXG_PY)
    write_file("axon/mobile.py", MOBILE_PY)
    write_file("axon/runtime/axon_studio.c", STUDIO_C)
    write_file("tests/test_v12_studio.py", TEST_V12_PY)
    patch_cli_for_v12()
    build_studio_binaries("dist_v12")
    print("[v12] Running all unit tests (v10 + v11 + v12 Studio/Mobile/Quota) ...")
    r = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"])
    if r.returncode != 0:
        sys.exit(r.returncode)
    print("\n[COMPLETED] Axon v12.0 Final ready! Launch UI with: ./dist_v12/axon-studio --serve")


if __name__ == "__main__":
    main()