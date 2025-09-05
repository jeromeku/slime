// Minimal LD_PRELOAD shim to trace CUDA IPC calls.
// Intercepts: cudaIpcGetMemHandle, cudaIpcOpenMemHandle, cudaIpcCloseMemHandle
// Optional: cudaIpcGetEventHandle, cudaIpcOpenEventHandle
//
// Build:
//   make -C codex/weight-sync/shims
// Run:
//   LD_PRELOAD=$PWD/codex/weight-sync/shims/libcuda_ipc_trace.so \
//   CUDA_IPC_TRACE_LOG=/tmp/ipc.log \
//   python codex/weight-sync/demos/minimal_ipc_demo.py
//
// Notes:
// - We avoid including CUDA headers to reduce build deps; provide minimal typedefs.
// - Logging is to file if CUDA_IPC_TRACE_LOG is set, else to stderr.

#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <execinfo.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

// Minimal CUDA runtime ABI types
typedef int cudaError_t;
typedef struct { char reserved[64]; } cudaIpcMemHandle_t;    // size per CUDA spec
typedef struct { char reserved[64]; } cudaIpcEventHandle_t;  // likewise

// Function pointer types
typedef cudaError_t (*PFN_cudaIpcGetMemHandle)(cudaIpcMemHandle_t*, void*);
typedef cudaError_t (*PFN_cudaIpcOpenMemHandle)(void**, cudaIpcMemHandle_t, unsigned int);
typedef cudaError_t (*PFN_cudaIpcCloseMemHandle)(void*);
typedef cudaError_t (*PFN_cudaIpcGetEventHandle)(cudaIpcEventHandle_t*, void*);
typedef cudaError_t (*PFN_cudaIpcOpenEventHandle)(void**, cudaIpcEventHandle_t);

static PFN_cudaIpcGetMemHandle   real_cudaIpcGetMemHandle   = NULL;
static PFN_cudaIpcOpenMemHandle  real_cudaIpcOpenMemHandle  = NULL;
static PFN_cudaIpcCloseMemHandle real_cudaIpcCloseMemHandle = NULL;
static PFN_cudaIpcGetEventHandle real_cudaIpcGetEventHandle = NULL;
static PFN_cudaIpcOpenEventHandle real_cudaIpcOpenEventHandle = NULL;

static FILE* logf = NULL;
static pthread_mutex_t log_mu = PTHREAD_MUTEX_INITIALIZER;

static long get_tid(void) {
    return (long)syscall(SYS_gettid);
}

static void now_str(char* buf, size_t n) {
    struct timeval tv; gettimeofday(&tv, NULL);
    struct tm tm; localtime_r(&tv.tv_sec, &tm);
    snprintf(buf, n, "%04d-%02d-%02d %02d:%02d:%02d.%03ld",
             tm.tm_year+1900, tm.tm_mon+1, tm.tm_mday,
             tm.tm_hour, tm.tm_min, tm.tm_sec, (long)tv.tv_usec/1000);
}

static void log_line(const char* fmt, ...) {
    if (!logf) return;
    char ts[64]; now_str(ts, sizeof ts);
    pthread_mutex_lock(&log_mu);
    fprintf(logf, "[%s] pid=%ld tid=%ld ", ts, (long)getpid(), get_tid());
    va_list ap; va_start(ap, fmt); vfprintf(logf, fmt, ap); va_end(ap);
    fputc('\n', logf);
    fflush(logf);
    pthread_mutex_unlock(&log_mu);
}

static void log_bt_if_enabled(void) {
    const char* en = getenv("CUDA_IPC_TRACE_STACK");
    if (!en || en[0] == '\0') return;
    void* bt[64]; int n = backtrace(bt, 64);
    char** syms = backtrace_symbols(bt, n);
    if (!syms) return;
    pthread_mutex_lock(&log_mu);
    for (int i = 0; i < n; ++i) fprintf(logf, "  [bt] %s\n", syms[i]);
    fflush(logf);
    pthread_mutex_unlock(&log_mu);
    free(syms);
}

__attribute__((constructor)) static void init(void) {
    const char* path = getenv("CUDA_IPC_TRACE_LOG");
    if (path && *path) {
        logf = fopen(path, "a");
        if (!logf) logf = stderr;
    } else {
        logf = stderr;
    }
    // Resolve real symbols lazily; we'll do it on first call to avoid load-order issues.
    log_line("[shim] cuda_ipc_trace loaded");
}

static void resolve_syms(void) {
    if (!real_cudaIpcGetMemHandle)
        real_cudaIpcGetMemHandle = (PFN_cudaIpcGetMemHandle)dlsym(RTLD_NEXT, "cudaIpcGetMemHandle");
    if (!real_cudaIpcOpenMemHandle)
        real_cudaIpcOpenMemHandle = (PFN_cudaIpcOpenMemHandle)dlsym(RTLD_NEXT, "cudaIpcOpenMemHandle");
    if (!real_cudaIpcCloseMemHandle)
        real_cudaIpcCloseMemHandle = (PFN_cudaIpcCloseMemHandle)dlsym(RTLD_NEXT, "cudaIpcCloseMemHandle");
    if (!real_cudaIpcGetEventHandle)
        real_cudaIpcGetEventHandle = (PFN_cudaIpcGetEventHandle)dlsym(RTLD_NEXT, "cudaIpcGetEventHandle");
    if (!real_cudaIpcOpenEventHandle)
        real_cudaIpcOpenEventHandle = (PFN_cudaIpcOpenEventHandle)dlsym(RTLD_NEXT, "cudaIpcOpenEventHandle");
}

__attribute__((destructor)) static void fini(void) {
    log_line("[shim] cuda_ipc_trace unloaded");
    if (logf && logf != stderr) fclose(logf);
}

// Hooks

cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t* handle, void* devPtr) {
    resolve_syms();
    log_line("cudaIpcGetMemHandle(devPtr=%p)", devPtr);
    log_bt_if_enabled();
    return real_cudaIpcGetMemHandle(handle, devPtr);
}

cudaError_t cudaIpcOpenMemHandle(void** devPtr, cudaIpcMemHandle_t handle, unsigned int flags) {
    resolve_syms();
    log_line("cudaIpcOpenMemHandle(flags=0x%x)", flags);
    log_bt_if_enabled();
    return real_cudaIpcOpenMemHandle(devPtr, handle, flags);
}

cudaError_t cudaIpcCloseMemHandle(void* devPtr) {
    resolve_syms();
    log_line("cudaIpcCloseMemHandle(devPtr=%p)", devPtr);
    return real_cudaIpcCloseMemHandle(devPtr);
}

// Optional event IPC hooks
cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t* handle, void* event) {
    resolve_syms();
    log_line("cudaIpcGetEventHandle(event=%p)", event);
    return real_cudaIpcGetEventHandle(handle, event);
}

cudaError_t cudaIpcOpenEventHandle(void** event, cudaIpcEventHandle_t handle) {
    resolve_syms();
    log_line("cudaIpcOpenEventHandle()");
    return real_cudaIpcOpenEventHandle(event, handle);
}

