/* Native metadata: safe to inspect before initializing any Julia image. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <julia.h>
#include <stdint.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#ifdef _WIN32
#define JUMPY_EXPORT __declspec(dllexport)
#else
#define JUMPY_EXPORT __attribute__((visibility("default")))
#endif

#ifndef JUMPY_TRIMMED
#error "Build through build.jl so the image profile is recorded correctly"
#endif

JUMPY_EXPORT const uint32_t jumpy_abi_version = 1;
JUMPY_EXPORT const uint32_t jumpy_image_trimmed = JUMPY_TRIMMED;

/* Does not use Julia evaluation, which is intentionally unavailable after
 * trimming. The options string is owned by the runtime, not by Julia's GC. */
JUMPY_EXPORT const char *jumpy_runtime_image_path(void)
{
    return jl_is_initialized() ? jl_options.image_file : NULL;
}

/* Tell JuliaCall which already-loaded runtime to attach to, including when
 * the library is bundled next to a different Julia installation. */
JUMPY_EXPORT const char *jumpy_runtime_library_path(void)
{
#ifdef _WIN32
    static char path[32768];
    HMODULE module;
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           (LPCSTR)&jl_init_with_image_handle, &module))
        return NULL;
    DWORD length = GetModuleFileNameA(module, path, sizeof(path));
    return length && length < sizeof(path) ? path : NULL;
#else
    Dl_info info;
    return dladdr((void *)&jl_init_with_image_handle, &info) ? info.dli_fname : NULL;
#endif
}
