/*
 * Compatibility shim for the host NVIDIA driver's libnvidia-glcore.so
 * (535.216.03), which references glibc malloc hooks that were removed in
 * glibc 2.34 (__malloc_hook / __realloc_hook / __free_hook /
 * __memalign_hook), plus the Xorg-internal ErrorF symbol it expects to be
 * provided by an X server it isn't running inside of. Without these,
 * dlopen-time symbol resolution for libnvidia-glcore.so fails with
 * "undefined symbol ... (fatal)", which silently breaks the NVIDIA Vulkan
 * ICD's ability to hand back vkCreateInstance whenever a caller (like Isaac
 * Sim/Kit) pulls in libnvidia-glcore as a dependency -- see CLAUDE.md at
 * /scratch/cluster/jshim12/CLAUDE.md for the full investigation.
 *
 * These are inert stand-ins: the hooks behave as "no hook installed" (their
 * original glibc default), and ErrorF is a no-op logger. Build:
 *   gcc -shared -fPIC -o libmalloc_hook_shim.so shim.c
 * Use via LD_PRELOAD=/path/to/libmalloc_hook_shim.so.
 */

#include <stddef.h>
#include <stdarg.h>

void *(*__malloc_hook)(size_t, const void *) = 0;
void *(*__realloc_hook)(void *, size_t, const void *) = 0;
void (*__free_hook)(void *, const void *) = 0;
void *(*__memalign_hook)(size_t, size_t, const void *) = 0;

void ErrorF(const char *fmt, ...) {
    (void)fmt;
}

/* Xorg mi-layer symbol; real signature is Bool miCreateDefColormap(ScreenPtr).
 * We don't have Xorg's headers here, so this is a best-effort ABI-compatible
 * stub (single pointer arg, int return) that reports success. */
int miCreateDefColormap(void *pScreen) {
    (void)pScreen;
    return 1;
}
