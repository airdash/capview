/*
 * capview plugin ABI — v1
 *
 * Plugins are shared libraries (.so) loaded at runtime. Each plugin must
 * export the symbols listed below with C linkage. capview calls them in
 * this order:
 *
 *   1. capview_filter_name()           — once, at load time
 *   2. capview_filter_init(...)        — once, after capture starts
 *   3. capview_filter_process(...)     — every frame
 *   4. capview_filter_destroy()        — at shutdown
 *
 * Frame data is raw pixel data in the capture pixel format (NV12, YUYV,
 * or UYVY). The plugin receives one input frame and writes one or more
 * output frames into the provided buffer. For example a frame-
 * interpolation plugin (like RIFE) would return 2 frames per input.
 *
 * Build example (gcc):
 *   gcc -shared -fPIC -O2 -o my_filter.so my_filter.c
 *
 * Config (capview.conf):
 *   plugins = /path/to/my_filter.so
 *   plugins = /path/to/another.so:arg1,arg2
 *
 * Multiple plugins are applied in order (pipeline).
 */

#ifndef CAPVIEW_PLUGIN_H
#define CAPVIEW_PLUGIN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Current ABI version — checked at load time. */
#define CAPVIEW_PLUGIN_ABI_VERSION 1

/*
 * Return the ABI version this plugin was built against.
 * Must return CAPVIEW_PLUGIN_ABI_VERSION.
 */
int capview_filter_abi_version(void);

/*
 * Return a short human-readable name for the filter (e.g. "RIFE 4.x").
 * The pointer must remain valid for the lifetime of the plugin.
 */
const char *capview_filter_name(void);

/*
 * Initialise the filter with capture parameters.
 *   width, height  — frame dimensions in pixels.
 *   fps            — capture frame rate.
 *   pixfmt         — V4L2 pixel format fourcc (e.g. 0x3231564E for NV12).
 *   args           — optional argument string from the config (may be NULL).
 *
 * Return 0 on success, non-zero on failure.
 */
int capview_filter_init(uint32_t width, uint32_t height,
                        uint32_t fps, uint32_t pixfmt,
                        const char *args);

/*
 * Process one input frame.
 *
 *   input          — pointer to raw frame data (read-only).
 *   input_len      — size of the input frame in bytes.
 *   output         — caller-provided buffer for output frame(s).
 *   output_cap     — capacity of the output buffer in bytes.
 *   output_len     — [out] total bytes written to output.
 *   width, height  — frame dimensions (same as passed to init).
 *
 * The plugin writes one or more complete frames sequentially into
 * `output`. Each output frame must be the same size as the input frame.
 *
 * Returns:
 *   > 0  — number of output frames written.
 *     0  — frame skipped (caller should not render anything).
 *   < 0  — error.
 */
int capview_filter_process(const uint8_t *input, uint32_t input_len,
                           uint8_t *output, uint32_t output_cap,
                           uint32_t *output_len,
                           uint32_t width, uint32_t height);

/*
 * Tear down the filter and free any resources.
 */
void capview_filter_destroy(void);

#ifdef __cplusplus
}
#endif

#endif /* CAPVIEW_PLUGIN_H */
