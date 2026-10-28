#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <libavcodec/avcodec.h>
#include <libavutil/error.h>
#include <libavutil/imgutils.h>
#include <libavutil/buffer.h>
#include <libavutil/pixfmt.h>
#include <libswscale/swscale.h>

typedef struct RosemaryOpaqueToken {
    uint64_t id;
    AVBufferRef *ref;
    struct RosemaryOpaqueToken *next;
} RosemaryOpaqueToken;

// Hard limits apply before get_buffer2 asks FFmpeg to allocate decoded
// reference frames. The separate Rust output queues do not cover this memory.
#define ROSEMARY_MAX_NATIVE_PIXELS (4096LL * 4096LL)
#define ROSEMARY_MAX_NATIVE_FRAME_BYTES (64U * 1024U * 1024U)
#define ROSEMARY_MAX_NATIVE_LIVE_BYTES (256U * 1024U * 1024U)
// Keep native result/status codes synchronized with the Rust FFI wrapper.
#define ROSEMARY_H264_PICTURE_REJECTED 6
#define ROSEMARY_TEST_SWS_CONTEXT_FAILURE 1
#define ROSEMARY_TEST_CONVERTED_FRAME_FAILURE 2

typedef struct RosemaryDecoder {
    size_t native_live_bytes;
    size_t native_peak_bytes;
    int source_budget_refused;
    int source_budget_identity_valid;
    uint64_t source_budget_id;
    int last_send_error;
    int test_send_enomem_after_accept;
    int test_fail_converter_allocation;
    RosemaryOpaqueToken *tokens;
    AVCodecContext *context;
    AVPacket *packet;
    AVFrame *frame;
    AVFrame *converted;
    struct SwsContext *converter;
} RosemaryDecoder;

typedef struct {
    RosemaryDecoder *decoder;
    AVBufferRef *original;
    size_t bytes;
} RosemaryTrackedBuffer;

static void rosemary_release_native_buffer(void *opaque, uint8_t *data) {
    RosemaryTrackedBuffer *tracked = opaque;
    (void)data;
    tracked->decoder->native_live_bytes -= tracked->bytes;
    av_buffer_unref(&tracked->original);
    free(tracked);
}

static int rosemary_refuse_source_budget(RosemaryDecoder *decoder, const AVFrame *frame) {
    decoder->source_budget_refused = 1;
    if (frame->opaque_ref && frame->opaque_ref->size == sizeof(uint64_t)) {
        memcpy(&decoder->source_budget_id, frame->opaque_ref->data, sizeof(uint64_t));
        decoder->source_budget_identity_valid = 1;
    }
    return AVERROR(ENOMEM);
}

static int rosemary_guarded_get_buffer2(AVCodecContext *context, AVFrame *frame, int flags) {
    RosemaryDecoder *decoder = context->opaque;
    if (!decoder) return AVERROR(EINVAL);
    if (frame->width <= 0 || frame->height <= 0) return AVERROR(EINVAL);
    if ((int64_t)frame->width * frame->height > ROSEMARY_MAX_NATIVE_PIXELS)
        return rosemary_refuse_source_budget(decoder, frame);
    // H.264's source format may be 10-bit or 4:4:4, not the I420 format we
    // eventually expose. Reserve padded source storage before FFmpeg allocates.
    int w = frame->width, h = frame->height;
    int alignments[AV_NUM_DATA_POINTERS] = {0};
    avcodec_align_dimensions2(context, &w, &h, alignments);
    if (w <= 0 || h <= 0) return AVERROR(EINVAL);
    if (w > 16384 || h > 16384)
        return rosemary_refuse_source_budget(decoder, frame);
    int predicted = av_image_get_buffer_size(frame->format, w + 128, h + 128, 64);
    if (predicted <= 0) return AVERROR(EINVAL);
    if ((size_t)predicted > ROSEMARY_MAX_NATIVE_FRAME_BYTES ||
        (size_t)predicted > ROSEMARY_MAX_NATIVE_LIVE_BYTES - decoder->native_live_bytes) {
        return rosemary_refuse_source_budget(decoder, frame);
    }
    int result = avcodec_default_get_buffer2(context, frame, flags);
    if (result < 0) return result;
    // A default FFmpeg H.264 software frame uses at most four plane buffers.
    // Wrap their AVBufferRefs, retaining the original allocation and charging
    // it until the very last decoder reference is released.
    size_t total = 0;
    for (int i = 0; i < AV_NUM_DATA_POINTERS && frame->buf[i]; ++i) {
        AVBufferRef *original = frame->buf[i];
        size_t bytes = original->size;
        if (bytes > ROSEMARY_MAX_NATIVE_FRAME_BYTES - total ||
            bytes > ROSEMARY_MAX_NATIVE_LIVE_BYTES - decoder->native_live_bytes) {
            int error = rosemary_refuse_source_budget(decoder, frame);
            av_frame_unref(frame);
            return error;
        }
        RosemaryTrackedBuffer *tracked = calloc(1, sizeof(*tracked));
        if (!tracked) { av_frame_unref(frame); return AVERROR(ENOMEM); }
        tracked->decoder = decoder;
        tracked->original = original;
        tracked->bytes = bytes;
        AVBufferRef *wrapped = av_buffer_create(original->data, original->size,
            rosemary_release_native_buffer, tracked, 0);
        if (!wrapped) {
            free(tracked);
            av_frame_unref(frame);
            return AVERROR(ENOMEM);
        }
        frame->buf[i] = wrapped;
        decoder->native_live_bytes += bytes;
        if (decoder->native_live_bytes > decoder->native_peak_bytes)
            decoder->native_peak_bytes = decoder->native_live_bytes;
        total += bytes;
    }
    return 0;
}

int rosemary_h264_last_send_error(RosemaryDecoder *decoder) {
    return decoder ? decoder->last_send_error : AVERROR(EINVAL);
}

// Enabled only by a test-only Rust FFI declaration; the fault occurs after
// libavcodec has actually consumed the next packet.
void rosemary_h264_test_send_enomem_after_accept(RosemaryDecoder *decoder) {
    if (decoder) decoder->test_send_enomem_after_accept = 1;
}

// Test-only one-shot failure after frame opaque identity has been populated.
// 1: sws converter creation; 2: converted AVFrame allocation.
void rosemary_h264_test_fail_converter_allocation(RosemaryDecoder *decoder, int stage) {
    if (decoder) decoder->test_fail_converter_allocation = stage;
}

int rosemary_h264_last_budget_id(RosemaryDecoder *decoder, uint64_t *id) {
    if (!decoder || !id || !decoder->source_budget_identity_valid) return 0;
    *id = decoder->source_budget_id;
    return 1;
}

size_t rosemary_h264_native_peak_bytes(RosemaryDecoder *decoder) {
    return decoder ? decoder->native_peak_bytes : 0;
}
size_t rosemary_h264_native_live_bytes(RosemaryDecoder *decoder) {
    return decoder ? decoder->native_live_bytes : 0;
}


typedef struct {
    int32_t width;
    int32_t height;
    int32_t format;
    int64_t pts;
    uint64_t id;
    int32_t identity_valid;
    int32_t color_range;
    int32_t colorspace;
    int32_t primaries;
    int32_t transfer;
    int32_t chroma_location;
    int32_t aspect_num;
    int32_t aspect_den;
    int32_t flags;
    int32_t decode_error_flags;
    int32_t repeat_pict;
    size_t crop_left, crop_top, crop_right, crop_bottom;
    const uint8_t *planes[3];
    intptr_t strides[3];
} RosemaryFrameView;

RosemaryDecoder *rosemary_h264_new(void) {
    const AVCodec *codec = avcodec_find_decoder(AV_CODEC_ID_H264);
    if (!codec) return NULL;
    RosemaryDecoder *decoder = calloc(1, sizeof(*decoder));
    if (!decoder) return NULL;
    decoder->context = avcodec_alloc_context3(codec);
    if (decoder->context) {
        decoder->context->flags |= AV_CODEC_FLAG_COPY_OPAQUE;
        decoder->context->max_pixels = ROSEMARY_MAX_NATIVE_PIXELS;
        // Avoid multiplying live decoder references by frame-thread count.
        decoder->context->thread_count = 1;
        decoder->context->thread_type = 0;
        decoder->context->opaque = decoder;
        decoder->context->get_buffer2 = rosemary_guarded_get_buffer2;
    }
    decoder->packet = av_packet_alloc();
    decoder->frame = av_frame_alloc();
    decoder->converted = av_frame_alloc();
    if (!decoder->context || !decoder->packet || !decoder->frame || !decoder->converted ||
        avcodec_open2(decoder->context, codec, NULL) < 0) {
        avcodec_free_context(&decoder->context);
        av_packet_free(&decoder->packet);
        av_frame_free(&decoder->frame);
        av_frame_free(&decoder->converted);
        free(decoder);
        return NULL;
    }
    return decoder;
}

void rosemary_h264_free(RosemaryDecoder *decoder) {
    if (!decoder) return;
    avcodec_free_context(&decoder->context);
    av_packet_free(&decoder->packet);
    av_frame_free(&decoder->frame);
    av_frame_free(&decoder->converted);
    sws_freeContext(decoder->converter);
    while (decoder->tokens) {
        RosemaryOpaqueToken *token = decoder->tokens;
        decoder->tokens = token->next;
        av_buffer_unref(&token->ref);
        free(token);
    }
    free(decoder);
}

int rosemary_h264_send(RosemaryDecoder *decoder, const uint8_t *data,
                      size_t length, int64_t pts, int64_t dts, uint64_t id) {
    if (!decoder) return AVERROR(EINVAL);
    if (!data) {
        int result = avcodec_send_packet(decoder->context, NULL);
        return result == AVERROR(EAGAIN) ? 1 : result;
    }
    if (!length || length > INT32_MAX) return AVERROR(EINVAL);
    av_packet_unref(decoder->packet);
    int result = av_new_packet(decoder->packet, (int)length);
    if (result < 0) return result;
    memcpy(decoder->packet->data, data, length);
    decoder->packet->pts = pts;
    decoder->packet->dts = dts;
    decoder->packet->opaque_ref = av_buffer_alloc(sizeof(id));
    if (!decoder->packet->opaque_ref) {
        av_packet_unref(decoder->packet);
        return AVERROR(ENOMEM);
    }
    memcpy(decoder->packet->opaque_ref->data, &id, sizeof(id));
    RosemaryOpaqueToken *token = calloc(1, sizeof(*token));
    if (!token) {
        av_packet_unref(decoder->packet);
        return AVERROR(ENOMEM);
    }
    token->ref = av_buffer_ref(decoder->packet->opaque_ref);
    if (!token->ref) {
        free(token);
        av_packet_unref(decoder->packet);
        return AVERROR(ENOMEM);
    }
    token->id = id;
    decoder->source_budget_refused = 0;
    decoder->source_budget_identity_valid = 0;
    result = avcodec_send_packet(decoder->context, decoder->packet);
    if (decoder->test_send_enomem_after_accept && result == 0) {
        decoder->test_send_enomem_after_accept = 0;
        result = AVERROR(ENOMEM);
    }
    decoder->last_send_error = result;
    av_packet_unref(decoder->packet);
    // EAGAIN is guaranteed not to consume input. Other errors can occur
    // after libavcodec takes the opaque reference into buffer_pkt; keep the
    // observer so the sample can be completed exactly once.
    if (result != AVERROR(EAGAIN)) {
        token->next = decoder->tokens;
        decoder->tokens = token;
    } else {
        av_buffer_unref(&token->ref);
        free(token);
    }
    if (result == AVERROR(EAGAIN)) return 1;
    // H.264 maps a failed get_buffer2(ENOMEM) to INVALIDDATA in decode.c.
    // This is a known guarded allocation refusal only when the callback
    // itself recorded it; unrelated errors remain fatal.
    if (decoder->source_budget_refused &&
        (result == AVERROR(ENOMEM) || result == AVERROR_INVALIDDATA))
        return 4; // Consumed packet, decoder source-frame budget exceeded.
    if (result == AVERROR_INVALIDDATA)
        return 3; // Recoverable corruption, ownership remains tracked.
    if (result < 0)
        return 5; // Potentially consumed, but decoder state is fatal.
    return result;
}

// After the last decoder-held reference disappears this sample cannot
// produce a future frame, regardless of the frame's presentation timestamp.
int rosemary_h264_token_released(RosemaryDecoder *decoder, uint64_t id) {
    if (!decoder) return -1;
    for (RosemaryOpaqueToken *t = decoder->tokens; t; t = t->next) {
        if (t->id == id) return av_buffer_get_ref_count(t->ref) == 1;
    }
    return -1;
}

void rosemary_h264_forget_token(RosemaryDecoder *decoder, uint64_t id) {
    if (!decoder) return;
    RosemaryOpaqueToken **link = &decoder->tokens;
    while (*link) {
        RosemaryOpaqueToken *token = *link;
        if (token->id == id) {
            *link = token->next;
            av_buffer_unref(&token->ref);
            free(token);
            return;
        }
        link = &token->next;
    }
}

// 0=frame, 1=need input, 2=EOF, 4=budget reject, 6=known picture reject;
// negative errors (including ENOMEM) leave the decoder session fatal.
int rosemary_h264_receive(RosemaryDecoder *decoder, RosemaryFrameView *out) {
    if (!decoder || !out) return AVERROR(EINVAL);
    av_frame_unref(decoder->frame);
    decoder->source_budget_refused = 0;
    decoder->source_budget_identity_valid = 0;
    int result = avcodec_receive_frame(decoder->context, decoder->frame);
    if (decoder->source_budget_refused &&
        (result == AVERROR(ENOMEM) || result == AVERROR_INVALIDDATA)) {
        memset(out, 0, sizeof(*out));
        if (decoder->source_budget_identity_valid) {
            out->identity_valid = 1;
            out->id = decoder->source_budget_id;
        }
        return 4;
    }
    if (result == AVERROR(EAGAIN)) return 1;
    if (result == AVERROR_EOF) return 2;
    if (result < 0) return result;
    AVFrame *f = decoder->frame;
    // A conversion/layout error concerns this identified picture rather than
    // the decode session. Populate identity before any fallible conversion.
    memset(out, 0, sizeof(*out));
    if (f->opaque_ref && f->opaque_ref->size == sizeof(out->id)) {
        memcpy(&out->id, f->opaque_ref->data, sizeof(out->id));
        out->identity_valid = 1;
    }
    if ((f->flags & AV_FRAME_FLAG_CORRUPT) || f->decode_error_flags) {
        out->flags = f->flags;
        out->decode_error_flags = f->decode_error_flags;
        return ROSEMARY_H264_PICTURE_REJECTED;
    }
    if (f->width <= 0 || f->height <= 0 || f->width > 32768 || f->height > 32768)
        return ROSEMARY_H264_PICTURE_REJECTED;
    AVFrame *source = f;
    int output_colorspace = source->colorspace;
    int color_range = f->color_range;
    if (color_range == AVCOL_RANGE_UNSPECIFIED && f->format == AV_PIX_FMT_YUVJ420P)
        color_range = AVCOL_RANGE_JPEG;
    // For uncommon H.264 chroma formats / bit depths, FFmpeg performs an
    // explicit, real colorspace conversion to our portable 8-bit output.
    // Ordinary 8-bit 4:2:0 frames keep their native memory layout.
    if (f->format != AV_PIX_FMT_YUV420P && f->format != AV_PIX_FMT_YUVJ420P &&
        f->format != AV_PIX_FMT_NV12) {
        // A whole-picture sws_scale mixes chroma across different fields.
        if (source->flags & AV_FRAME_FLAG_INTERLACED)
            return ROSEMARY_H264_PICTURE_REJECTED;
        int sws_matrix;
        switch (source->colorspace) {
        case AVCOL_SPC_RGB:
        case AVCOL_SPC_UNSPECIFIED:
            // swscale emits BT.601 YCbCr even for planar RGB input. The
            // output matrix describes converted pixels, not source pixels.
            sws_matrix = SWS_CS_ITU601;
            output_colorspace = AVCOL_SPC_SMPTE170M;
            break;
        case AVCOL_SPC_BT709: sws_matrix = SWS_CS_ITU709; break;
        case AVCOL_SPC_FCC: sws_matrix = SWS_CS_FCC; break;
        case AVCOL_SPC_BT470BG:
        case AVCOL_SPC_SMPTE170M: sws_matrix = SWS_CS_ITU601; break;
        case AVCOL_SPC_SMPTE240M: sws_matrix = SWS_CS_SMPTE240M; break;
        case AVCOL_SPC_BT2020_NCL: sws_matrix = SWS_CS_BT2020; break;
        // In particular, swscale cannot implement BT.2020 constant-luminance.
        default: return ROSEMARY_H264_PICTURE_REJECTED;
        }
        if (decoder->test_fail_converter_allocation == ROSEMARY_TEST_SWS_CONTEXT_FAILURE) {
            decoder->test_fail_converter_allocation = 0;
            return AVERROR(ENOMEM);
        }
        decoder->converter = sws_getCachedContext(
            decoder->converter, f->width, f->height, f->format,
            f->width, f->height, AV_PIX_FMT_YUV420P,
            SWS_BILINEAR, NULL, NULL, NULL);
        if (!decoder->converter) return AVERROR(ENOMEM);
        int *inverse, *table, src_range, dst_range, brightness, contrast, saturation;
        if (sws_getColorspaceDetails(decoder->converter, &inverse, &src_range,
                &table, &dst_range, &brightness, &contrast, &saturation) < 0)
            return AVERROR_INVALIDDATA;
        // The output range is explicitly the source range. If unknown, use
        // swscale's documented limited-range default but do not claim it was signaled.
        int full = color_range == AVCOL_RANGE_JPEG;
        const int *coefficients = sws_getCoefficients(sws_matrix);
        if (sws_setColorspaceDetails(decoder->converter, coefficients, full,
                coefficients, full, brightness, contrast, saturation) < 0)
            return AVERROR_INVALIDDATA;
        av_frame_unref(decoder->converted);
        decoder->converted->width = f->width;
        decoder->converted->height = f->height;
        decoder->converted->format = AV_PIX_FMT_YUV420P;
        if (decoder->test_fail_converter_allocation == ROSEMARY_TEST_CONVERTED_FRAME_FAILURE) {
            decoder->test_fail_converter_allocation = 0;
            return AVERROR(ENOMEM);
        }
        result = av_frame_get_buffer(decoder->converted, 32);
        if (result < 0) return result;
        result = sws_scale(decoder->converter,
            (const uint8_t * const *)f->data, f->linesize,
            0, f->height, decoder->converted->data, decoder->converted->linesize);
        if (result != f->height) return AVERROR_INVALIDDATA;
        if (color_range == AVCOL_RANGE_UNSPECIFIED) color_range = AVCOL_RANGE_MPEG;
        f = decoder->converted;
    }
    int fmt = f->format == AV_PIX_FMT_NV12 ? 1 : 0;
    out->width = f->width;
    out->height = f->height;
    out->format = fmt;
    out->pts = source->best_effort_timestamp;
    out->color_range = color_range;
    out->colorspace = output_colorspace;
    out->primaries = source->color_primaries;
    out->transfer = source->color_trc;
    out->chroma_location = f == source ? source->chroma_location : AVCHROMA_LOC_UNSPECIFIED;
    out->aspect_num = source->sample_aspect_ratio.num;
    out->aspect_den = source->sample_aspect_ratio.den;
    out->flags = source->flags;
    out->decode_error_flags = source->decode_error_flags;
    out->repeat_pict = source->repeat_pict;
    out->crop_left = source->crop_left;
    out->crop_top = source->crop_top;
    out->crop_right = source->crop_right;
    out->crop_bottom = source->crop_bottom;
    for (unsigned i = 0; i < 3; ++i) {
        out->planes[i] = f->data[i];
        out->strides[i] = f->linesize[i];
    }
    return 0;
}
