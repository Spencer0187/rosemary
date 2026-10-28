#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <libavcodec/avcodec.h>
#include <libavutil/error.h>
#include <libavutil/buffer.h>
#include <libavutil/pixfmt.h>
#include <libswscale/swscale.h>

typedef struct RosemaryOpaqueToken {
    uint64_t id;
    AVBufferRef *ref;
    struct RosemaryOpaqueToken *next;
} RosemaryOpaqueToken;

typedef struct {
    RosemaryOpaqueToken *tokens;
    AVCodecContext *context;
    AVPacket *packet;
    AVFrame *frame;
    AVFrame *converted;
    struct SwsContext *converter;
} RosemaryDecoder;

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
    if (decoder->context) decoder->context->flags |= AV_CODEC_FLAG_COPY_OPAQUE;
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
    result = avcodec_send_packet(decoder->context, decoder->packet);
    av_packet_unref(decoder->packet);
    if (result == 0) {
        token->next = decoder->tokens;
        decoder->tokens = token;
    } else {
        av_buffer_unref(&token->ref);
        free(token);
    }
    return result == AVERROR(EAGAIN) ? 1 : result;
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

// 0=frame available, 1=more input needed, 2=drained, negative=decode error.
int rosemary_h264_receive(RosemaryDecoder *decoder, RosemaryFrameView *out) {
    if (!decoder || !out) return AVERROR(EINVAL);
    av_frame_unref(decoder->frame);
    int result = avcodec_receive_frame(decoder->context, decoder->frame);
    if (result == AVERROR(EAGAIN)) return 1;
    if (result == AVERROR_EOF) return 2;
    if (result < 0) return result;
    AVFrame *f = decoder->frame;
    if (f->width <= 0 || f->height <= 0 || f->width > 32768 || f->height > 32768)
        return AVERROR_INVALIDDATA;
    AVFrame *source = f;
    int color_range = f->color_range;
    if (color_range == AVCOL_RANGE_UNSPECIFIED && f->format == AV_PIX_FMT_YUVJ420P)
        color_range = AVCOL_RANGE_JPEG;
    // For uncommon H.264 chroma formats / bit depths, FFmpeg performs an
    // explicit, real colorspace conversion to our portable 8-bit output.
    // Ordinary 8-bit 4:2:0 frames keep their native memory layout.
    if (f->format != AV_PIX_FMT_YUV420P && f->format != AV_PIX_FMT_YUVJ420P &&
        f->format != AV_PIX_FMT_NV12) {
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
        const int *coefficients = sws_getCoefficients(
            source->colorspace == AVCOL_SPC_BT709 ? SWS_CS_ITU709 :
            source->colorspace == AVCOL_SPC_BT2020_NCL ? SWS_CS_BT2020 : SWS_CS_ITU601);
        if (sws_setColorspaceDetails(decoder->converter, coefficients, full,
                coefficients, full, brightness, contrast, saturation) < 0)
            return AVERROR_INVALIDDATA;
        av_frame_unref(decoder->converted);
        decoder->converted->width = f->width;
        decoder->converted->height = f->height;
        decoder->converted->format = AV_PIX_FMT_YUV420P;
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
    memset(out, 0, sizeof(*out));
    out->width = f->width;
    out->height = f->height;
    out->format = fmt;
    out->pts = source->best_effort_timestamp;
    if (source->opaque_ref && source->opaque_ref->size == sizeof(out->id)) {
        memcpy(&out->id, source->opaque_ref->data, sizeof(out->id));
        out->identity_valid = 1;
    }
    out->color_range = color_range;
    out->colorspace = source->colorspace;
    out->primaries = source->color_primaries;
    out->transfer = source->color_trc;
    out->chroma_location = f == source ? source->chroma_location : AVCHROMA_LOC_UNSPECIFIED;
    out->aspect_num = source->sample_aspect_ratio.num;
    out->aspect_den = source->sample_aspect_ratio.den;
    out->flags = source->flags;
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
