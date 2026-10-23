#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <libavcodec/avcodec.h>
#include <libavutil/error.h>
#include <libavutil/pixfmt.h>
#include <libswscale/swscale.h>

typedef struct {
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
    const uint8_t *planes[3];
    intptr_t strides[3];
} RosemaryFrameView;

RosemaryDecoder *rosemary_h264_new(void) {
    const AVCodec *codec = avcodec_find_decoder(AV_CODEC_ID_H264);
    if (!codec) return NULL;
    RosemaryDecoder *decoder = calloc(1, sizeof(*decoder));
    if (!decoder) return NULL;
    decoder->context = avcodec_alloc_context3(codec);
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
    free(decoder);
}

int rosemary_h264_send(RosemaryDecoder *decoder, const uint8_t *data,
                      size_t length, int64_t pts, int64_t dts) {
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
    result = avcodec_send_packet(decoder->context, decoder->packet);
    av_packet_unref(decoder->packet);
    return result == AVERROR(EAGAIN) ? 1 : result;
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
        f = decoder->converted;
    }
    int fmt = f->format == AV_PIX_FMT_NV12 ? 1 : 0;
    memset(out, 0, sizeof(*out));
    out->width = f->width;
    out->height = f->height;
    out->format = fmt;
    out->pts = decoder->frame->best_effort_timestamp;
    for (unsigned i = 0; i < 3; ++i) {
        out->planes[i] = f->data[i];
        out->strides[i] = f->linesize[i];
    }
    return 0;
}
