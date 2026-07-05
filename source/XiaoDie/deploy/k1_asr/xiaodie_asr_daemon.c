#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "sherpa-onnx/c-api/c-api.h"

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

static double now_s(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

static void trim(char *s) {
  size_t n = strlen(s);
  while (n > 0 && (s[n - 1] == '\n' || s[n - 1] == '\r' || isspace((unsigned char)s[n - 1]))) {
    s[--n] = '\0';
  }
  char *p = s;
  while (*p && isspace((unsigned char)*p)) {
    ++p;
  }
  if (p != s) {
    memmove(s, p, strlen(p) + 1);
  }
}

static void join_path(char *dst, size_t dst_size, const char *dir, const char *name) {
  size_t n = strlen(dir);
  if (n > 0 && dir[n - 1] == '/') {
    snprintf(dst, dst_size, "%s%s", dir, name);
  } else {
    snprintf(dst, dst_size, "%s/%s", dir, name);
  }
}

static void json_string(FILE *out, const char *s) {
  fputc('"', out);
  for (; s && *s; ++s) {
    unsigned char c = (unsigned char)*s;
    if (c == '"' || c == '\\') {
      fputc('\\', out);
      fputc(c, out);
    } else if (c == '\n') {
      fputs("\\n", out);
    } else if (c == '\r') {
      fputs("\\r", out);
    } else if (c == '\t') {
      fputs("\\t", out);
    } else if (c < 0x20) {
      fprintf(out, "\\u%04x", c);
    } else {
      fputc(c, out);
    }
  }
  fputc('"', out);
}

static int recognize_file(const SherpaOnnxOnlineRecognizer *recognizer, const char *path, int tail_pad_ms) {
  double started = now_s();
  const SherpaOnnxWave *wave = SherpaOnnxReadWave(path);
  if (!wave) {
    printf("{\"ok\":false,\"error\":\"failed_to_read_wav\",\"path\":");
    json_string(stdout, path);
    printf("}\n");
    fflush(stdout);
    return 1;
  }

  const SherpaOnnxOnlineStream *stream = SherpaOnnxCreateOnlineStream(recognizer);
  if (!stream) {
    SherpaOnnxFreeWave(wave);
    printf("{\"ok\":false,\"error\":\"failed_to_create_stream\",\"path\":");
    json_string(stdout, path);
    printf("}\n");
    fflush(stdout);
    return 1;
  }

  const int32_t chunk_size = 3200;
  int32_t offset = 0;
  while (offset < wave->num_samples) {
    int32_t n = chunk_size;
    if (offset + n > wave->num_samples) {
      n = wave->num_samples - offset;
    }
    SherpaOnnxOnlineStreamAcceptWaveform(stream, wave->sample_rate, wave->samples + offset, n);
    offset += n;
    while (SherpaOnnxIsOnlineStreamReady(recognizer, stream)) {
      SherpaOnnxDecodeOnlineStream(recognizer, stream);
    }
  }

  if (tail_pad_ms > 0) {
    int32_t tail_samples =
        wave->sample_rate > 0 ? (int32_t)((int64_t)wave->sample_rate * tail_pad_ms / 1000) : 8000;
    float *tail = (float *)calloc((size_t)tail_samples, sizeof(float));
    if (tail) {
      SherpaOnnxOnlineStreamAcceptWaveform(stream, wave->sample_rate, tail, tail_samples);
      free(tail);
      while (SherpaOnnxIsOnlineStreamReady(recognizer, stream)) {
        SherpaOnnxDecodeOnlineStream(recognizer, stream);
      }
    }
  }

  SherpaOnnxOnlineStreamInputFinished(stream);
  while (SherpaOnnxIsOnlineStreamReady(recognizer, stream)) {
    SherpaOnnxDecodeOnlineStream(recognizer, stream);
  }

  const SherpaOnnxOnlineRecognizerResult *result =
      SherpaOnnxGetOnlineStreamResult(recognizer, stream);
  double elapsed = now_s() - started;
  double audio_s = wave->sample_rate > 0 ? (double)wave->num_samples / (double)wave->sample_rate : 0.0;
  double rtf = audio_s > 0 ? elapsed / audio_s : 0.0;

  printf("{\"ok\":true,\"text\":");
  json_string(stdout, result && result->text ? result->text : "");
  printf(",\"audio_s\":%.3f,\"elapsed_s\":%.3f,\"rtf\":%.3f,\"path\":", audio_s, elapsed, rtf);
  json_string(stdout, path);
  printf("}\n");
  fflush(stdout);

  SherpaOnnxDestroyOnlineRecognizerResult(result);
  SherpaOnnxDestroyOnlineStream(stream);
  SherpaOnnxFreeWave(wave);
  return 0;
}

int main(int argc, char **argv) {
  const char *model_dir =
      "/home/vicky/xiaodie/asr/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05";
  int32_t threads = 4;
  int32_t debug = 0;
  int tail_pad_ms = 500;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--model-dir") == 0 && i + 1 < argc) {
      model_dir = argv[++i];
    } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
      threads = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--tail-pad-ms") == 0 && i + 1 < argc) {
      tail_pad_ms = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--debug") == 0) {
      debug = 1;
    } else if (strcmp(argv[i], "--help") == 0) {
      fprintf(stderr, "usage: %s [--model-dir DIR] [--threads N] [--tail-pad-ms N] [--debug]\n", argv[0]);
      return 0;
    } else {
      fprintf(stderr, "unknown argument: %s\n", argv[i]);
      return 2;
    }
  }

  char encoder[PATH_MAX], decoder[PATH_MAX], joiner[PATH_MAX], tokens[PATH_MAX];
  join_path(encoder, sizeof(encoder), model_dir, "encoder.int8.onnx");
  join_path(decoder, sizeof(decoder), model_dir, "decoder.onnx");
  join_path(joiner, sizeof(joiner), model_dir, "joiner.int8.onnx");
  join_path(tokens, sizeof(tokens), model_dir, "tokens.txt");

  SherpaOnnxOnlineRecognizerConfig config;
  memset(&config, 0, sizeof(config));
  config.feat_config.sample_rate = 16000;
  config.feat_config.feature_dim = 80;
  config.model_config.transducer.encoder = encoder;
  config.model_config.transducer.decoder = decoder;
  config.model_config.transducer.joiner = joiner;
  config.model_config.tokens = tokens;
  config.model_config.provider = "cpu";
  config.model_config.num_threads = threads;
  config.model_config.debug = debug;
  config.model_config.model_type = "zipformer2";
  config.decoding_method = "greedy_search";
  config.max_active_paths = 4;

  double load_started = now_s();
  const SherpaOnnxOnlineRecognizer *recognizer = SherpaOnnxCreateOnlineRecognizer(&config);
  double load_s = now_s() - load_started;
  if (!recognizer) {
    fprintf(stderr, "[asr_daemon] failed to create recognizer from %s\n", model_dir);
    return 1;
  }
  fprintf(stderr, "[asr_daemon] ready model_dir=%s threads=%d tail_pad_ms=%d load_s=%.3f\n",
          model_dir, threads, tail_pad_ms, load_s);
  fflush(stderr);

  char line[PATH_MAX];
  while (fgets(line, sizeof(line), stdin)) {
    trim(line);
    if (!line[0]) {
      continue;
    }
    if (strcmp(line, "::quit") == 0) {
      break;
    }
    if (strcmp(line, "::ping") == 0) {
      printf("{\"ok\":true,\"event\":\"pong\"}\n");
      fflush(stdout);
      continue;
    }
    recognize_file(recognizer, line, tail_pad_ms);
  }

  SherpaOnnxDestroyOnlineRecognizer(recognizer);
  fprintf(stderr, "[asr_daemon] stopped\n");
  return 0;
}
