#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sherpa-onnx/c-api/c-api.h"

typedef struct ProgressState {
  double start_s;
  int seen_first;
  int chunks;
  int total_samples;
} ProgressState;

static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static char *join_path(const char *a, const char *b) {
  size_t n = strlen(a) + strlen(b) + 2;
  char *out = (char *)malloc(n);
  if (!out) {
    return NULL;
  }
  snprintf(out, n, "%s/%s", a, b);
  return out;
}

static char *read_file(const char *filename) {
  FILE *f = fopen(filename, "rb");
  if (!f) {
    fprintf(stderr, "Failed to open %s: %s\n", filename, strerror(errno));
    return NULL;
  }
  if (fseek(f, 0, SEEK_END) != 0) {
    fclose(f);
    return NULL;
  }
  long n = ftell(f);
  if (n < 0) {
    fclose(f);
    return NULL;
  }
  rewind(f);
  char *s = (char *)malloc((size_t)n + 1);
  if (!s) {
    fclose(f);
    return NULL;
  }
  size_t got = fread(s, 1, (size_t)n, f);
  fclose(f);
  s[got] = 0;
  while (got > 0 && (s[got - 1] == '\n' || s[got - 1] == '\r')) {
    s[--got] = 0;
  }
  return s;
}

static int progress_callback(const float *samples, int32_t n, float p, void *arg) {
  (void)samples;
  ProgressState *state = (ProgressState *)arg;
  state->chunks += 1;
  state->total_samples += n;
  if (!state->seen_first) {
    state->seen_first = 1;
    fprintf(stderr, "first_chunk_s=%.3f first_chunk_samples=%d progress=%.3f\n",
            now_s() - state->start_s, n, p);
    fflush(stderr);
  }
  return 1;
}

static void usage(const char *prog) {
  fprintf(stderr,
          "Usage: %s OUTPUT_WAV TEXT_FILE [repeat]\n"
          "\n"
          "Env:\n"
          "  XIAODIE_TTS_BASE      default: /home/vicky/xiaodie/tts\n"
          "  XIAODIE_TTS_THREADS   default: 6\n"
          "  XIAODIE_TTS_SENTENCES default: 4\n",
          prog);
}

int main(int argc, char **argv) {
  if (argc < 3) {
    usage(argv[0]);
    return 2;
  }

  const char *base = getenv("XIAODIE_TTS_BASE");
  if (!base || !base[0]) {
    base = "/home/vicky/xiaodie/tts";
  }
  int threads = 6;
  int max_sentences = 4;
  const char *threads_env = getenv("XIAODIE_TTS_THREADS");
  const char *sent_env = getenv("XIAODIE_TTS_SENTENCES");
  if (threads_env && threads_env[0]) {
    threads = atoi(threads_env);
  }
  if (sent_env && sent_env[0]) {
    max_sentences = atoi(sent_env);
  }

  int repeat = 1;
  if (argc >= 4) {
    repeat = atoi(argv[3]);
    if (repeat < 1) {
      repeat = 1;
    }
  }

  char *model = join_path(base, "vits-piper-zh_CN-chaowen-medium/zh_CN-chaowen-medium.onnx");
  char *tokens = join_path(base, "vits-piper-zh_CN-chaowen-medium/tokens.txt");
  char *lexicon = join_path(base, "vits-piper-zh_CN-chaowen-medium/lexicon.txt");
  size_t rules_n = strlen(base) * 3 + 256;
  char *rules = (char *)malloc(rules_n);
  if (rules) {
    snprintf(rules, rules_n, "%s/%s,%s/%s,%s/%s", base,
             "vits-piper-zh_CN-chaowen-medium/number.fst", base,
             "vits-piper-zh_CN-chaowen-medium/date.fst", base,
             "vits-piper-zh_CN-chaowen-medium/phone.fst");
  }
  char *text = read_file(argv[2]);
  if (!model || !tokens || !lexicon || !rules || !text) {
    fprintf(stderr, "Allocation or file read failure\n");
    return 1;
  }

  SherpaOnnxOfflineTtsConfig config;
  memset(&config, 0, sizeof(config));
  config.model.vits.model = model;
  config.model.vits.tokens = tokens;
  config.model.vits.lexicon = lexicon;
  config.model.vits.noise_scale = 0.667f;
  config.model.vits.noise_scale_w = 0.8f;
  config.model.vits.length_scale = 1.0f;
  config.model.num_threads = threads;
  config.model.provider = "cpu";
  config.model.debug = 0;
  config.rule_fsts = rules;
  config.max_num_sentences = max_sentences;
  config.silence_scale = 0.2f;

  fprintf(stderr, "config threads=%d max_sentences=%d text_bytes=%zu\n", threads,
          max_sentences, strlen(text));

  double t0 = now_s();
  const SherpaOnnxOfflineTts *tts = SherpaOnnxCreateOfflineTts(&config);
  double t1 = now_s();
  if (!tts) {
    fprintf(stderr, "Failed to create offline TTS\n");
    return 1;
  }
  fprintf(stderr, "load_s=%.3f sample_rate=%d speakers=%d\n", t1 - t0,
          SherpaOnnxOfflineTtsSampleRate(tts),
          SherpaOnnxOfflineTtsNumSpeakers(tts));

  for (int i = 0; i < repeat; ++i) {
    char out[1024];
    if (repeat == 1) {
      snprintf(out, sizeof(out), "%s", argv[1]);
    } else {
      snprintf(out, sizeof(out), "%s.%d.wav", argv[1], i + 1);
    }

    SherpaOnnxGenerationConfig gen;
    memset(&gen, 0, sizeof(gen));
    gen.silence_scale = 0.2f;
    gen.sid = 0;
    gen.speed = 1.0f;

    ProgressState progress;
    memset(&progress, 0, sizeof(progress));
    progress.start_s = now_s();

    double g0 = now_s();
    const SherpaOnnxGeneratedAudio *audio =
        SherpaOnnxOfflineTtsGenerateWithConfig(tts, text, &gen, progress_callback, &progress);
    double g1 = now_s();
    if (!audio) {
      fprintf(stderr, "Generation failed at repeat %d\n", i + 1);
      SherpaOnnxDestroyOfflineTts(tts);
      return 1;
    }

    double w0 = now_s();
    int ok = SherpaOnnxWriteWave(audio->samples, audio->n, audio->sample_rate, out);
    double w1 = now_s();
    double audio_s = (double)audio->n / (double)audio->sample_rate;
    fprintf(stderr,
            "repeat=%d synth_s=%.3f write_s=%.3f audio_s=%.3f rtf=%.3f chunks=%d "
            "callback_samples=%d output=%s ok=%d\n",
            i + 1, g1 - g0, w1 - w0, audio_s, (g1 - g0) / audio_s,
            progress.chunks, progress.total_samples, out, ok);
    SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);
  }

  SherpaOnnxDestroyOfflineTts(tts);
  free(model);
  free(tokens);
  free(lexicon);
  free(rules);
  free(text);
  return 0;
}
