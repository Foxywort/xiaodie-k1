#define _POSIX_C_SOURCE 200809L

#include <ctype.h>
#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sherpa-onnx/c-api/c-api.h"

typedef struct Job {
  int id;
  int is_marker;
  char *text;
  struct Job *next;
} Job;

typedef struct Queue {
  Job *head;
  Job *tail;
  int closed;
  int next_id;
  pthread_mutex_t mu;
  pthread_cond_t cv;
} Queue;

typedef struct AudioJob {
  int id;
  int is_marker;
  char *text;
  unsigned char *pcm;
  size_t bytes;
  int samples;
  double synth_s;
  double audio_s;
  struct AudioJob *next;
} AudioJob;

typedef struct AudioQueue {
  AudioJob *head;
  AudioJob *tail;
  int closed;
  pthread_mutex_t mu;
  pthread_cond_t cv;
} AudioQueue;

typedef struct TtsState {
  const SherpaOnnxOfflineTts *tts;
  int sample_rate;
  int output_rate;
  int output_channels;
  char device[128];
  int play;
  volatile sig_atomic_t cancel;
  Queue queue;
  AudioQueue audio_queue;
} TtsState;

static int g_min_chars = 25;
static int g_max_chars = 90;

static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static char *xstrdup(const char *s) {
  size_t n = strlen(s) + 1;
  char *p = (char *)malloc(n);
  if (p) {
    memcpy(p, s, n);
  }
  return p;
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

static void queue_init(Queue *q) {
  memset(q, 0, sizeof(*q));
  q->next_id = 1;
  pthread_mutex_init(&q->mu, NULL);
  pthread_cond_init(&q->cv, NULL);
}

static void queue_clear_locked(Queue *q) {
  Job *j = q->head;
  while (j) {
    Job *next = j->next;
    free(j->text);
    free(j);
    j = next;
  }
  q->head = NULL;
  q->tail = NULL;
}

static void queue_push(Queue *q, char *text) {
  if (!text || !text[0]) {
    free(text);
    return;
  }
  Job *j = (Job *)calloc(1, sizeof(Job));
  if (!j) {
    free(text);
    return;
  }
  pthread_mutex_lock(&q->mu);
  j->id = q->next_id++;
  j->text = text;
  if (q->tail) {
    q->tail->next = j;
  } else {
    q->head = j;
  }
  q->tail = j;
  pthread_cond_signal(&q->cv);
  pthread_mutex_unlock(&q->mu);
  fprintf(stderr, "queued id=%d bytes=%zu text=%s\n", j->id, strlen(j->text), j->text);
  fflush(stderr);
}

static void queue_push_marker(Queue *q, char *marker) {
  if (!marker || !marker[0]) {
    free(marker);
    return;
  }
  Job *j = (Job *)calloc(1, sizeof(Job));
  if (!j) {
    free(marker);
    return;
  }
  pthread_mutex_lock(&q->mu);
  j->id = q->next_id++;
  j->is_marker = 1;
  j->text = marker;
  if (q->tail) {
    q->tail->next = j;
  } else {
    q->head = j;
  }
  q->tail = j;
  pthread_cond_signal(&q->cv);
  pthread_mutex_unlock(&q->mu);
  fprintf(stderr, "mark_queued id=%d token=%s\n", j->id, j->text);
  fflush(stderr);
}

static Job *queue_pop(Queue *q) {
  pthread_mutex_lock(&q->mu);
  while (!q->head && !q->closed) {
    pthread_cond_wait(&q->cv, &q->mu);
  }
  Job *j = q->head;
  if (j) {
    q->head = j->next;
    if (!q->head) {
      q->tail = NULL;
    }
  }
  pthread_mutex_unlock(&q->mu);
  return j;
}

static void queue_close(Queue *q) {
  pthread_mutex_lock(&q->mu);
  q->closed = 1;
  pthread_cond_broadcast(&q->cv);
  pthread_mutex_unlock(&q->mu);
}

static void queue_reset(Queue *q) {
  pthread_mutex_lock(&q->mu);
  queue_clear_locked(q);
  pthread_cond_broadcast(&q->cv);
  pthread_mutex_unlock(&q->mu);
}

static void audio_queue_init(AudioQueue *q) {
  memset(q, 0, sizeof(*q));
  pthread_mutex_init(&q->mu, NULL);
  pthread_cond_init(&q->cv, NULL);
}

static void audio_job_free(AudioJob *j) {
  if (!j) return;
  free(j->text);
  free(j->pcm);
  free(j);
}

static void audio_queue_clear_locked(AudioQueue *q) {
  AudioJob *j = q->head;
  while (j) {
    AudioJob *next = j->next;
    audio_job_free(j);
    j = next;
  }
  q->head = NULL;
  q->tail = NULL;
}

static void audio_queue_push(AudioQueue *q, AudioJob *j) {
  if (!j) return;
  pthread_mutex_lock(&q->mu);
  if (q->tail) {
    q->tail->next = j;
  } else {
    q->head = j;
  }
  q->tail = j;
  pthread_cond_signal(&q->cv);
  pthread_mutex_unlock(&q->mu);
}

static AudioJob *audio_queue_pop(AudioQueue *q) {
  pthread_mutex_lock(&q->mu);
  while (!q->head && !q->closed) {
    pthread_cond_wait(&q->cv, &q->mu);
  }
  AudioJob *j = q->head;
  if (j) {
    q->head = j->next;
    if (!q->head) {
      q->tail = NULL;
    }
    j->next = NULL;
  }
  pthread_mutex_unlock(&q->mu);
  return j;
}

static void audio_queue_close(AudioQueue *q) {
  pthread_mutex_lock(&q->mu);
  q->closed = 1;
  pthread_cond_broadcast(&q->cv);
  pthread_mutex_unlock(&q->mu);
}

static void audio_queue_reset(AudioQueue *q) {
  pthread_mutex_lock(&q->mu);
  audio_queue_clear_locked(q);
  pthread_cond_broadcast(&q->cv);
  pthread_mutex_unlock(&q->mu);
}

static int starts_with(const char *s, const char *prefix) {
  return strncmp(s, prefix, strlen(prefix)) == 0;
}

static int utf8_char_len(unsigned char c) {
  if (c < 0x80) return 1;
  if ((c & 0xE0) == 0xC0) return 2;
  if ((c & 0xF0) == 0xE0) return 3;
  if ((c & 0xF8) == 0xF0) return 4;
  return 1;
}

static int utf8_count(const char *s, size_t n) {
  int count = 0;
  size_t i = 0;
  while (i < n) {
    i += utf8_char_len((unsigned char)s[i]);
    count += 1;
  }
  return count;
}

static int match_at(const char *s, size_t i, const char *needle) {
  size_t n = strlen(needle);
  return strncmp(s + i, needle, n) == 0;
}

static int is_boundary_at(const char *s, size_t i, size_t n, size_t *end) {
  const char *marks[] = {"。", "！", "？", "!", "?", "\n"};
  for (size_t k = 0; k < sizeof(marks) / sizeof(marks[0]); ++k) {
    size_t m = strlen(marks[k]);
    if (i + m <= n && strncmp(s + i, marks[k], m) == 0) {
      *end = i + m;
      return 1;
    }
  }
  if (i + 6 <= n && strncmp(s + i, "……", 6) == 0) {
    *end = i + 6;
    return 1;
  }
  return 0;
}

static int is_closer_at(const char *s, size_t i, size_t n, size_t *end) {
  const char *closers[] = {"”", "’", "）", ")", "】", "》", "」", "』", "\"", "'"};
  for (size_t k = 0; k < sizeof(closers) / sizeof(closers[0]); ++k) {
    size_t m = strlen(closers[k]);
    if (i + m <= n && strncmp(s + i, closers[k], m) == 0) {
      *end = i + m;
      return 1;
    }
  }
  return 0;
}

static size_t absorb_after_boundary(const char *s, size_t pos, size_t n) {
  while (pos < n) {
    if ((unsigned char)s[pos] < 0x80 && isspace((unsigned char)s[pos])) {
      pos += 1;
      continue;
    }
    size_t end = pos;
    if (is_closer_at(s, pos, n, &end)) {
      pos = end;
      continue;
    }
    break;
  }
  return pos;
}

static void trim_ascii(char *s) {
  size_t n = strlen(s);
  size_t start = 0;
  while (start < n && isspace((unsigned char)s[start])) start++;
  size_t end = n;
  while (end > start && isspace((unsigned char)s[end - 1])) end--;
  if (start > 0) {
    memmove(s, s + start, end - start);
  }
  s[end - start] = 0;
}

static char *take_prefix(char **buf, size_t end) {
  char *s = *buf;
  size_t n = strlen(s);
  if (end > n) end = n;
  char *part = (char *)malloc(end + 1);
  if (!part) return NULL;
  memcpy(part, s, end);
  part[end] = 0;
  trim_ascii(part);
  size_t rest_n = n - end;
  memmove(s, s + end, rest_n);
  s[rest_n] = 0;
  trim_ascii(s);
  return part;
}

static int extract_ready_segment(char **buf, int force, char **out) {
  char *s = *buf;
  size_t n = strlen(s);
  if (n == 0) return 0;

  size_t chosen = 0;
  size_t first_boundary = 0;
  size_t last_under_max = 0;
  size_t i = 0;
  while (i < n) {
    size_t bend = 0;
    if (is_boundary_at(s, i, n, &bend)) {
      bend = absorb_after_boundary(s, bend, n);
      if (!first_boundary) first_boundary = bend;
      int chars = utf8_count(s, bend);
      if (chars <= g_max_chars) {
        last_under_max = bend;
      }
      if (chars >= g_min_chars && chars <= g_max_chars) {
        chosen = bend;
        i = bend;
        continue;
      }
      if (chars > g_max_chars) {
        chosen = chosen ? chosen : (last_under_max ? last_under_max : bend);
        break;
      }
      i = bend;
      continue;
    }
    i += utf8_char_len((unsigned char)s[i]);
  }

  if (!chosen && force && n > 0) {
    chosen = last_under_max ? last_under_max : (first_boundary ? first_boundary : n);
  }
  if (!chosen) return 0;
  *out = take_prefix(buf, chosen);
  return *out && (*out)[0];
}

static void append_text(char **buf, const char *text) {
  size_t a = strlen(*buf);
  size_t b = strlen(text);
  char *next = (char *)realloc(*buf, a + b + 1);
  if (!next) return;
  memcpy(next + a, text, b + 1);
  *buf = next;
}

static void float_to_s16(float x, unsigned char out[2]) {
  if (x > 1.0f) x = 1.0f;
  if (x < -1.0f) x = -1.0f;
  int16_t v = (int16_t)(x * 32767.0f);
  out[0] = (unsigned char)(v & 0xff);
  out[1] = (unsigned char)((v >> 8) & 0xff);
}

static void *synth_worker_main(void *arg) {
  TtsState *state = (TtsState *)arg;
  for (;;) {
    Job *job = queue_pop(&state->queue);
    if (!job) break;

    if (job->is_marker) {
      AudioJob *aj = (AudioJob *)calloc(1, sizeof(AudioJob));
      if (aj) {
        aj->id = job->id;
        aj->is_marker = 1;
        aj->text = xstrdup(job->text);
        audio_queue_push(&state->audio_queue, aj);
        fprintf(stderr, "mark_ready id=%d token=%s\n", job->id, job->text);
      } else {
        fprintf(stderr, "failed id=%d reason=oom marker=%s\n", job->id, job->text);
      }
      fflush(stderr);
      free(job->text);
      free(job);
      continue;
    }

    SherpaOnnxGenerationConfig gen;
    memset(&gen, 0, sizeof(gen));
    gen.silence_scale = 0.2f;
    gen.sid = 0;
    gen.speed = 1.0f;

    double t0 = now_s();
    const SherpaOnnxGeneratedAudio *audio =
        SherpaOnnxOfflineTtsGenerateWithConfig(state->tts, job->text, &gen, NULL,
                                               NULL);
    double t1 = now_s();
    if (audio) {
      double audio_s = (double)audio->n / (double)audio->sample_rate;
      size_t bytes = (size_t)audio->n * 2;
      AudioJob *aj = (AudioJob *)calloc(1, sizeof(AudioJob));
      if (aj) {
        aj->pcm = (unsigned char *)malloc(bytes);
      }
      if (aj && aj->pcm) {
        for (int32_t i = 0; i < audio->n; ++i) {
          float_to_s16(audio->samples[i], aj->pcm + (size_t)i * 2);
        }
        aj->id = job->id;
        aj->text = xstrdup(job->text);
        aj->bytes = bytes;
        aj->samples = audio->n;
        aj->synth_s = t1 - t0;
        aj->audio_s = audio_s;
        audio_queue_push(&state->audio_queue, aj);
        fprintf(stderr,
                "synth_done id=%d synth_s=%.3f audio_s=%.3f rtf=%.3f bytes=%zu "
                "text=%s\n",
                job->id, t1 - t0, audio_s, (t1 - t0) / audio_s, bytes,
                job->text);
      } else {
        audio_job_free(aj);
        fprintf(stderr, "failed id=%d reason=oom text=%s\n", job->id, job->text);
      }
      SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);
    } else {
      fprintf(stderr, "failed id=%d text=%s\n", job->id, job->text);
    }
    fflush(stderr);
    free(job->text);
    free(job);
  }
  return NULL;
}

static void *playback_worker_main(void *arg) {
  TtsState *state = (TtsState *)arg;
  FILE *pipe = NULL;
  int pipe_rc = 0;
  if (state->play) {
    char play_cmd[768];
    snprintf(play_cmd, sizeof(play_cmd),
             "ffmpeg -hide_banner -loglevel error "
             "-f s16le -ar %d -ac 1 -i pipe:0 "
             "-f s16le -ar %d -ac %d pipe:1 | "
             "aplay -q -D %s -f S16_LE -c %d -r %d -t raw -",
             state->sample_rate, state->output_rate, state->output_channels,
             state->device, state->output_channels, state->output_rate);
    pipe = popen(play_cmd, "w");
    if (!pipe) {
      fprintf(stderr, "failed to start playback pipe: %s\n", strerror(errno));
      fflush(stderr);
    } else {
      fprintf(stderr, "playback_pipe_ready device=%s input_rate=%d output_rate=%d "
                      "output_channels=%d\n",
              state->device, state->sample_rate, state->output_rate,
              state->output_channels);
      fflush(stderr);
    }
  }

  for (;;) {
    AudioJob *aj = audio_queue_pop(&state->audio_queue);
    if (!aj) break;
    if (aj->is_marker) {
      fprintf(stderr, "mark_done id=%d token=%s\n", aj->id, aj->text ? aj->text : "");
      fflush(stderr);
      audio_job_free(aj);
      continue;
    }
    double t0 = now_s();
    size_t wrote = 0;
    if (pipe) {
      wrote = fwrite(aj->pcm, 1, aj->bytes, pipe);
      fflush(pipe);
    }
    double t1 = now_s();
    fprintf(stderr,
            "play_write id=%d write_s=%.3f audio_s=%.3f wrote=%zu/%zu "
            "synth_s=%.3f text=%s\n",
            aj->id, t1 - t0, aj->audio_s, wrote, aj->bytes, aj->synth_s,
            aj->text ? aj->text : "");
    fflush(stderr);
    audio_job_free(aj);
  }

  if (pipe) {
    pipe_rc = pclose(pipe);
    fprintf(stderr, "playback_pipe_closed rc=%d\n", pipe_rc);
    fflush(stderr);
  }
  return NULL;
}

static void usage(const char *prog) {
  fprintf(stderr,
          "Usage: %s [--no-play] [--device hw:CARD=sndes8326,DEV=0]\n"
          "Protocol on stdin:\n"
          "  raw UTF-8 text line: append LLM output chunk\n"
          "  ::flush             synthesize remaining incomplete ending\n"
          "  ::mark TOKEN        report after prior audio jobs enter playback\n"
          "  ::reset             clear pending text/jobs\n"
          "  ::quit              exit\n",
          prog);
}

int main(int argc, char **argv) {
  TtsState state;
  memset(&state, 0, sizeof(state));
  state.play = 1;
  state.output_rate = 48000;
  state.output_channels = 2;
  snprintf(state.device, sizeof(state.device), "hw:CARD=sndes8326,DEV=0");

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--no-play") == 0) {
      state.play = 0;
    } else if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
      snprintf(state.device, sizeof(state.device), "%s", argv[++i]);
    } else if (strcmp(argv[i], "--help") == 0) {
      usage(argv[0]);
      return 0;
    }
  }

  const char *base = getenv("XIAODIE_TTS_BASE");
  if (!base || !base[0]) base = "/home/vicky/xiaodie/tts";
  int threads = 6;
  int max_sentences = 4;
  const char *e = getenv("XIAODIE_TTS_THREADS");
  if (e && e[0]) threads = atoi(e);
  e = getenv("XIAODIE_TTS_SENTENCES");
  if (e && e[0]) max_sentences = atoi(e);
  e = getenv("XIAODIE_TTS_OUTPUT_RATE");
  if (e && e[0]) state.output_rate = atoi(e);
  e = getenv("XIAODIE_TTS_OUTPUT_CHANNELS");
  if (e && e[0]) state.output_channels = atoi(e);
  e = getenv("XIAODIE_TTS_CHUNK_MIN_CHARS");
  if (e && e[0]) g_min_chars = atoi(e);
  e = getenv("XIAODIE_TTS_CHUNK_MAX_CHARS");
  if (e && e[0]) g_max_chars = atoi(e);
  if (g_min_chars < 1) g_min_chars = 1;
  if (g_max_chars < g_min_chars) g_max_chars = g_min_chars;

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
  if (!model || !tokens || !lexicon || !rules) return 1;

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

  double t0 = now_s();
  state.tts = SherpaOnnxCreateOfflineTts(&config);
  double t1 = now_s();
  if (!state.tts) {
    fprintf(stderr, "failed to load Chaowen full\n");
    return 1;
  }
  state.sample_rate = SherpaOnnxOfflineTtsSampleRate(state.tts);
  fprintf(stderr,
          "ready load_s=%.3f sample_rate=%d output_rate=%d output_channels=%d "
          "speakers=%d threads=%d max_sentences=%d chunk_chars=%d-%d play=%d "
          "device=%s\n",
          t1 - t0, state.sample_rate, state.output_rate, state.output_channels,
          SherpaOnnxOfflineTtsNumSpeakers(state.tts), threads, max_sentences,
          g_min_chars, g_max_chars, state.play, state.device);
  fflush(stderr);

  queue_init(&state.queue);
  audio_queue_init(&state.audio_queue);
  pthread_t synth_worker;
  pthread_t playback_worker;
  pthread_create(&playback_worker, NULL, playback_worker_main, &state);
  pthread_create(&synth_worker, NULL, synth_worker_main, &state);

  char *buf = xstrdup("");
  char line[4096];
  while (fgets(line, sizeof(line), stdin)) {
    size_t n = strlen(line);
    while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = 0;
    if (strcmp(line, "::quit") == 0) {
      break;
    }
    if (strcmp(line, "::reset") == 0) {
      queue_reset(&state.queue);
      audio_queue_reset(&state.audio_queue);
      if (buf) buf[0] = 0;
      fprintf(stderr, "reset ok\n");
      fflush(stderr);
      continue;
    }
    if (strcmp(line, "::flush") == 0) {
      char *seg = NULL;
      while (extract_ready_segment(&buf, 1, &seg)) {
        queue_push(&state.queue, seg);
        seg = NULL;
      }
      continue;
    }
    if (starts_with(line, "::mark ")) {
      char *seg = NULL;
      while (extract_ready_segment(&buf, 1, &seg)) {
        queue_push(&state.queue, seg);
        seg = NULL;
      }
      queue_push_marker(&state.queue, xstrdup(line + 7));
      continue;
    }
    append_text(&buf, line);
    char *seg = NULL;
    while (extract_ready_segment(&buf, 0, &seg)) {
      queue_push(&state.queue, seg);
      seg = NULL;
    }
  }

  char *seg = NULL;
  while (extract_ready_segment(&buf, 1, &seg)) {
    queue_push(&state.queue, seg);
    seg = NULL;
  }
  free(buf);
  queue_close(&state.queue);
  pthread_join(synth_worker, NULL);
  audio_queue_close(&state.audio_queue);
  pthread_join(playback_worker, NULL);
  SherpaOnnxDestroyOfflineTts(state.tts);
  free(model);
  free(tokens);
  free(lexicon);
  free(rules);
  return 0;
}
