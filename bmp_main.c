#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <immintrin.h>

#include <time.h>
#include <stdatomic.h>
#include <sys/syscall.h>
#include <unistd.h>
const char* log_prefix(const char* func, int line) {
  struct timespec spec; clock_gettime(CLOCK_REALTIME, &spec);
  long long current_msec = spec.tv_sec * 1000L + spec.tv_nsec / 1000000;
  static _Atomic long long start_msec_storage = -1;
  long long start_msec = -1;
  if (atomic_compare_exchange_strong(&start_msec_storage, &start_msec, current_msec))
    start_msec = current_msec;
  long long delta_msec = current_msec - start_msec;
  const int max_func_len = 10;
  static __thread char prefix[100];
  sprintf(prefix, "%lld.%03lld %*s():%d    ", delta_msec / 1000, delta_msec % 1000, max_func_len, func, line);
  sprintf(prefix + max_func_len + 13, "[tid=%ld]", syscall(__NR_gettid));
  return prefix;
}
#define log_printf_impl(fmt, ...) { time_t t = time(0); dprintf(2, "%s: " fmt "%s", log_prefix(__FUNCTION__, __LINE__), __VA_ARGS__); }
// Format: <time_since_start> <func_name>:<line> : <custom_message>
#define log_printf(...) log_printf_impl(__VA_ARGS__, "")
#define SWAP(a, b) { __typeof__(a) c = (a); (a) = (b); (b) = (c); }

#define BUFFER_CAPACITY 4096

struct __attribute__ ((packed)) BITMAPFILEHEADER {
  uint16_t bfType;
  uint32_t bfSize;
  uint16_t bfReserved1;
  uint16_t bfReserved2;
  uint32_t bfOffBits;
};
typedef struct BITMAPFILEHEADER BITMAPFILEHEADER;

struct __attribute__ ((packed)) BITMAPINFO {
  uint32_t biSize;
  uint32_t biWidth;
  uint32_t biHeight;
  uint16_t biPlanes;
  uint16_t biBitCount;
  uint32_t biCompression;
  uint32_t biSizeImage;
  uint32_t biXPelsPerMeter;
  uint32_t biYPelsPerMeter;
  uint32_t biClrUsed;
  uint32_t biClrImportant;
};
typedef struct BITMAPINFO BITMAPINFO;

struct ARGB {
  uint8_t BLUE;
  uint8_t GREEN;
  uint8_t RED;
  uint8_t ALPHA;
};
typedef struct ARGB ARGB;

uint32_t closet_8x(uint32_t x) {
  if ((x % 8) == 0) {
    return x;
  }
  else {
    return x + 8 - (x % 8);
  }
}

void calc_pixel_color(float* res_adr, float* front_adr, float* back_adr, __m256 alpha_res, __m256 alpha_front, __m256 alpha_back) {
  __m256 front_color = _mm256_load_ps(front_adr);
  __m256 front_part = _mm256_mul_ps(front_color, alpha_front);

  __m256 back_color = _mm256_load_ps(back_adr);
  __m256 mul_alpha = _mm256_mul_ps(alpha_front, alpha_back);
  __m256 back_alpha_part = _mm256_sub_ps(alpha_back, mul_alpha);
  __m256 back_part = _mm256_mul_ps(back_color, back_alpha_part);

  __m256 res_color = _mm256_add_ps(front_part, back_part);
  res_color = _mm256_div_ps(res_color, alpha_res);
  _mm256_store_ps(res_adr, res_color);
}

void read_pixels(float* A_adr, float* R_adr, float* G_adr, float* B_adr, FILE* src) {
  ARGB buf[BUFFER_CAPACITY];
  fread(buf, sizeof(ARGB), BUFFER_CAPACITY, src);

  for (int i = 0; i < BUFFER_CAPACITY; ++i) {
    A_adr[i] = (float)(buf[i].ALPHA / 255.0);
    R_adr[i] = (float)(buf[i].RED / 255.0);
    G_adr[i] = (float)(buf[i].GREEN / 255.0);
    B_adr[i] = (float)(buf[i].BLUE / 255.0);
  }
}

void copy_file(FILE* src, FILE* dst) {
  fseek(src, 0, SEEK_END);
  size_t sz = ftell(src);
  fseek(src, 0, SEEK_SET);
  uint8_t* buf = malloc(sz);
  fread(buf, sz, 1, src);
  fwrite(buf, sz, 1, dst);
  free(buf);
}

void full_calc_8pix(
    float* A_res,   float* R_res,   float* G_res,   float* B_res,
    float* A_front, float* R_front, float* G_front, float* B_front,
    float* A_back,  float* R_back,  float* G_back,  float* B_back
    ) {
  __m256 A_back_m256 = _mm256_load_ps(A_back);
  __m256 A_front_m256 = _mm256_load_ps(A_front);
  __m256 A_mul_m256 = _mm256_mul_ps(A_back_m256, A_front_m256);
  __m256 A_sum_m256 = _mm256_add_ps(A_back_m256, A_front_m256);
  __m256 A_res_m256 = _mm256_sub_ps(A_sum_m256, A_mul_m256);

  _mm256_store_ps(A_res, A_res_m256);
  calc_pixel_color(R_res, R_front, R_back, A_res_m256, A_front_m256, A_back_m256);
  calc_pixel_color(G_res, G_front, G_back, A_res_m256, A_front_m256, A_back_m256);
  calc_pixel_color(B_res, B_front, B_back, A_res_m256, A_front_m256, A_back_m256);
}

int main(int argc, char* argv[]) {
  log_printf("Program started\n");

  //uint32_t numCPU = sysconf(_SC_NPROCESSORS_ONLN);
  //printf("PROCESSORS COUNT: %d", numCPU);

  if (argc < 3) {
    perror("Not enough arguments.");
    return 1;
  }

  FILE* back = fopen(argv[1], "rb");

  if (back == NULL) {
    perror("Background image failed to open.");
    return 2;
  }

  FILE* front = fopen(argv[2], "rb");

  if (front == NULL) {
    perror("Foreground image failed to open.");
    return 2;
  }

  FILE* result = fopen(argv[3], "wb");

  if (result == NULL) {
    perror("Resulting image failed to open/create.");
    return 2;
  }

  log_printf("All files opened\n");

  BITMAPFILEHEADER BH_front;
  BITMAPINFO BI_front;

  fread(&BH_front, sizeof(BH_front), 1, front);
  fread(&BI_front, sizeof(BI_front), 1, front);

  BITMAPFILEHEADER BH_back;
  BITMAPINFO BI_back;

  fread(&BH_back, sizeof(BH_back), 1, back);
  fread(&BI_back, sizeof(BI_back), 1, back);

  copy_file(back, result);

  fseek(front, BH_front.bfOffBits, SEEK_SET);
  fseek(back, BH_back.bfOffBits, SEEK_SET);
  fseek(result, BH_back.bfOffBits, SEEK_SET);

  if (BI_back.biWidth != BI_front.biWidth || BI_back.biHeight != BI_front.biHeight) {
    perror("Images have different resolutions.");
    fclose(back);
    fclose(front);
    fclose(result);
    return 3;
  }

  uint32_t width = BI_back.biWidth;
  uint32_t height = BI_back.biHeight;

  uint32_t wh = width * height;
  uint32_t wh_8x = closet_8x(wh);

  log_printf("Allocating started\n");

  float* A_back = (float*)aligned_alloc(32, sizeof(float) * BUFFER_CAPACITY);
  float* R_back = (float*)aligned_alloc(32, sizeof(float) * BUFFER_CAPACITY);
  float* G_back = (float*)aligned_alloc(32, sizeof(float) * BUFFER_CAPACITY);
  float* B_back = (float*)aligned_alloc(32, sizeof(float) * BUFFER_CAPACITY);

  float* A_front = (float*)aligned_alloc(32, sizeof(float) * BUFFER_CAPACITY);
  float* R_front = (float*)aligned_alloc(32, sizeof(float) * BUFFER_CAPACITY);
  float* G_front = (float*)aligned_alloc(32, sizeof(float) * BUFFER_CAPACITY);
  float* B_front = (float*)aligned_alloc(32, sizeof(float) * BUFFER_CAPACITY);

  float* A_res = (float*)aligned_alloc(32, sizeof(float) * BUFFER_CAPACITY);
  float* R_res = (float*)aligned_alloc(32, sizeof(float) * BUFFER_CAPACITY);
  float* G_res = (float*)aligned_alloc(32, sizeof(float) * BUFFER_CAPACITY);
  float* B_res = (float*)aligned_alloc(32, sizeof(float) * BUFFER_CAPACITY);

  log_printf("Allocating finished\n");

  ARGB res_buffer[BUFFER_CAPACITY];

  for(int i = 0; i < wh_8x; i += BUFFER_CAPACITY) {
    read_pixels(A_front, R_front, G_front, B_front, front);
    read_pixels(A_back, R_back, G_back, B_back, back);

    for (int j = 0; j < BUFFER_CAPACITY; j += 8) {
      full_calc_8pix(
          A_res + j,   R_res + j,   G_res + j,   B_res + j,
          A_front + j, R_front + j, G_front + j, B_front + j,
          A_back + j,  R_back + j,  G_back + j,  B_back + j
          );
    }

    ARGB argb_res;

    for (int j = 0; j < BUFFER_CAPACITY; ++j) {
      argb_res.ALPHA = (uint8_t)(A_res[j] * 255);
      argb_res.RED   = (uint8_t)(R_res[j] * 255);
      argb_res.GREEN = (uint8_t)(G_res[j] * 255);
      argb_res.BLUE  = (uint8_t)(B_res[j] * 255);

      res_buffer[j] = argb_res;
    }

    fwrite(res_buffer, sizeof(ARGB), BUFFER_CAPACITY, result);
  }

  free(A_back);
  free(R_back);
  free(G_back);
  free(B_back);

  free(A_front);
  free(R_front);
  free(G_front);
  free(B_front);

  free(A_res);
  free(R_res);
  free(G_res);
  free(B_res);

  fclose(back);
  fclose(front);
  fclose(result);

  log_printf("Program finished\n");
  return 0;
}
