#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <immintrin.h>

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

int closet_8x(int x) {
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

void read_pixel(float* A_adr, float* R_adr, float* G_adr, float* B_adr, ARGB* pixel, FILE* src){
  fread(pixel, sizeof(ARGB), 1, src);
  *A_adr = (float)(pixel->ALPHA / 255.0);
  *R_adr = (float)(pixel->RED / 255.0);
  *G_adr = (float)(pixel->GREEN / 255.0);
  *B_adr = (float)(pixel->BLUE / 255.0);
}

int main(int argc, char* argv[]) {
  printf("NEW)");
  if (argc < 3) {
    return 1;
  }

  FILE* back = fopen(argv[1], "rb");
  FILE* front = fopen(argv[2], "rb");
  FILE* res = fopen(argv[3], "wb");

  if (back == NULL || front == NULL || res == NULL) {
    return 2;
  }

  BITMAPFILEHEADER BH_front;
  BITMAPINFO BI_front;

  fread(&BH_front, sizeof(BH_front), 1, front);
  fread(&BI_front, sizeof(BI_front), 1, front);

  BITMAPFILEHEADER BH_back;
  BITMAPINFO BI_back;

  fread(&BH_back, sizeof(BH_back), 1, back);
  fread(&BI_back, sizeof(BI_back), 1, back);


  fseek(back, 0, SEEK_END);
  size_t sz = ftell(back);
  fseek(back, 0, SEEK_SET);
  uint8_t* buf = malloc(sz);
  fread(buf, sz, 1, back);
  fwrite(buf, sz, 1, res);

  fseek(back, BH_back.bfOffBits, SEEK_SET);
  fseek(front, BH_front.bfOffBits, SEEK_SET);

  if(BI_back.biWidth != BI_front.biWidth || BI_back.biHeight != BI_front.biHeight) {
    fclose(back);
    fclose(front);
    fclose(res);
    return 3;
  }

  int width = BI_back.biWidth;
  int height = BI_back.biHeight;

  int wh = width * height;
  int wh_8x = closet_8x(wh);

  float* A_back = (float*)aligned_alloc(32, sizeof(float) * 8 * 12);
  float* R_back = A_back + 8;
  float* G_back = R_back + 8;
  float* B_back = G_back + 8;

  float* A_front = B_back + 8;
  float* R_front = A_front + 8;
  float* G_front = R_front + 8;
  float* B_front = G_front + 8;

  float* A_res = B_front + 8;
  float* R_res = A_res + 8;
  float* G_res = R_res + 8;
  float* B_res = G_res + 8;

  ARGB argb_back;
  ARGB argb_front;
  ARGB argb_res;

  fseek(res, BH_back.bfOffBits, SEEK_SET);

  for(int i = 0; i < wh_8x; ++i) {
    read_pixel(A_front +  (i % 8), R_front + (i % 8), G_front + (i % 8), B_front + (i % 8), &argb_front, front);
    read_pixel(A_back + (i % 8), R_back + (i % 8), G_back + (i % 8), B_back + (i % 8), &argb_back, back);

    if (i % 8 == 7) {
      __m256 A_back_m256 = _mm256_load_ps(A_back);
      __m256 A_front_m256 = _mm256_load_ps(A_front);

      __m256 A_mul_m256 = _mm256_mul_ps(A_back_m256, A_front_m256);
      __m256 A_sum_m256 = _mm256_add_ps(A_back_m256, A_front_m256);
      __m256 A_res_m256 = _mm256_sub_ps(A_sum_m256, A_mul_m256);

      _mm256_store_ps(A_res, A_res_m256);

      calc_pixel_color(R_res, R_front, R_back, A_res_m256, A_front_m256, A_back_m256);
      calc_pixel_color(G_res, G_front, G_back, A_res_m256, A_front_m256, A_back_m256);
      calc_pixel_color(B_res, B_front, B_back, A_res_m256, A_front_m256, A_back_m256);

      int write_count = (8) * (8 <= wh - i) + (wh - i) * (8 > wh - i);
      
      for(int j = 0; j < write_count ; ++j) {
        argb_res.ALPHA = (uint8_t)(A_res[j] * 255);
        argb_res.RED   = (uint8_t)(R_res[j] * 255);
        argb_res.GREEN = (uint8_t)(G_res[j] * 255);
        argb_res.BLUE  = (uint8_t)(B_res[j] * 255);

        fwrite(&argb_res, sizeof(ARGB), 1, res);
      }
    }
  }

  free(A_back);

  fclose(back);
  fclose(front);
  fclose(res);
  return 0;
}
