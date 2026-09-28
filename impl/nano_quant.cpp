/* nano_quant.cpp -- all the code you write goes in this file.
 *
 * Replace the todo() call in each function with an implementation. What each
 * function must compute is in include/nano_quant.h. Keep the prototypes as
 * they are and add nothing to the header: the framework only calls what the
 * header declares.
 *
 *   make                    build nano-quant, nq2gguf and nq-selftest
 *   ./nq-selftest a q4_0    check parts A and B, group by group
 *   make test               build, make the test files, run tests/run.sh
 *
 * Do the arithmetic in float, never through double, and keep the compiler
 * flags in the Makefile.
 */
#include <cstdint>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nano_quant.h"

static void todo(const char *who) {
  fprintf(stderr, "nano_quant.cpp: %s is not implemented yet\n", who);
  exit(3);
}

/* ================= Part A: reading and assembling bits ================= */

uint64_t rd_u64le(const uint8_t *p) {
  uint64_t res = 0;
  for (int i = 0; i < 8; ++i) {
    res += ((uint64_t)(*(p + i))) << (8 * i);
  }
  return res;
}

float bf16_to_f32(uint16_t h) {
  float res = 0.0f;
  uint32_t bits = ((uint32_t)h) << 16;
  memcpy(&res, &bits, sizeof(res));
  return res;
}

float fp16_to_f32(uint16_t h) {
  uint32_t sign = ((uint32_t)(h & 0x8000u)) << 16;
  uint32_t exp = (h >> 10) & 0x1fu; // 5 bits
  uint32_t frac = h & 0x3ffu;       // 10 bits

  uint32_t bits = 0;

  if (exp == 0x1f) { // NaN or Infinity
    uint32_t exp32 = 0xffu << 23;
    uint32_t frac32 = frac << 13;
    bits = sign | exp32 | frac32;
  } else if (exp == 0) {
    if (frac == 0) { // Zero
      bits = sign;
    } else { // Subnormal
      int e = -14;
      uint32_t f = frac;

      while ((f & 0x400u) == 0) {
        f <<= 1;
        --e;
      }

      f &= 0x3ffu;

      uint32_t exp32 = (uint32_t)(e + 127) << 23;
      uint32_t frac32 = f << 13;

      bits = sign | exp32 | frac32;
    }
  } else { // Normalized
    uint32_t exp32 = (exp - 15 + 127) << 23;
    uint32_t frac32 = frac << 13;
    bits = sign | exp32 | frac32;
  }

  float res = 0.0f;
  memcpy(&res, &bits, sizeof(res));
  return res;
}

uint16_t f32_to_fp16(float f) {
  uint32_t intf = 0;
  memcpy(&intf, &f, sizeof(f));

  uint32_t sign = (intf >> 31) << 15;
  uint32_t exp = (intf >> (32 - 9)) & 0xff; // 8 bits
  uint32_t frac = intf & 0x7fffff;          // 23 bits

  uint16_t bits = 0;

  if (exp == 0) { // Zero or subnormal
    bits = sign;
  } else if (exp == 0xff) {
    if (frac == 0) // Infinity
      bits = sign | 0x7c00;
    else // NaN
      bits =
          sign | 0x7e00 |
          (frac >> 13); // use 0x7e00 to keep NaN from transforming to Infinity
  } else {              // Normalized
    // exp : [1,255]
    int real_exp = exp - 127; // [-126,127]

    if (-14 <= real_exp && real_exp <= 15) { // To fp16 normalized
      uint32_t frac16 = frac >> 13;          // 10 bits
      uint32_t exp16 = real_exp + 15;

      uint32_t discarded = frac & 0x1fff; // 13 bits discarded
      uint32_t half_bit = 0x1000;         // 1 << 12

      if (discarded > half_bit || (discarded == half_bit && (frac16 & 1))) {
        ++frac16;
      }

      if (frac16 == 0x400) { // overflow in fraction
        frac16 = 0;
        ++exp16;
      }

      if (exp16 == 0x1f) {    // overflow in exponent
        bits = sign | 0x7c00; // Infinity
      } else {
        bits = sign | (exp16 << 10) | frac16;
      }
    } else if (real_exp < -14) { // Try to tranform to fp16 subnormal
      uint32_t significand =
          0x800000 | frac; // fp32 = significand × 2^(real_exp - 23)
      int shift = -real_exp - 1;

      uint32_t real_frac; // fp16 = real_frac × 2^(-14-10)
      // real_frac = significand × 2^(real_exp + 1)

      if (shift > 24) { // Too small, becomes zero
        real_frac = 0;
      } else {
        real_frac = significand >> shift;

        uint32_t discarded = significand & ((1u << shift) - 1);
        uint32_t halfway = 1u << (shift - 1);

        if (discarded > halfway || (discarded == halfway && (real_frac & 1))) {
          real_frac++;
        }
      }

      if (real_frac >=
          0x400 /** 1 << 10 (11 bits)**/) { // fp16 = 1 x 2^(-14) = 0x0400
                                            // overflow, to normalized
                                            // exp16 = -14 + 15 = 1
                                            // frac16 = 0
        bits = sign | 0x0400;
      } else { // to subnormal
        bits = sign | (real_frac & 0x3ff);
      }

    } else { // To Infinity
      bits = sign | 0x7c00;
    }
  }

  return bits;
}

/* ================= Part B: three block formats ================= */

void q4_0_quantize(const float *x, uint8_t *blk) {
  (void)x;
  (void)blk;
  todo("q4_0_quantize");
}

void q4_0_dequantize(const uint8_t *blk, float *x) {
  (void)blk;
  (void)x;
  todo("q4_0_dequantize");
}

void q4_1_quantize(const float *x, uint8_t *blk) {
  (void)x;
  (void)blk;
  todo("q4_1_quantize");
}

void q4_1_dequantize(const uint8_t *blk, float *x) {
  (void)blk;
  (void)x;
  todo("q4_1_dequantize");
}

/* Given. Write put_scale_min so that this function reads back what it wrote.
 */
void get_scale_min(int j, const uint8_t *q, uint8_t *sc, uint8_t *m) {
  if (j < 4) {
    *sc = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *sc = (uint8_t)((q[j + 4] & 0xf) | ((q[j - 4] >> 6) << 4));
    *m = (uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
  }
}

void put_scale_min(int j, uint8_t *q, uint8_t sc, uint8_t m) {
  (void)j;
  (void)q;
  (void)sc;
  (void)m;
  todo("put_scale_min");
}

void q4_k_quantize(const float *x, uint8_t *blk) {
  (void)x;
  (void)blk;
  todo("q4_k_quantize");
}

void q4_k_dequantize(const uint8_t *blk, float *x) {
  (void)blk;
  (void)x;
  todo("q4_k_dequantize");
}
