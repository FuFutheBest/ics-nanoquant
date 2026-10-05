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
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nano_quant.h"

// static void todo(const char *who) {
//   fprintf(stderr, "nano_quant.cpp: %s is not implemented yet\n", who);
//   exit(3);
// }

/* ================= Part A: reading and assembling bits ================= */

/*
 * +------+------+----------+----------+-------+
 * | type | sign | exponent | fraction | total |
 * +------+------+----------+----------+-------+
 * | fp16 | 1    | 5        | 10       | 16    |
 * | f32  | 1    | 8        | 23       | 32    |
 * | bf16 | 1    | 8        | 7        | 16    |
 * +------+------+----------+----------+-------+
 *
 * f32:
 * Normal     0<exp<255
 * Subnormal  exp=0
 * Zero       exp=0       frac=0
 * Infinity   exp=255     frac=0
 * NaN        exp=255     frac!=0
 */

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
  float amax = x[0];
  for (int i = 1; i < 32; ++i) {
    if (fabsf(x[i]) > fabsf(amax)) {
      amax = x[i];
    }
  }
  float d = amax / -8.0f;
  float id = (d != 0) ? 1.0f / d : 0.0f;

  uint16_t fp16_d = f32_to_fp16(d);
  memcpy(&blk[0], &fp16_d, 2);

  for (int i = 0; i < 32; ++i) {
    int q = (int)(x[i] * id + 8.5f);
    if (q < 0)
      q = 0;
    if (q > 15)
      q = 15;

    if (i < 16) {
      blk[2 + i] = (uint8_t)q;
    } else {
      blk[2 + (i - 16)] |= (uint8_t)(q << 4);
    }
  }
}

void q4_0_dequantize(const uint8_t *blk, float *x) {
  uint16_t fp16_d;
  memcpy(&fp16_d, &blk[0], 2);
  float d = fp16_to_f32(fp16_d);

  int q;
  for (int i = 0; i < 32; ++i) {
    if (i < 16) {
      q = blk[2 + i] & 0x0F;
    } else {
      q = (blk[2 + (i - 16)] >> 4) & 0x0F;
    }
    x[i] = d * (float)(q - 8);
  }
}

void q4_1_quantize(const float *x, uint8_t *blk) {
  float hi = x[0];
  float lo = x[0];
  for (int i = 1; i < 32; ++i) {
    if (x[i] > hi) {
      hi = x[i];
    }
    if (x[i] < lo) {
      lo = x[i];
    }
  }
  float d = (hi - lo) / 15.0f;
  float id = d != 0 ? 1.0f / d : 0.0f;
  float m = lo;
  uint16_t fp16_d = f32_to_fp16(d);
  uint16_t fp16_m = f32_to_fp16(m);
  memcpy(&blk[0], &fp16_d, 2);
  memcpy(&blk[2], &fp16_m, 2);
  for (int i = 0; i < 32; ++i) {
    int q = (int)((x[i] - lo) * id + 0.5f);
    if (q < 0)
      q = 0;
    if (q > 15)
      q = 15;
    if (i < 16) {
      blk[4 + i] = (uint8_t)q;
    } else {
      blk[4 + (i - 16)] |= (uint8_t)(q << 4);
    }
  }
}

void q4_1_dequantize(const uint8_t *blk, float *x) {
  float d, m;
  uint16_t fp16_d, fp16_m;
  memcpy(&fp16_d, &blk[0], 2);
  memcpy(&fp16_m, &blk[2], 2);
  d = fp16_to_f32(fp16_d);
  m = fp16_to_f32(fp16_m);

  int q;
  for (int i = 0; i < 32; ++i) {
    if (i < 16) {
      q = blk[4 + i] & 0x0F;
    } else {
      q = (blk[4 + (i - 16)] >> 4) & 0x0F;
    }
    x[i] = d * (float)q + m;
  }
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
  if (j < 4) {
    uint8_t q_sc = q[j];
    uint8_t q_m = q[j + 4];
    uint8_t sc_write = (q_sc & 0xc0) | (sc & 0x3f);
    uint8_t m_write = (q_m & 0xc0) | (m & 0x3f);
    memcpy(q + j, &sc_write, 1);
    memcpy(q + j + 4, &m_write, 1);
  } else {
    uint8_t sc_lo4 = sc & 0xf;
    uint8_t sc_hi2 = (sc >> 4) & 0x3;
    uint8_t m_lo4 = (m & 0xf);
    uint8_t m_hi2 = (m >> 4) & 0x3;
    uint8_t q_sc = q[j - 4];
    uint8_t q_m = q[j];

    uint8_t lo_write = sc_lo4 | (m_lo4 << 4);
    uint8_t hi_sc_write = (q_sc & 0x3f) | (sc_hi2 << 6);
    uint8_t hi_m_write = (q_m & 0x3f) | (m_hi2 << 6);

    memcpy(q + j + 4, &lo_write, 1);
    memcpy(q + j - 4, &hi_sc_write, 1);
    memcpy(q + j, &hi_m_write, 1);
  }
}

void q4_k_quantize(const float *x, uint8_t *blk) {
  float s[8], o[8];
  for (int j = 0; j < 8; ++j) {
    nq_q4_k_fit(x + 32 * j, &s[j], &o[j]);
  }

  float smax = s[0], omax = o[0];
  for (int j = 1; j < 8; ++j) {
    if (s[j] > smax)
      smax = s[j];
    if (o[j] > omax)
      omax = o[j];
  }

  if (smax == -0.0f) {
    smax = 0.0f;
  }

  // << 请输入文本 >>
  if (omax == -0.0f) {
    omax = 0.0f;
  }

  float d = smax / 63.0f;
  float dmin = omax / 63.0f;
  float is = (smax > 0) ? 63.0f / smax : 0.0f;
  float io = (omax > 0) ? 63.0f / omax : 0.0f;

  uint16_t fp16_d = f32_to_fp16(d);
  uint16_t fp16_dmin = f32_to_fp16(dmin);

  memcpy(&blk[0], &fp16_d, 2);
  memcpy(&blk[2], &fp16_dmin, 2);

  float d1 = fp16_to_f32(fp16_d);
  float dmin1 = fp16_to_f32(fp16_dmin);

  uint8_t sc, m;
  float D, M;
  int idx, q;
  uint8_t blk_lo, blk_write;
  for (int j = 0; j < 8; ++j) {
    sc = (uint8_t)nq_round(is * s[j]);
    m = (uint8_t)nq_round(io * o[j]);
    put_scale_min(j, blk + 4, sc, m);

    D = d1 * (float)sc;
    M = dmin1 * (float)m;
    for (int i = 0; i < 32; ++i) {
      idx = j * 32 + i;
      q = (D != 0) ? (int)nq_round((x[idx] + M) / D) : 0;
      if (q < 0)
        q = 0;
      if (q > 15)
        q = 15;
      if (j % 2 == 0) {
        memcpy(&blk[16 + 32 * (j / 2) + i], &q, 1);
      } else {
        blk_lo = blk[16 + 32 * (j / 2) + i];
        blk_write = (blk_lo & 0x0F) | ((q & 0x0F) << 4);
        memcpy(&blk[16 + 32 * (j / 2) + i], &blk_write, 1);
      }
    }
  }
}

void q4_k_dequantize(const uint8_t *blk, float *x) {
  uint16_t fp16_d, fp16_dmin;
  memcpy(&fp16_d, &blk[0], 2);
  memcpy(&fp16_dmin, &blk[2], 2);

  float d = fp16_to_f32(fp16_d);
  float dmin = fp16_to_f32(fp16_dmin);

  uint8_t sc, m;
  for (int i = 0; i < 8; ++i) {
    get_scale_min(i, blk + 4, &sc, &m);
    float D = d * (float)sc;
    float M = dmin * (float)m;

    for (int j = 0; j < 32; ++j) {
      int idx = i * 32 + j;
      int q;
      if (i % 2 == 0) {
        q = blk[16 + 32 * (i / 2) + j] & 0x0F;
      } else {
        q = (blk[16 + 32 * (i / 2) + j] >> 4) & 0x0F;
      }
      x[idx] = D * (float)q - M;
    }
  }
}
