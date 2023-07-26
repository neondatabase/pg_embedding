#include "postgres.h"
#include "embedding.h"
#include "math.h"

#ifdef __x86_64__
#include <immintrin.h>

#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#endif

__attribute__((target("avx2,fma")))
static dist_t l2_dist_impl_avx2(const coord_t *x, const coord_t *y, size_t n)
{
    const size_t elts_per_vector = sizeof(__m256) / sizeof(float);
    const size_t unroll = 2;
    const size_t elts_per_loop = elts_per_vector * unroll;
    float partial_result[elts_per_vector];
    __m256 accum1 = _mm256_setzero_ps();
    __m256 accum2 = _mm256_setzero_ps();
    for(size_t i = 0; i < n; i += elts_per_loop)
    {
        __m256 vecX1 = _mm256_loadu_ps(x + i);
        __m256 vecY1 = _mm256_loadu_ps(y + i);
        __m256 vecSub1 = _mm256_sub_ps(vecX1, vecY1);
        accum1 = _mm256_fmadd_ps(vecSub1, vecSub1, accum1);

        __m256 vecX2 = _mm256_loadu_ps(x + i + elts_per_vector);
        __m256 vecY2 = _mm256_loadu_ps(y + i + elts_per_vector);
        __m256 vecSub2 = _mm256_sub_ps(vecX2, vecY2);
        accum2 = _mm256_fmadd_ps(vecSub2, vecSub2, accum2);
    }
    accum1 = _mm256_add_ps(accum1, accum2);

    // Do final full vector calculations, not unrolled
    size_t tail = n - n % elts_per_loop;
    // The tail within the tail; the last few elements that don't even need a 
    // whole __m256
    size_t subtail_size = n % elts_per_vector;
    size_t subtail = n - subtail_size;
    for(size_t i = tail; i < subtail; i += elts_per_vector)
    {
        __m256 vecX1 = _mm256_loadu_ps(x + tail + i);
        __m256 vecY1 = _mm256_loadu_ps(y + tail + i);
        __m256 vecSub1 = _mm256_sub_ps(vecX1, vecY1);
        accum1 = _mm256_fmadd_ps(vecSub1, vecSub1, accum1);
    }

#if defined(__AVX512VL__)
    // the final set of elements, less than elts_per_vector
    __mmask8 mask = (1 << subtail_size) - 1;
    __m256 x_rest = _mm256_maskz_loadu_ps(mask, x + subtail);
    __m256 y_rest = _mm256_maskz_loadu_ps(mask, y + subtail);
    __m256 sub_rest = _mm256_sub_ps(x_rest, y_rest);
    accum1 = _mm256_fmadd_ps(sub_rest, sub_rest, accum1);
#endif

    _mm256_storeu_ps(partial_result, accum1);
    float res1 = partial_result[0] + partial_result[4];
    float res2 = partial_result[1] + partial_result[5];
    float res3 = partial_result[2] + partial_result[6];
    float res4 = partial_result[3] + partial_result[7];
    res1 += res3;
    res2 += res4;
    res1 += res2;

#if !defined(__AVX512VL__)
    for(size_t i = subtail; i < n; i++)
    {
        float dist = x[i] - y[i];
        res1 += (dist * dist);
    }
#endif

    return sqrtf(res1);
}

static dist_t l2_dist_impl_sse(const coord_t *x, const coord_t *y, size_t n)
{
    coord_t PORTABLE_ALIGN32 TmpRes[sizeof(__m128) / sizeof(float)];
    size_t qty16 = n / 16;
    const coord_t *pEnd1 = x + (qty16 * 16);
    const coord_t *pEnd2 = x + n;
	dist_t res;

    __m128 diff, v1, v2;
    __m128 sum = _mm_set1_ps(0);

    while (x < pEnd1) {
        v1 = _mm_loadu_ps(x);
        x += 4;
        v2 = _mm_loadu_ps(y);
        y += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(x);
        x += 4;
        v2 = _mm_loadu_ps(y);
        y += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(x);
        x += 4;
        v2 = _mm_loadu_ps(y);
        y += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(x);
        x += 4;
        v2 = _mm_loadu_ps(y);
        y += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }
    _mm_store_ps(TmpRes, sum);
    res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

	// Handle case when dimensions is not aligned on 16.
    while (x < pEnd2)
    {
        dist_t diff = *x++ - *y++;
        res += diff * diff;
    }

    return sqrtf(res);
}

#else

static dist_t l2_dist_impl(coord_t const* ax, coord_t const* bx, size_t dim)
{
	dist_t 		distance = 0.0;
	for (size_t i = 0; i < dim; i++)
	{
		dist_t diff = ax[i] - bx[i];
		distance += diff * diff;
	}
	return sqrtf(distance);
}

#endif

static dist_t cosine_dist_impl(coord_t const* ax, coord_t const* bx, size_t dim)
{
	dist_t 		distance = 0.0;
	dist_t 		norma = 0.0;
	dist_t 		normb = 0.0;
	for (size_t i = 0; i < dim; i++)
	{
		distance += ax[i] * bx[i];
		norma += ax[i] * ax[i];
		normb += bx[i] * bx[i];
	}
	return 1 - (distance / sqrt(norma * normb));
}

static dist_t manhattan_dist_impl(coord_t const* ax, coord_t const* bx, size_t dim)
{
	dist_t 		distance = 0.0;
	for (size_t i = 0; i < dim; i++)
	{
		distance += fabs(ax[i] - bx[i]);
	}
	return distance;
}

static dist_t (*dist_func_table[3])(coord_t const* ax, coord_t const* bx, size_t size);

void hnsw_init_dist_func(void)
{
#ifdef __x86_64__
    dist_func_table[DIST_L2] = (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
        ? l2_dist_impl_avx2 : l2_dist_impl_sse;
#else
	dist_func_table[DIST_L2] = l2_dist_impl;
#endif
	dist_func_table[DIST_COSINE] = cosine_dist_impl;
	dist_func_table[DIST_MANHATTAN] = manhattan_dist_impl;
};

dist_t hnsw_dist_func(dist_func_t func, coord_t const* ax, coord_t const* bx, size_t dim)
{
	return dist_func_table[func](ax, bx, dim);
}
