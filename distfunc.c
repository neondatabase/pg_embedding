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

__attribute__((target("avx2")))
static dist_t l2_dist_impl_avx2(const coord_t *x, const coord_t *y, size_t n)
{
	coord_t PORTABLE_ALIGN32 TmpRes[sizeof(__m256) / sizeof(float)];
    size_t qty16 = n / 16;
    const coord_t *pEnd1 = x + (qty16 * 16);
    const coord_t *pEnd2 = x + n;
    __m256 diff, v1, v2;
    __m256 sum = _mm256_set1_ps(0);
	dist_t res;

    while (x < pEnd1) {
        v1 = _mm256_loadu_ps(x);
        x += 8;
        v2 = _mm256_loadu_ps(y);
        y += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

        v1 = _mm256_loadu_ps(x);
        x += 8;
        v2 = _mm256_loadu_ps(y);
        y += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    }
    _mm256_store_ps(TmpRes, sum);
    res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];

    // Handle case when dimensions is not aligned on 16.
    while (x < pEnd2)
    {
        dist_t diff = *x++ - *y++;
        res += diff * diff;
    }

	return sqrtf(res);
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
    dist_func_table[DIST_L2] = __builtin_cpu_supports("avx2")
		? l2_dist_impl_avx2 : l2_dist_impl_sse;
#else
	dist_func_table[DIST_L2] = l2_dist_impl;
#endif
	dist_func_table[DIST_COSINE] = cosine_dist_impl;
	dist_func_table[DIST_MANHATTAN] = manhattan_dist_impl;
};

dist_t hnsw_dist_func(dist_func_t dist_func, coord_t const* ax, coord_t const* bx, size_t dim)
{
	return dist_func_table[dist_func](ax, bx, dim);
}
