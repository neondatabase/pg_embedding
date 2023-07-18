#include "postgres.h"
#include "embedding.h"
#include "math.h"

dist_t l2_dist_impl(coord_t const* ax, coord_t const* bx, size_t dim)
{
	dist_t 		distance = 0.0;
	for (size_t i = 0; i < dim; i++)
	{
		dist_t diff = ax[i] - bx[i];
		distance += diff * diff;
	}
	return distance;
}

dist_t cosine_dist_impl(coord_t const* ax, coord_t const* bx, size_t dim)
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

dist_t manhattan_dist_impl(coord_t const* ax, coord_t const* bx, size_t dim)
{
	dist_t 		distance = 0.0;
	for (size_t i = 0; i < dim; i++)
	{
		distance += fabs(ax[i] - bx[i]);
	}
	return distance;
}

static dist_t (*const dist_func_table[])(coord_t const* ax, coord_t const* bx, size_t size) =
{
	l2_dist_impl,
	cosine_dist_impl,
	manhattan_dist_impl
};

dist_t hnsw_dist_func(HnswMetadata* meta, coord_t const* ax, coord_t const* bx)
{
	return dist_func_table[meta->dist_func](ax, bx, meta->dim);
}
