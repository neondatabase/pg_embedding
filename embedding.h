#pragma once

typedef float    coord_t;
typedef float    dist_t;
typedef uint32_t idx_t;
typedef uint64_t label_t;

typedef enum {
	DIST_L2,
	DIST_COSINE,
	DIST_MANHATTAN
} dist_func_t;

typedef struct
{
	size_t		dim;
	size_t		data_size;
	size_t		offset_data;
	size_t		offset_label;
	size_t		size_data_per_element;
	size_t		elems_per_page;
	size_t		M;
	size_t		maxM;
	size_t		efConstruction;
	size_t		efSearch;
	idx_t		enterpoint_node;
	dist_func_t dist_func;
} HnswMetadata;

extern bool hnsw_is_deleted(label_t label);

extern bool hnsw_search(HnswMetadata* meta, const coord_t *point, size_t* n_results, label_t** results);
extern bool hnsw_bind_point(HnswMetadata* meta, const coord_t *point, idx_t idx);
extern bool hnsw_begin_read(HnswMetadata* meta, idx_t idx, idx_t** indexes, coord_t** coords, label_t* label);
extern void hnsw_end_read(HnswMetadata* meta);
extern void hnsw_begin_write(HnswMetadata* meta, idx_t idx, idx_t** indexes, coord_t** coords, label_t* label);
extern void hnsw_end_write(HnswMetadata* meta);

extern void hnsw_prefetch(HnswMetadata* meta, idx_t idx);

extern dist_t hnsw_dist_func(dist_func_t dist, coord_t const* ax, coord_t const* bx, size_t dim);
extern void   hnsw_init_dist_func(void);
