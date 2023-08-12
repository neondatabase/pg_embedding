#include <vector>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <memory>
#include <random>
#include <assert.h>

#ifdef USE_OMP
#include <omp.h>
#endif

extern "C" {
#include "embedding.h"
}

// 39 corresponds to 10000 / 256 -> to avoid warnings on PQ tests with randu10k
const size_t min_points_per_centroid = 39;
const size_t max_points_per_centroid = 256;
const int seed = 1234;
const size_t nredo = 1;
const size_t niter = 25;

class RandomGenerator {
    std::mt19937 mt;
  public:
	RandomGenerator(unsigned int seed) : mt(seed) {}

    /// random positive integer
    int rand_int(int max) {
		return mt() % max;
	}

	float rand_float() {
		return mt() / float(mt.max());
	}
};

static void
rand_perm(int* perm, size_t n, int64_t seed) {
    for (size_t i = 0; i < n; i++)
        perm[i] = i;

    RandomGenerator rng(seed);

    for (size_t i = 0; i + 1 < n; i++) {
        int i2 = i + rng.rand_int(n - i);
        std::swap(perm[i], perm[i2]);
    }
}

static size_t
subsample_training_set(
	size_t d,
	size_t k,
	size_t nx,
	const coord_t* x,
	const dist_t* weights,
	coord_t** x_out,
	dist_t** weights_out)
{
    std::vector<int> perm(nx);
    rand_perm(perm.data(), nx, seed);
    nx = k * max_points_per_centroid;
    coord_t* x_new = new coord_t[nx * d];
    *x_out = x_new;
    for (size_t i = 0; i < nx; i++) {
        memcpy(x_new + i * d, x + perm[i] * d, d * sizeof(coord_t));
    }
    if (weights) {
        dist_t* weights_new = new dist_t[nx];
        for (size_t i = 0; i < nx; i++) {
            weights_new[i] = weights[perm[i]];
        }
        *weights_out = weights_new;
    } else {
        *weights_out = nullptr;
    }
    return nx;
}

/** compute centroids as (weighted) sum of training points
 *
 * @param x            training vectors, size n * d
 * @param weights      per-training vector weight, size n (or NULL)
 * @param assign       nearest centroid for each training vector, size n
 * @param k_frozen     do not update the k_frozen first centroids
 * @param centroids    centroid vectors (output only), size k * d
 * @param hassign      histogram of assignments per centroid (size k),
 *                     should be 0 on input
 *
 */
void compute_centroids(
        size_t d,
        size_t k,
        size_t n,
        size_t k_frozen,
        const coord_t* x,
        const idx_t* assign,
        const dist_t* weights,
        coord_t* hassign,
        coord_t* centroids)
{
    k -= k_frozen;
    centroids += k_frozen * d;

    memset(centroids, 0, sizeof(coord_t) * d * k);

#ifdef USE_OMP
#pragma omp parallel
    {
        int nt = omp_get_num_threads();
        int rank = omp_get_thread_num();

        // this thread is taking care of centroids c0:c1
        size_t c0 = (k * rank) / nt;
        size_t c1 = (k * (rank + 1)) / nt;
#else
    {
#endif
		for (size_t i = 0; i < n; i++) {
			idx_t ci = assign[i];
			assert(ci < k + k_frozen);
			ci -= k_frozen;
#ifdef USE_OMP
			if (ci > c0 || ci >= c1)
				continue;
#endif
			coord_t* c = centroids + ci * d;
			const coord_t* xi = &x[i * d];
			if (weights) {
				dist_t w = weights[i];
				hassign[ci] += w;
				for (size_t j = 0; j < d; j++) {
					c[j] += xi[j] * w;
				}
			} else {
				hassign[ci] += 1.0;
				for (size_t j = 0; j < d; j++) {
					c[j] += xi[j];
				}
			}
		}
	}
#ifdef USE_OMP
#pragma omp parallel for
#endif
	for (size_t ci = 0; ci < k; ci++) {
        if (hassign[ci] == 0) {
            continue;
        }
        coord_t norm = 1 / hassign[ci];
        coord_t* c = centroids + ci * d;
        for (size_t j = 0; j < d; j++) {
            c[j] *= norm;
        }
    }
}

// a bit above machine epsilon for float16
#define EPS (1 / 1024.)

/** Handle empty clusters by splitting larger ones.
 *
 * It works by slightly changing the centroids to make 2 clusters from
 * a single one. Takes the same arguments as compute_centroids.
 *
 * @return           nb of spliting operations (larger is worse)
 */
static size_t
split_clusters(
        size_t d,
        size_t k,
        size_t n,
        size_t k_frozen,
        coord_t* hassign,
        coord_t* centroids)
{
    k -= k_frozen;
    centroids += k_frozen * d;

    /* Take care of void clusters */
    size_t nsplit = 0;
    RandomGenerator rng(seed);
    for (size_t ci = 0; ci < k; ci++) {
        if (hassign[ci] == 0) { /* need to redefine a centroid */
            size_t cj;
            for (cj = 0; 1; cj = (cj + 1) % k) {
                /* probability to pick this cluster for split */
                float p = (hassign[cj] - 1.0) / (float)(n - k);
                float r = rng.rand_float();
                if (r < p) {
                    break; /* found our cluster to be split */
                }
            }
            memcpy(centroids + ci * d,
                   centroids + cj * d,
                   sizeof(coord_t) * d);

            /* small symmetric pertubation */
            for (size_t j = 0; j < d; j++) {
                if (j % 2 == 0) {
                    centroids[ci * d + j] *= 1 + EPS;
                    centroids[cj * d + j] *= 1 - EPS;
                } else {
                    centroids[ci * d + j] *= 1 - EPS;
                    centroids[cj * d + j] *= 1 + EPS;
                }
            }

            /* assume even split of the cluster */
            hassign[ci] = hassign[cj] / 2;
            hassign[cj] -= hassign[ci];
            nsplit++;
        }
    }

    return nsplit;
}

/*
 * For each vector from training set x locate nearest centroid and store its index and distance in `assign` and `dis` arrays
 */
static void
calculate_distances(HnswMetadata* meta, coord_t const* centroids, size_t nx, coord_t const* x, dist_t* dis, idx_t* assign)
{
    size_t d = meta->pqSubdim; ///< dimension of the vectors
    idx_t k = 1 << meta->pqBits; ///< nb of centroids

#ifdef USE_OMP
#pragma omp parallel for
#endif
	for (size_t i = 0; i < nx; i++) {
		const coord_t* x_i = x + i * d;
		const coord_t* y_j = centroids;
		dist_t min_dist = HUGE_VALF;
		idx_t min_idx = 0;
		for (idx_t j = 0; j < k; j++, y_j += d) {
			dist_t ip = hnsw_dist_func(meta->dist_func, x_i, y_j, d);
			if (ip < min_dist) {
				min_dist = ip;
				min_idx = j;
			}
		}
		dis[i] = min_dist;
		assign[i] = min_idx;
	}
}

/*
 * Construct centriods for the specified training set
 */
bool
pq_train(HnswMetadata* meta, size_t slice_len, coord_t const* slice, coord_t* centroids)
{
    size_t d = meta->pqSubdim; ///< dimension of the vectors
    size_t k = 1 << meta->pqBits; ///< nb of centroids
    const coord_t* x = slice;
	size_t nx = slice_len;
	size_t sizeof_centroids = d * k * sizeof(coord_t);
    std::unique_ptr<coord_t[]> del1;
    std::unique_ptr<dist_t[]> del3;
	dist_t* weights = nullptr;

    if (nx > k * max_points_per_centroid) {
        coord_t* x_new;
        dist_t* weights_new;
        nx = subsample_training_set(d, k, nx, x, weights, &x_new, &weights_new);
        del1.reset(x_new);
        x = x_new;
        del3.reset(weights_new);
        weights = weights_new;
    } else if (nx < k * min_points_per_centroid)
		return false;

    if (nx == k) {
        // this is a corner case, just copy training set to clusters
		memcpy(centroids, slice, sizeof_centroids);
        return true;
    }

    std::unique_ptr<idx_t[]> assign(new idx_t[nx]);
    std::unique_ptr<coord_t[]> dis(new dist_t[nx]);

    // remember best iteration for redo
    coord_t best_obj = HUGE_VALF;
    std::vector<coord_t> best_centroids;
	best_centroids.resize(sizeof_centroids/sizeof(coord_t));

    // temporary buffer to decode vectors during the optimization
    for (size_t redo = 0; redo < nredo; redo++) {
        // initialize (remaining) centroids with random points from the dataset
        std::vector<int> perm(nx);

        rand_perm(perm.data(), nx, seed + 1 + redo * 15486557L);

		for (size_t i = 0; i < k; i++) {
			memcpy(&centroids[i * d], &x[perm[i] * d], sizeof(coord_t) * d);
		}

        // k-means iterations

        coord_t obj = 0;
        for (size_t i = 0; i < niter; i++) {
			calculate_distances(meta, centroids, nx, x, dis.get(), assign.get());

            // accumulate objective
            obj = 0;
            for (size_t j = 0; j < nx; j++) {
                obj += dis[j];
            }

            // update the centroids
            std::vector<coord_t> hassign(k);

            size_t k_frozen = 0;
            compute_centroids(
                    d,
                    k,
                    nx,
                    k_frozen,
                    x,
                    assign.get(),
                    weights,
                    hassign.data(),
                    centroids);

            split_clusters(d, k, nx, k_frozen, hassign.data(), centroids);
        }

        if (nredo > 1) {
            if (obj < best_obj) {
                memcpy(best_centroids.data(), centroids, sizeof_centroids);
                best_obj = obj;
            }
        }
    }
    if (nredo > 1) {
        memcpy(centroids, best_centroids.data(), sizeof_centroids);
    }
	return true;
}

