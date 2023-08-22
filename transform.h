#pragma once

#include <stdint.h>
#include <random>
#include <vector>

class RandomGenerator {
    std::mt19937 mt;
  public:
	RandomGenerator(unsigned int seed) : mt(seed) {}

    /// random positive integer
    int rand_int(int max) {
		return mt() % max;
	}

	int rand_int() {
		return mt() & 0x7fffffff;
	}

	float rand_float() {
		return mt() / float(mt.max());
	}

	double rand_double() {
		return mt() / double(mt.max());
	}
};

extern void rand_perm(int* perm, size_t n, int64_t seed);

/** RAII object for a set of possibly transformed vectors (deallocated only if
 * they are indeed transformed)
 */
struct TransformedVectors {
    const float* x;
    bool own_x;
    TransformedVectors(const float* x_orig, const float* x) : x(x) {
        own_x = x_orig != x;
    }

    ~TransformedVectors() {
        if (own_x) {
            delete[] x;
        }
    }
};

/** Any transformation applied on a set of vectors */
struct VectorTransform {
    int d_in;  ///! input dimension
    int d_out; ///! output dimension

    explicit VectorTransform(int d_in = 0, int d_out = 0)
            : d_in(d_in), d_out(d_out), is_trained(true) {}

    /// set if the VectorTransform does not require training, or if
    /// training is done already
    bool is_trained;

    /** Perform training on a representative set of vectors. Does
     * nothing by default.
     *
     * @param n      nb of training vectors
     * @param x      training vecors, size n * d
     */
    virtual void train(size_t n, const float* x);

    /** apply the transformation and return the result in an allocated pointer
     * @param     n number of vectors to transform
     * @param     x input vectors, size n * d_in
     * @return    output vectors, size n * d_out
     */
    float* apply(size_t n, const float* x) const;

    /** apply the transformation and return the result in a provided matrix
     * @param     n number of vectors to transform
     * @param     x input vectors, size n * d_in
     * @param    xt output vectors, size n * d_out
     */
    virtual void apply_noalloc(size_t n, const float* x, float* xt) const = 0;

    /// reverse transformation. May not be implemented or may return
    /// approximate result
    virtual void reverse_transform(size_t n, const float* xt, float* x) const;

    // check that the two transforms are identical (to merge indexes)
    virtual void check_identical(const VectorTransform& other) const = 0;

    virtual ~VectorTransform() {}
};

/** Generic linear transformation, with bias term applied on output
 * y = A * x + b
 */
struct LinearTransform : VectorTransform {
    bool have_bias; ///! whether to use the bias term

    /// check if matrix A is orthonormal (enables reverse_transform)
    bool is_orthonormal;

    /// Transformation matrix, size d_out * d_in
    std::vector<float> A;

    /// bias vector, size d_out
    std::vector<float> b;

    /// both d_in > d_out and d_out < d_in are supported
    explicit LinearTransform(
            int d_in = 0,
            int d_out = 0,
            bool have_bias = false);

    /// same as apply, but result is pre-allocated
    void apply_noalloc(size_t n, const float* x, float* xt) const override;

    /// compute x = A^T * (x - b)
    /// is reverse transform if A has orthonormal lines
    void transform_transpose(size_t n, const float* y, float* x) const;

    /// works only if is_orthonormal
    void reverse_transform(size_t n, const float* xt, float* x) const override;

    /// compute A^T * A to set the is_orthonormal flag
    void set_is_orthonormal();

    bool verbose;
    void print_if_verbose(
            const char* name,
            const std::vector<double>& mat,
            int n,
            int d) const;

    void check_identical(const VectorTransform& other) const override;

    ~LinearTransform() override {}
};

/// Randomly rotate a set of vectors
struct RandomRotationMatrix : LinearTransform {
    /// both d_in > d_out and d_out < d_in are supported
    RandomRotationMatrix(int d_in, int d_out)
            : LinearTransform(d_in, d_out, false) {}

    /// must be called before the transform is used
    void init(int seed);

    // initializes with an arbitrary seed
    void train(size_t n, const float* x) override;

    RandomRotationMatrix() {}
};

/** Applies a principal component analysis on a set of vectors,
 *  with optionally whitening and random rotation. */
struct PCAMatrix : LinearTransform {
    /** after transformation the components are multiplied by
     * eigenvalues^eigen_power
     *
     * =0: no whitening
     * =-0.5: full whitening
     */
    float eigen_power;

    /// value added to eigenvalues to avoid division by 0 when whitening
    float epsilon;

    /// random rotation after PCA
    bool random_rotation;

    /// ratio between # training vectors and dimension
    size_t max_points_per_d;

    /// try to distribute output eigenvectors in this many bins
    int balanced_bins;

    /// Mean, size d_in
    std::vector<float> mean;

    /// eigenvalues of covariance matrix (= squared singular values)
    std::vector<float> eigenvalues;

    /// PCA matrix, size d_in * d_in
    std::vector<float> PCAMat;

    // the final matrix is computed after random rotation and/or whitening
    explicit PCAMatrix(
            int d_in = 0,
            int d_out = 0,
            float eigen_power = 0,
            bool random_rotation = false);

    /// train on n vectors. If n < d_in then the eigenvector matrix
    /// will be completed with 0s
    void train(size_t n, const float* x) override;

    /// copy pre-trained PCA matrix
    void copy_from(const PCAMatrix& other);

    /// called after mean, PCAMat and eigenvalues are computed
    void prepare_Ab();
};
