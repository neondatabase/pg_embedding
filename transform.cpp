#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <assert.h>

#include "transform.h"

extern "C" {

// this is to keep the clang syntax checker happy
#ifndef FINTEGER
#define FINTEGER int
#endif

/* declare BLAS functions, see http://www.netlib.org/clapack/cblas/ */

int sgemm_(
        const char* transa,
        const char* transb,
        FINTEGER* m,
        FINTEGER* n,
        FINTEGER* k,
        const float* alpha,
        const float* a,
        FINTEGER* lda,
        const float* b,
        FINTEGER* ldb,
        float* beta,
        float* c,
        FINTEGER* ldc);

int sgeqrf_(
        FINTEGER* m,
        FINTEGER* n,
        float* a,
        FINTEGER* lda,
        float* tau,
        float* work,
        FINTEGER* lwork,
        FINTEGER* info);

int sorgqr_(
        FINTEGER* m,
        FINTEGER* n,
        FINTEGER* k,
        float* a,
        FINTEGER* lda,
        float* tau,
        float* work,
        FINTEGER* lwork,
        FINTEGER* info);

int dgemm_(
        const char* transa,
        const char* transb,
        FINTEGER* m,
        FINTEGER* n,
        FINTEGER* k,
        const double* alpha,
        const double* a,
        FINTEGER* lda,
        const double* b,
        FINTEGER* ldb,
        double* beta,
        double* c,
        FINTEGER* ldc);

int ssyrk_(
        const char* uplo,
        const char* trans,
        FINTEGER* n,
        FINTEGER* k,
        float* alpha,
        float* a,
        FINTEGER* lda,
        float* beta,
        float* c,
        FINTEGER* ldc);

/* Lapack functions from http://www.netlib.org/clapack/old/single/ */

int ssyev_(
        const char* jobz,
        const char* uplo,
        FINTEGER* n,
        float* a,
        FINTEGER* lda,
        float* w,
        float* work,
        FINTEGER* lwork,
        FINTEGER* info);

int dsyev_(
        const char* jobz,
        const char* uplo,
        FINTEGER* n,
        double* a,
        FINTEGER* lda,
        double* w,
        double* work,
        FINTEGER* lwork,
        FINTEGER* info);

int sgesvd_(
        const char* jobu,
        const char* jobvt,
        FINTEGER* m,
        FINTEGER* n,
        float* a,
        FINTEGER* lda,
        float* s,
        float* u,
        FINTEGER* ldu,
        float* vt,
        FINTEGER* ldvt,
        float* work,
        FINTEGER* lwork,
        FINTEGER* info);

int dgesvd_(
        const char* jobu,
        const char* jobvt,
        FINTEGER* m,
        FINTEGER* n,
        double* a,
        FINTEGER* lda,
        double* s,
        double* u,
        FINTEGER* ldu,
        double* vt,
        FINTEGER* ldvt,
        double* work,
        FINTEGER* lwork,
        FINTEGER* info);
}

static float
fvec_norm_L2sqr(const float* x, size_t d) {
    // the double in the _ref is suspected to be a typo. Some of the manual
    // implementations this replaces used float.
    float res = 0;
    for (size_t i = 0; i != d; ++i) {
        res += x[i] * x[i];
    }

    return res;
}

static void
fvec_renorm_L2(size_t d, size_t nx, float* __restrict x) {
#pragma omp parallel for schedule(guided)
    for (size_t i = 0; i < nx; i++) {
        float* __restrict xi = x + i * d;

        float nr = fvec_norm_L2sqr(xi, d);

        if (nr > 0) {
            size_t j;
            const float inv_nr = 1.0 / sqrtf(nr);
            for (j = 0; j < d; j++)
                xi[j] *= inv_nr;
        }
    }
}

const float* fvecs_maybe_subsample(
        size_t d,
        size_t* n,
        size_t nmax,
        const float* x,
        bool verbose,
        int64_t seed = 1234) {
    if (*n <= nmax)
        return x; // nothing to do

    size_t n2 = nmax;
    if (verbose) {
        printf("  Input training set too big (max size is %zd), sampling "
               "%zd / %zd vectors\n",
               nmax,
               n2,
               *n);
    }
    std::vector<int> subset(*n);
    rand_perm(subset.data(), *n, seed);
    float* x_subset = new float[n2 * d];
    for (size_t i = 0; i < n2; i++)
        memcpy(&x_subset[i * d], &x[subset[i] * size_t(d)], sizeof(x[0]) * d);
    *n = n2;
    return x_subset;
}

static void
float_randn(float* x, size_t n, int seed) {
    // only try to parallelize on large enough arrays
    const size_t nblock = n < 1024 ? 1 : 1024;

    RandomGenerator rng0(seed);
    int a0 = rng0.rand_int(), b0 = rng0.rand_int();

#pragma omp parallel for
    for (size_t j = 0; j < nblock; j++) {
        RandomGenerator rng(a0 + j * b0);

        double a = 0, b = 0, s = 0;
        int state = 0; /* generate two number per "do-while" loop */

        const size_t istart = j * n / nblock;
        const size_t iend = (j + 1) * n / nblock;

        for (size_t i = istart; i < iend; i++) {
            /* Marsaglia's method (see Knuth) */
            if (state == 0) {
                do {
                    a = 2.0 * rng.rand_double() - 1;
                    b = 2.0 * rng.rand_double() - 1;
                    s = a * a + b * b;
                } while (s >= 1.0);
                x[i] = a * sqrt(-2.0 * log(s) / s);
            } else
                x[i] = b * sqrt(-2.0 * log(s) / s);
            state = 1 - state;
        }
    }
}


void matrix_qr(int m, int n, float* a) {
    assert(m >= n);
    FINTEGER mi = m, ni = n, ki = mi < ni ? mi : ni;
    std::vector<float> tau(ki);
    FINTEGER lwork = -1, info;
    float work_size;

    sgeqrf_(&mi, &ni, a, &mi, tau.data(), &work_size, &lwork, &info);
    lwork = (FINTEGER)work_size;
    std::vector<float> work(lwork);

    sgeqrf_(&mi, &ni, a, &mi, tau.data(), work.data(), &lwork, &info);

    sorgqr_(&mi, &ni, &ki, a, &mi, tau.data(), work.data(), &lwork, &info);
}

void
rand_perm(int* perm, size_t n, int64_t seed) {
    for (size_t i = 0; i < n; i++)
        perm[i] = i;

    RandomGenerator rng(seed);

    for (size_t i = 0; i + 1 < n; i++) {
        size_t i2 = i + rng.rand_int(n - i);
        std::swap(perm[i], perm[i2]);
    }
}

/*********************************************
 * VectorTransform
 *********************************************/

float* VectorTransform::apply(size_t n, const float* x) const {
    float* xt = new float[n * d_out];
    apply_noalloc(n, x, xt);
    return xt;
}

void VectorTransform::train(size_t, const float*) {
    // does nothing by default
}

void VectorTransform::reverse_transform(size_t, const float*, float*) const {
	assert(false);
}

void VectorTransform::check_identical(const VectorTransform& other) const {
	assert(false);
}

/*********************************************
 * LinearTransform
 *********************************************/

/// both d_in > d_out and d_out < d_in are supported
LinearTransform::LinearTransform(size_t d_in, size_t d_out, bool have_bias)
        : VectorTransform(d_in, d_out),
          have_bias(have_bias),
          is_orthonormal(false),
          verbose(false) {
    is_trained = false; // will be trained when A and b are initialized
}

void LinearTransform::apply_noalloc(size_t n, const float* x, float* xt) const {
    assert(is_trained);

    float c_factor;
    if (have_bias) {
        assert(b.size() == (size_t)d_out);
        float* xi = xt;
        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < d_out; j++)
                *xi++ = b[j];
        c_factor = 1.0;
    } else {
        c_factor = 0.0;
    }

    assert(A.size() == (size_t)d_out * d_in);

    float one = 1;
    FINTEGER nbiti = d_out, ni = n, di = d_in;
    sgemm_("Transposed",
           "Not transposed",
           &nbiti,
           &ni,
           &di,
           &one,
           A.data(),
           &di,
           x,
           &di,
           &c_factor,
           xt,
           &nbiti);
}

void LinearTransform::transform_transpose(size_t n, const float* y, float* x)
        const {
    if (have_bias) { // allocate buffer to store bias-corrected data
        float* y_new = new float[n * d_out];
        const float* yr = y;
        float* yw = y_new;
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < d_out; j++) {
                *yw++ = *yr++ - b[j];
            }
        }
        y = y_new;
    }

    {
        FINTEGER dii = d_in, doi = d_out, ni = n;
        float one = 1.0, zero = 0.0;
        sgemm_("Not",
               "Not",
               &dii,
               &ni,
               &doi,
               &one,
               A.data(),
               &dii,
               y,
               &doi,
               &zero,
               x,
               &dii);
    }

    if (have_bias)
        delete[] y;
}

void LinearTransform::set_is_orthonormal() {
    if (d_out > d_in) {
        // not clear what we should do in this case
        is_orthonormal = false;
        return;
    }
    if (d_out == 0) { // borderline case, unnormalized matrix
        is_orthonormal = true;
        return;
    }

    double eps = 4e-5;
    assert(A.size() >= (size_t)d_out * d_in);
    {
        std::vector<float> ATA(d_out * d_out);
        FINTEGER dii = d_in, doi = d_out;
        float one = 1.0, zero = 0.0;

        sgemm_("Transposed",
               "Not",
               &doi,
               &doi,
               &dii,
               &one,
               A.data(),
               &dii,
               A.data(),
               &dii,
               &zero,
               ATA.data(),
               &doi);

        is_orthonormal = true;
        for (size_t i = 0; i < d_out; i++) {
            for (size_t j = 0; j < d_out; j++) {
                float v = ATA[i + j * d_out];
                if (i == j)
                    v -= 1;
                if (fabs(v) > eps) {
                    is_orthonormal = false;
                }
            }
        }
    }
}

void LinearTransform::reverse_transform(size_t n, const float* xt, float* x)
        const {
    if (is_orthonormal) {
        transform_transpose(n, xt, x);
    } else {
        assert(false /* reverse transform not implemented for non-orthonormal matrices */);
    }
}

void LinearTransform::print_if_verbose(
        const char* name,
        const std::vector<double>& mat,
        size_t n,
        size_t d) const {
    if (!verbose)
        return;
    printf("matrix %s: %zd*%zd [\n", name, n, d);
    assert(mat.size() >= n * d);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < d; j++) {
            printf("%10.5g ", mat[i * d + j]);
        }
        printf("\n");
    }
    printf("]\n");
}

void LinearTransform::check_identical(const VectorTransform& other_in) const {
    VectorTransform::check_identical(other_in);
    auto other = dynamic_cast<const LinearTransform*>(&other_in);
    assert(other);
    assert(other->A == A && other->b == b);
}

/*********************************************
 * RandomRotationMatrix
 *********************************************/

void RandomRotationMatrix::init(int seed) {
    if (d_out <= d_in) {
        A.resize(d_out * d_in);
        float* q = A.data();
        float_randn(q, d_out * d_in, seed);
        matrix_qr(d_in, d_out, q);
    } else {
        // use tight-frame transformation
        A.resize(d_out * d_out);
        float* q = A.data();
        float_randn(q, d_out * d_out, seed);
        matrix_qr(d_out, d_out, q);
        // remove columns
        size_t i, j;
        for (i = 0; i < d_out; i++) {
            for (j = 0; j < d_in; j++) {
                q[i * d_in + j] = q[i * d_out + j];
            }
        }
        A.resize(d_in * d_out);
    }
    is_orthonormal = true;
    is_trained = true;
}

void RandomRotationMatrix::train(size_t /*n*/, const float* /*x*/) {
    // initialize with some arbitrary seed
    init(12345);
}

/*********************************************
 * PCAMatrix
 *********************************************/

PCAMatrix::PCAMatrix(
        size_t d_in,
        size_t d_out,
        float eigen_power,
        bool random_rotation)
        : LinearTransform(d_in, d_out, true),
          eigen_power(eigen_power),
          random_rotation(random_rotation) {
    is_trained = false;
    max_points_per_d = 1000;
    balanced_bins = 0;
    epsilon = 0;
}

/// Compute the eigenvalue decomposition of symmetric matrix cov,
/// dimensions d_in-by-d_in. Output eigenvectors in cov.

void eig(size_t d_in, double* cov, double* eigenvalues, int verbose) {
    { // compute eigenvalues and vectors
        FINTEGER info = 0, lwork = -1, di = d_in;
        double workq;

        dsyev_("Vectors as well",
               "Upper",
               &di,
               cov,
               &di,
               eigenvalues,
               &workq,
               &lwork,
               &info);
        lwork = FINTEGER(workq);
        double* work = new double[lwork];

        dsyev_("Vectors as well",
               "Upper",
               &di,
               cov,
               &di,
               eigenvalues,
               work,
               &lwork,
               &info);

        delete[] work;

        if (info != 0) {
            fprintf(stderr,
                    "WARN ssyev info returns %d, "
                    "a very bad PCA matrix is learnt\n",
                    int(info));
            // do not throw exception, as the matrix could still be useful
        }

        if (verbose && d_in <= 10) {
            printf("info=%ld new eigvals=[", long(info));
            for (size_t j = 0; j < d_in; j++)
                printf("%g ", eigenvalues[j]);
            printf("]\n");

            double* ci = cov;
            printf("eigenvecs=\n");
            for (size_t i = 0; i < d_in; i++) {
                for (size_t j = 0; j < d_in; j++)
                    printf("%10.4g ", *ci++);
                printf("\n");
            }
        }
    }

    // revert order of eigenvectors & values

    for (size_t i = 0; i < d_in / 2; i++) {
        std::swap(eigenvalues[i], eigenvalues[d_in - 1 - i]);
        double* v1 = cov + i * d_in;
        double* v2 = cov + (d_in - 1 - i) * d_in;
        for (size_t j = 0; j < d_in; j++)
            std::swap(v1[j], v2[j]);
    }
}

void PCAMatrix::train(size_t n, const float* x_in) {
    const float* x = fvecs_maybe_subsample(
            d_in, (size_t*)&n, max_points_per_d * d_in, x_in, verbose);
    TransformedVectors tv(x_in, x);

    // compute mean
    mean.clear();
    mean.resize(d_in, 0.0);
    if (have_bias) { // we may want to skip the bias
        const float* xi = x;
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < d_in; j++)
                mean[j] += *xi++;
        }
        for (size_t j = 0; j < d_in; j++)
            mean[j] /= n;
    }
    if (verbose) {
        printf("mean=[");
        for (size_t j = 0; j < d_in; j++)
            printf("%g ", mean[j]);
        printf("]\n");
    }

    if (n >= d_in) {
        // compute covariance matrix, store it in PCA matrix
        PCAMat.resize(d_in * d_in);
        float* cov = PCAMat.data();
        { // initialize with  mean * mean^T term
            float* ci = cov;
            for (size_t i = 0; i < d_in; i++) {
                for (size_t j = 0; j < d_in; j++)
                    *ci++ = -n * mean[i] * mean[j];
            }
        }
        {
            FINTEGER di = d_in, ni = n;
            float one = 1.0;
            ssyrk_("Up",
                   "Non transposed",
                   &di,
                   &ni,
                   &one,
                   (float*)x,
                   &di,
                   &one,
                   cov,
                   &di);
        }
        if (verbose && d_in <= 10) {
            float* ci = cov;
            printf("cov=\n");
            for (size_t i = 0; i < d_in; i++) {
                for (size_t j = 0; j < d_in; j++)
                    printf("%10g ", *ci++);
                printf("\n");
            }
        }

        std::vector<double> covd(d_in * d_in);
        for (size_t i = 0; i < (size_t)d_in * d_in; i++)
            covd[i] = cov[i];

        std::vector<double> eigenvaluesd(d_in);

        eig(d_in, covd.data(), eigenvaluesd.data(), verbose);

        for (size_t i = 0; i < (size_t)d_in * d_in; i++)
            PCAMat[i] = covd[i];
        eigenvalues.resize(d_in);

        for (size_t i = 0; i < d_in; i++)
            eigenvalues[i] = eigenvaluesd[i];

    } else {
        std::vector<float> xc(n * d_in);

        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < d_in; j++)
                xc[i * d_in + j] = x[i * d_in + j] - mean[j];

        // compute Gram matrix
        std::vector<float> gram(n * n);
        {
            FINTEGER di = d_in, ni = n;
            float one = 1.0, zero = 0.0;
            ssyrk_("Up",
                   "Transposed",
                   &ni,
                   &di,
                   &one,
                   xc.data(),
                   &di,
                   &zero,
                   gram.data(),
                   &ni);
        }

        if (verbose && d_in <= 10) {
            float* ci = gram.data();
            printf("gram=\n");
            for (size_t i = 0; i < n; i++) {
                for (size_t j = 0; j < n; j++)
                    printf("%10g ", *ci++);
                printf("\n");
            }
        }

        std::vector<double> gramd(n * n);
        for (size_t i = 0; i < (size_t)n * n; i++)
            gramd[i] = gram[i];

        std::vector<double> eigenvaluesd(n);

        // eig will fill in only the n first eigenvals

        eig(n, gramd.data(), eigenvaluesd.data(), verbose);

        PCAMat.resize(d_in * n);

        for (size_t i = 0; i < (size_t)n * n; i++)
            gram[i] = gramd[i];

        eigenvalues.resize(d_in);
        // fill in only the n first ones
        for (size_t i = 0; i < n; i++)
            eigenvalues[i] = eigenvaluesd[i];

        { // compute PCAMat = x' * v
            FINTEGER di = d_in, ni = n;
            float one = 1.0;

            sgemm_("Non",
                   "Non Trans",
                   &di,
                   &ni,
                   &ni,
                   &one,
                   xc.data(),
                   &di,
                   gram.data(),
                   &ni,
                   &one,
                   PCAMat.data(),
                   &di);
        }

        if (verbose && d_in <= 10) {
            float* ci = PCAMat.data();
            printf("PCAMat=\n");
            for (size_t i = 0; i < n; i++) {
                for (size_t j = 0; j < d_in; j++)
                    printf("%10g ", *ci++);
                printf("\n");
            }
        }
        fvec_renorm_L2(d_in, n, PCAMat.data());
    }

    prepare_Ab();
    is_trained = true;
}

void PCAMatrix::copy_from(const PCAMatrix& other) {
    assert(other.is_trained);
    mean = other.mean;
    eigenvalues = other.eigenvalues;
    PCAMat = other.PCAMat;
    prepare_Ab();
    is_trained = true;
}

void PCAMatrix::prepare_Ab() {
    assert((size_t)d_out * d_in <= PCAMat.size());
    if (!random_rotation) {
        A = PCAMat;
        A.resize(d_out * d_in); // strip off useless dimensions

        // first scale the components
        if (eigen_power != 0) {
            float* ai = A.data();
            for (size_t i = 0; i < d_out; i++) {
                float factor = pow(eigenvalues[i] + epsilon, eigen_power);
                for (size_t j = 0; j < d_in; j++)
                    *ai++ *= factor;
            }
        }

        if (balanced_bins != 0) {
            assert(d_out % balanced_bins == 0);
            size_t dsub = d_out / balanced_bins;
            std::vector<float> Ain;
            std::swap(A, Ain);
            A.resize(d_out * d_in);

            std::vector<float> accu(balanced_bins);
            std::vector<unsigned> counter(balanced_bins);

            // greedy assignment
            for (size_t i = 0; i < d_out; i++) {
                // find best bin
                size_t best_j = 0;
                float min_w = 1e30;
                for (size_t j = 0; j < balanced_bins; j++) {
                    if (counter[j] < dsub && accu[j] < min_w) {
                        min_w = accu[j];
                        best_j = j;
                    }
                }
                int row_dst = best_j * dsub + counter[best_j];
                accu[best_j] += eigenvalues[i];
                counter[best_j]++;
                memcpy(&A[row_dst * d_in], &Ain[i * d_in], d_in * sizeof(A[0]));
            }

            if (verbose) {
                printf("  bin accu=[");
                for (size_t i = 0; i < balanced_bins; i++)
                    printf("%g ", accu[i]);
                printf("]\n");
            }
        }

    } else {
        assert(balanced_bins == 0);
        RandomRotationMatrix rr(d_out, d_out);

        rr.init(5);

        // apply scaling on the rotation matrix (right multiplication)
        if (eigen_power != 0) {
            for (size_t i = 0; i < d_out; i++) {
                float factor = pow(eigenvalues[i], eigen_power);
                for (size_t j = 0; j < d_out; j++)
                    rr.A[j * d_out + i] *= factor;
            }
        }

        A.resize(d_in * d_out);
        {
            FINTEGER dii = d_in, doo = d_out;
            float one = 1.0, zero = 0.0;

            sgemm_("Not",
                   "Not",
                   &dii,
                   &doo,
                   &doo,
                   &one,
                   PCAMat.data(),
                   &dii,
                   rr.A.data(),
                   &doo,
                   &zero,
                   A.data(),
                   &dii);
        }
    }

    b.clear();
    b.resize(d_out);

    for (size_t i = 0; i < d_out; i++) {
        float accu = 0;
        for (size_t j = 0; j < d_in; j++)
            accu -= mean[j] * A[j + i * d_in];
        b[i] = accu;
    }

    is_orthonormal = eigen_power == 0;
}

