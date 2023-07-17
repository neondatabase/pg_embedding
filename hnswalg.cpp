#include "hnswalg.h"
#include <cmath>

#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PREFETCH(addr,hint) __builtin_prefetch(addr, 0, hint)
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#define PREFETCH(addr,hint)
#endif

template <HNSW_DistType Dist>
HierarchicalNSW_Impl<Dist>::HierarchicalNSW_Impl(
    size_t dim_,
    size_t maxelements_,
    size_t M_,
    size_t maxM_,
    size_t efConstruction_)
{
    dim = dim_;
    data_size = dim * sizeof(coord_t);

    efConstruction = efConstruction_;

    maxelements = maxelements_;
    M = M_;
    maxM = maxM_;
    size_links_level0 = (maxM + 1) * sizeof(idx_t);
    size_data_per_element = size_links_level0 + data_size  + sizeof(label_t);
    offset_data = size_links_level0;
	offset_label = offset_data + data_size;

    enterpoint_node = 0;
    cur_element_count = 0;

#ifdef __x86_64__
    use_avx2 = __builtin_cpu_supports("avx2");
#endif
}

template <HNSW_DistType Dist>
std::priority_queue<std::pair<dist_t, idx_t>> HierarchicalNSW_Impl<Dist>::searchBaseLayer(
    const coord_t *point,
    size_t ef)
{
	std::vector<uint32_t> visited;
	visited.resize((cur_element_count + 31) >> 5);

    std::priority_queue<std::pair<dist_t, idx_t >> topResults;
    std::priority_queue<std::pair<dist_t, idx_t >> candidateSet;

    dist_t dist = fstdistfunc(point, getDataByInternalId(enterpoint_node));

    topResults.emplace(dist, enterpoint_node);
    candidateSet.emplace(-dist, enterpoint_node);
    visited[enterpoint_node >> 5] = 1 << (enterpoint_node & 31);
    dist_t lowerBound = dist;

    while (!candidateSet.empty())
    {
        std::pair<dist_t, idx_t> curr_el_pair = candidateSet.top();
        if (-curr_el_pair.first > lowerBound)
            break;

        candidateSet.pop();
        idx_t curNodeNum = curr_el_pair.second;

        idx_t* data = get_linklist0(curNodeNum);
        size_t size = *data++;

        PREFETCH(getDataByInternalId(*data), 0);

        for (size_t j = 0; j < size; ++j) {
            size_t tnum = *(data + j);

            PREFETCH(getDataByInternalId(*(data + j + 1)), 0);

            if (!(visited[tnum >> 5] & (1 << (tnum & 31)))) {
				visited[tnum >> 5] |= 1 << (tnum & 31);

                dist = fstdistfunc(point, getDataByInternalId(tnum));

                if (topResults.top().first > dist || topResults.size() < ef) {
                    candidateSet.emplace(-dist, tnum);

                    PREFETCH(get_linklist0(candidateSet.top().second), 0);
                    topResults.emplace(dist, tnum);

                    if (topResults.size() > ef)
                        topResults.pop();

                    lowerBound = topResults.top().first;
                }
            }
        }
    }
    return topResults;
}

template <HNSW_DistType Dist>
void HierarchicalNSW_Impl<Dist>::getNeighborsByHeuristic(
    std::priority_queue<std::pair<dist_t, idx_t>> &topResults,
    size_t NN)
{
    if (topResults.size() < NN)
        return;

    std::priority_queue<std::pair<dist_t, idx_t>> resultSet;
    std::vector<std::pair<dist_t, idx_t>> returnlist;

    while (topResults.size() > 0) {
        resultSet.emplace(-topResults.top().first, topResults.top().second);
        topResults.pop();
    }

    while (resultSet.size()) {
        if (returnlist.size() >= NN)
            break;
        std::pair<dist_t, idx_t> curen = resultSet.top();
        dist_t dist_to_query = -curen.first;
        resultSet.pop();
        bool good = true;
        for (std::pair<dist_t, idx_t> curen2 : returnlist) {
            dist_t curdist = fstdistfunc(getDataByInternalId(curen2.second),
                                         getDataByInternalId(curen.second));
            if (curdist < dist_to_query) {
                good = false;
                break;
            }
        }
        if (good) returnlist.push_back(curen);
    }
    for (std::pair<dist_t, idx_t> elem : returnlist)
        topResults.emplace(-elem.first, elem.second);
}

template <HNSW_DistType Dist>
void HierarchicalNSW_Impl<Dist>::mutuallyConnectNewElement(
    const coord_t *point,
    idx_t cur_c,
    std::priority_queue<std::pair<dist_t, idx_t>> topResults)
{
    getNeighborsByHeuristic(topResults, M);

    std::vector<idx_t> res;
    res.reserve(M);
    while (topResults.size() > 0) {
        res.push_back(topResults.top().second);
        topResults.pop();
    }
    {
        idx_t* data = get_linklist0(cur_c);
        if (*data)
            throw std::runtime_error("Should be blank");

        *data++ = res.size();

        for (size_t idx = 0; idx < res.size(); idx++) {
            if (data[idx])
                throw std::runtime_error("Should be blank");
            data[idx] = res[idx];
        }
    }
    for (size_t idx = 0; idx < res.size(); idx++) {
        if (res[idx] == cur_c)
            throw std::runtime_error("Connection to the same element");

        size_t resMmax = maxM;
        idx_t *ll_other = get_linklist0(res[idx]);
        idx_t sz_link_list_other = *ll_other;

        if (sz_link_list_other > resMmax || sz_link_list_other < 0)
            throw std::runtime_error("Bad sz_link_list_other");

        if (sz_link_list_other < resMmax) {
            idx_t *data = ll_other + 1;
            data[sz_link_list_other] = cur_c;
            *ll_other = sz_link_list_other + 1;
        } else {
            // finding the "weakest" element to replace it with the new one
            idx_t *data = ll_other + 1;
            dist_t d_max = fstdistfunc(getDataByInternalId(cur_c), getDataByInternalId(res[idx]));
            // Heuristic:
            std::priority_queue<std::pair<dist_t, idx_t>> candidates;
            candidates.emplace(d_max, cur_c);

            for (size_t j = 0; j < sz_link_list_other; j++)
                candidates.emplace(fstdistfunc(getDataByInternalId(data[j]), getDataByInternalId(res[idx])), data[j]);

            getNeighborsByHeuristic(candidates, resMmax);

            size_t indx = 0;
            while (!candidates.empty()) {
                data[indx] = candidates.top().second;
                candidates.pop();
                indx++;
            }
            *ll_other = indx;
        }
    }
}

template <HNSW_DistType Dist>
void HierarchicalNSW_Impl<Dist>::addPoint(const coord_t *point, label_t label)
{
    if (cur_element_count >= maxelements) {
        throw std::runtime_error("The number of elements exceeds the specified limit");
    }
    idx_t cur_c = cur_element_count++;
    memset((char *) get_linklist0(cur_c), 0, size_data_per_element);
    memcpy(getDataByInternalId(cur_c), point, data_size);
    memcpy(getExternalLabel(cur_c), &label, sizeof label);

    // Do nothing for the first element
    if (cur_c != 0) {
        std::priority_queue <std::pair<dist_t, idx_t>> topResults = searchBaseLayer(point, efConstruction);
        mutuallyConnectNewElement(point, cur_c, topResults);
    }
}

template <HNSW_DistType Dist>
std::priority_queue<std::pair<dist_t, label_t>> HierarchicalNSW_Impl<Dist>::searchKnn(const coord_t *query, size_t k)
{
	std::priority_queue<std::pair<dist_t, label_t>> topResults;
	auto topCandidates = searchBaseLayer(query, k);
    while (topCandidates.size() > k) {
        topCandidates.pop();
	}
	while (!topCandidates.empty()) {
		std::pair<dist_t, idx_t> rez = topCandidates.top();
		label_t label;
		memcpy(&label, getExternalLabel(rez.second), sizeof(label));
		topResults.push(std::pair<dist_t, label_t>(rez.first, label));
		topCandidates.pop();
	}

    return topResults;
}

dist_t dist_l2_scalar(const coord_t *x, const coord_t *y, size_t n)
{
    dist_t 	distance = 0.0;

    #pragma clang loop vectorize(enable)
    for (size_t i = 0; i < n; i++)
    {
        dist_t diff = x[i] - y[i];
        distance += diff * diff;
    }
    return distance;

}

dist_t dist_manhattan_scalar(const coord_t *x, const coord_t *y, size_t n)
{
    dist_t 	distance = 0.0;

    #pragma clang loop vectorize(enable)
    for (size_t i = 0; i < n; i++)
    {
        dist_t diff = std::abs(x[i] - y[i]);
        distance += diff;
    }
    return distance;
}

#ifdef __x86_64__
#include <immintrin.h>

__attribute__((target("avx2")))
dist_t dist_manhattan_avx2(const coord_t *x, const coord_t *y, size_t n)
{
    const size_t elts_per_vector = sizeof(__m256) / sizeof(coord_t);
    float partial_result[elts_per_vector];
    __m256 accumulator = _mm256_setzero_ps();
    __m256 vecMask = _mm256_set1_ps(-0.0f);
    for(size_t i = 0; i < n; i += elts_per_vector)
    {
        __m256 vecX = _mm256_loadu_ps(x + i);
        __m256 vecY = _mm256_loadu_ps(y + i);
        __m256 vecSub = _mm256_sub_ps(vecX, vecY);
        __m256 vecAbs = _mm256_andnot_ps(vecMask, vecSub);
        accumulator = _mm256_add_ps(accumulator, vecAbs);
    }
    _mm256_storeu_ps(partial_result, accumulator);
    float res1 = partial_result[0] + partial_result[4];
    float res2 = partial_result[1] + partial_result[5];
    float res3 = partial_result[2] + partial_result[6];
    float res4 = partial_result[3] + partial_result[7];
    res1 += res3;
    res2 += res4;
    res1 += res2;

    size_t tail_size = n % elts_per_vector;
    size_t tail = n - tail_size;
    for(int i = 0; i < tail_size; i++)
    {
        dist_t dist = x[tail + i] - y[tail + i];
        res1 += dist;
    }
    return res1;
}

__attribute__((target("avx2,fma")))
dist_t dist_l2_avx2(const coord_t *x, const coord_t *y, size_t n)
{
    constexpr size_t elts_per_vector = sizeof(__m256) / sizeof(coord_t);
    constexpr size_t unroll = 2;
    constexpr size_t elts_per_loop = elts_per_vector * unroll;
    float partial_result[elts_per_vector];
    __m256 accum1 = _mm256_setzero_ps();
    __m256 accum2 = _mm256_setzero_ps();
    for(size_t i = 0; i < n; i += elts_per_loop)
    {
        __m256 vecX1 = _mm256_loadu_ps(x + i);
        __m256 vecY1 = _mm256_loadu_ps(y + i);
        __m256 vecSub1 = _mm256_sub_ps(vecX1, vecY1);
        accum1 = _mm256_fmadd_ps(vecSub1, vecSub1, accum2);

        __m256 vecX2 = _mm256_loadu_ps(x + i + elts_per_vector);
        __m256 vecY2 = _mm256_loadu_ps(y + i + elts_per_vector);
        __m256 vecSub2 = _mm256_sub_ps(vecX2, vecY2);
        accum2 = _mm256_fmadd_ps(vecSub2, vecSub2, accum2);
    }
    accum1 = _mm256_add_ps(accum1, accum2);
    _mm256_storeu_ps(partial_result, accum1);
    float res1 = partial_result[0] + partial_result[4];
    float res2 = partial_result[1] + partial_result[5];
    float res3 = partial_result[2] + partial_result[6];
    float res4 = partial_result[3] + partial_result[7];
    res1 += res3;
    res2 += res4;
    res1 += res2;

    size_t tail_size = n % elts_per_loop;
    size_t tail = n - tail_size;
    for(int i = 0; i < tail_size; i++)
    {
        dist_t dist = x[tail + i] - y[tail + i];
        res1 += dist;
    } 
    return res1;
}

dist_t dist_l2_sse(const coord_t *x, const coord_t *y, size_t n)
{
    const size_t elts_per_vector = sizeof(__m128) / sizeof(coord_t);
    float partial_result[elts_per_vector];
    __m128 accumulator = _mm_setzero_ps();
    for(size_t i = 0; i < n; i += elts_per_vector)
    {
        __m128 vecX = _mm_loadu_ps(x + i);
        __m128 vecY = _mm_loadu_ps(y + i);
        __m128 vecSub = _mm_sub_ps(vecX, vecY);
        __m128 vecSquare = _mm_mul_ps(vecSub, vecSub);
        accumulator = _mm_add_ps(vecSquare, accumulator);
    }
    _mm_storeu_ps(partial_result, accumulator);
    float res1 = partial_result[0] + partial_result[2];
    float res2 = partial_result[1] + partial_result[3];
    res1 += res2;

    size_t tail_size = n % elts_per_vector;
    size_t tail = n - tail_size;
    for(int i = 0; i < tail_size; i++)
    {
        dist_t diff = x[tail + i] - y[tail + i];
        dist_t square = diff * diff;
        res1 += square;
    }
    return res1;
}
#endif

template <HNSW_DistType Dist>
dist_t dist(bool use_avx2, const coord_t *x, const coord_t *y, size_t n);

template <>
dist_t dist<HNSW_Dist_L2>(bool use_avx2, const coord_t *x, const coord_t *y, size_t n)
{
#ifndef __x86_64__
    return dist_l2_scalar(x, y, n);
#else
    if(use_avx2)
        return dist_l2_avx2(x, y, n);

    return dist_l2_sse(x, y, n);
#endif
}

template <>
dist_t dist<HNSW_Dist_Manhattan>(bool use_avx2, const coord_t *x, const coord_t *y, size_t n)
{
#ifndef __x86_64__
    return dist_manhattan_scalar(x, y, n);
#else
    if(use_avx2)
        return dist_manhattan_avx2(x, y, n);
    return dist_manhattan_scalar(x, y, n);
#endif
}

template <HNSW_DistType Dist>
dist_t HierarchicalNSW_Impl<Dist>::fstdistfunc(const coord_t *x, const coord_t *y)
{
#ifndef __x86_64__
    return dist<Dist>(false, x, y, dim);
#else
    return dist<Dist>(use_avx2, x, y, dim);
#endif
}

bool hnsw_search(HierarchicalNSW* hnsw, const coord_t *point, size_t efSearch, size_t* n_results, label_t** results)
{
	try
	{
                std::priority_queue<std::pair<dist_t, label_t>> result;
                switch(hnsw->dist_type)
                {
                case HNSW_Dist_L2:
                    result = hnsw->impl_l2.searchKnn(point, efSearch);
                    break;
                case HNSW_Dist_Manhattan:
                    result = hnsw->impl_manhattan.searchKnn(point, efSearch);
                    break;
                default:
                    return false;
                }
		size_t nResults = result.size();
		*results = (label_t*)malloc(nResults*sizeof(label_t));
		for (size_t i = nResults; i-- != 0;)
		{
			(*results)[i] = result.top().second;
			result.pop();
		}
		*n_results = nResults;
		return true;
	}
	catch (std::exception& x)
	{
		return false;
	}
}

bool hnsw_add_point(HierarchicalNSW* hnsw, const coord_t *point, label_t label)
{
	try
	{
            switch(hnsw->dist_type)
            {
            case HNSW_Dist_L2:
                hnsw->impl_l2.addPoint(point, label);
                break;
            case HNSW_Dist_Manhattan:
                hnsw->impl_manhattan.addPoint(point, label);
                break;
            default:
                fprintf(stderr, "Invalid distfunc\n");
                return false;
            }
            return true;
	}
	catch (std::exception& x)
	{
		fprintf(stderr, "Catch %s\n", x.what());
		return false;
	}
}

void hnsw_init(HierarchicalNSW* hnsw, size_t dims, size_t maxelements, size_t M, size_t maxM, size_t efConstruction, HNSW_DistType dist)
{
    hnsw->dist_type = dist;
    switch(hnsw->dist_type)
    {
    case HNSW_Dist_L2:
	new ((void*)&hnsw->impl_l2) HierarchicalNSW_Impl<HNSW_Dist_L2>(
            dims,
            maxelements,
            M,
            maxM,
            efConstruction);
        break;
    case HNSW_Dist_Manhattan:
	new ((void*)&hnsw->impl_manhattan) HierarchicalNSW_Impl<HNSW_Dist_Manhattan>(
            dims,
            maxelements,
            M,
            maxM,
            efConstruction);
        break;
    default:
        break;
    }
}


int hnsw_dimensions(HierarchicalNSW* hnsw)
{
    switch(hnsw->dist_type)
    {
    case HNSW_Dist_L2:
        return (int)hnsw->impl_l2.dim;
    case HNSW_Dist_Manhattan:
        return (int)hnsw->impl_manhattan.dim;
    default:
        return -1;
    }
}

size_t hnsw_count(HierarchicalNSW* hnsw)
{
    switch(hnsw->dist_type)
    {
    case HNSW_Dist_L2:
        return (int)hnsw->impl_l2.cur_element_count;
    case HNSW_Dist_Manhattan:
        return (int)hnsw->impl_manhattan.cur_element_count;
    default:
        return -1;
    }
}

size_t hnsw_sizeof(void)
{
	return sizeof(HierarchicalNSW);
}
