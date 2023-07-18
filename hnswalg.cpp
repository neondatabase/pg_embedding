#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <cmath>
#include <queue>
#include <stdexcept>

extern "C" {
#include "embedding.h"
}

static std::priority_queue<std::pair<dist_t, idx_t>>
searchBaseLayer(HnswMetadata* meta, const coord_t *point, size_t ef)
{
	std::vector<uint32_t> visited;
	const size_t init_visited_size = 64*1024;
	coord_t* p_coords;
	idx_t* p_indexes;

	visited.resize(init_visited_size);

    std::priority_queue<std::pair<dist_t, idx_t >> topResults;
    std::priority_queue<std::pair<dist_t, idx_t >> candidateSet;

	idx_t enterpoint_node = meta->enterpoint_node;
	if (!hnsw_begin_read(meta, enterpoint_node, NULL, &p_coords, NULL))
		return topResults;

    dist_t dist = hnsw_dist_func(meta, point, p_coords);
	hnsw_end_read(meta);

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

		hnsw_begin_read(meta, curNodeNum, &p_indexes, NULL, NULL);
        size_t size = p_indexes[0];

        for (size_t j = 0; j < size; ++j) {
            size_t tnum = p_indexes[1 + j];

			if (visited.size() <= (tnum >> 5))
				visited.resize((tnum >> 5) + 1);

            if (!(visited[tnum >> 5] & (1 << (tnum & 31)))) {
				visited[tnum >> 5] |= 1 << (tnum & 31);

				hnsw_begin_read(meta, tnum, NULL, &p_coords, NULL);
                dist = hnsw_dist_func(meta, point, p_coords);
				hnsw_end_read(meta);

                if (topResults.top().first > dist || topResults.size() < ef) {
                    candidateSet.emplace(-dist, tnum);

                    topResults.emplace(dist, tnum);

                    if (topResults.size() > ef)
                        topResults.pop();

                    lowerBound = topResults.top().first;
                }
            }
        }
		hnsw_end_read(meta);
    }
    return topResults;
}


void getNeighborsByHeuristic(HnswMetadata* meta, std::priority_queue<std::pair<dist_t, idx_t>> &topResults, size_t NN)
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
			coord_t *p_coords, *p_coords2;
			hnsw_begin_read(meta, curen2.second, NULL, &p_coords2, NULL);
			hnsw_begin_read(meta, curen.second, NULL, &p_coords, NULL);
            dist_t curdist = hnsw_dist_func(meta, p_coords2, p_coords);
			hnsw_end_read(meta);
			hnsw_end_read(meta);
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

void mutuallyConnectNewElement(HnswMetadata* meta, const coord_t *point, idx_t cur_c,
                               std::priority_queue<std::pair<dist_t, idx_t>> topResults)
{
    getNeighborsByHeuristic(meta, topResults, meta->M);

	idx_t   *p_indexes;
	coord_t *p_coord, *p_coord2;
    std::vector<idx_t> res;
    res.reserve(meta->M);
    while (topResults.size() > 0) {
        res.push_back(topResults.top().second);
        topResults.pop();
    }
    {
		hnsw_begin_write(meta, cur_c, &p_indexes, NULL, NULL);
        if (*p_indexes)
            throw std::runtime_error("Should be blank");

        *p_indexes++ = res.size();

        for (size_t idx = 0; idx < res.size(); idx++) {
            if (p_indexes[idx])
                throw std::runtime_error("Should be blank");
            p_indexes[idx] = res[idx];
        }
		hnsw_end_write(meta);
    }
    for (size_t idx = 0; idx < res.size(); idx++) {
        if (res[idx] == cur_c)
            throw std::runtime_error("Connection to the same element");

        size_t resMmax = meta->maxM;
		hnsw_begin_write(meta, res[idx], &p_indexes, &p_coord, NULL);
        idx_t sz_link_list_other = *p_indexes;

        if (sz_link_list_other > resMmax || sz_link_list_other < 0)
            throw std::runtime_error("Bad sz_link_list_other");

        if (sz_link_list_other < resMmax) {
            p_indexes[1 + sz_link_list_other] = cur_c;
            *p_indexes = sz_link_list_other + 1;
        } else {
            // finding the "weakest" element to replace it with the new one
			hnsw_begin_read(meta, cur_c, NULL, &p_coord2, NULL);
            dist_t d_max = hnsw_dist_func(meta, p_coord2, p_coord);
			hnsw_end_read(meta);
            // Heuristic:
            std::priority_queue<std::pair<dist_t, idx_t>> candidates;
            candidates.emplace(d_max, cur_c);

            for (size_t j = 0; j < sz_link_list_other; j++)
			{
				hnsw_begin_read(meta, p_indexes[1 + j], NULL, &p_coord2, NULL);
				candidates.emplace(hnsw_dist_func(meta, p_coord2, p_coord), p_indexes[1 + j]);
				hnsw_end_read(meta);
			}
            getNeighborsByHeuristic(meta, candidates, resMmax);

            size_t indx = 0;
            while (!candidates.empty()) {
                p_indexes[1 + indx] = candidates.top().second;
                candidates.pop();
                indx++;
            }
            *p_indexes = indx;
        }
		hnsw_end_write(meta);
    }
}

void bindPoint(HnswMetadata* meta, coord_t const* point, idx_t cur_c)
{
    // Do nothing for the first element
    if (cur_c != 0) {
        std::priority_queue <std::pair<dist_t, idx_t>> topResults = searchBaseLayer(meta, point, meta->efConstruction);
        mutuallyConnectNewElement(meta, point, cur_c, topResults);
    }
}

std::priority_queue<std::pair<dist_t, label_t>> searchKnn(HnswMetadata* meta, const coord_t *query, size_t k)
{
	std::priority_queue<std::pair<dist_t, label_t>> topResults;
	auto topCandidates = searchBaseLayer(meta, query, k);
    while (topCandidates.size() > k) {
        topCandidates.pop();
	}
	while (!topCandidates.empty()) {
		std::pair<dist_t, idx_t> rez = topCandidates.top();
		label_t label;
		hnsw_begin_read(meta, rez.second, NULL, NULL, &label);
		topResults.push(std::pair<dist_t, label_t>(rez.first, label));
		topCandidates.pop();
		hnsw_end_read(meta);
	}

    return topResults;
}



bool hnsw_search(HnswMetadata* meta, const coord_t *point, size_t* n_results, label_t** results)
{
	try
	{
		auto result = searchKnn(meta, point, meta->efSearch);
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

bool hnsw_bind_point(HnswMetadata* meta, const coord_t *point, idx_t cur)
{
	try
	{
		bindPoint(meta, point, cur);
		return true;
	}
	catch (std::exception& x)
	{
		fprintf(stderr, "Catch %s\n", x.what());
		return false;
	}
}

