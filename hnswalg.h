// Copyright (c) 2017 Dmitry Baranchuk
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

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

struct HierarchicalNSW : HnswMetadata
{
  public:
	HierarchicalNSW(size_t dim, size_t maxelements, size_t M, size_t maxM, size_t efConstruction);
	~HierarchicalNSW();


	inline coord_t *getDataByInternalId(idx_t internal_id) const {
		return (coord_t *)&data_level0_memory[internal_id * size_data_per_element + offset_data];
	}

	inline idx_t *get_linklist0(idx_t internal_id) const {
		return (idx_t*)&data_level0_memory[internal_id * size_data_per_element];
	}

	inline label_t *getExternalLabel(idx_t internal_id) const {
		return (label_t *)&data_level0_memory[internal_id * size_data_per_element + offset_label];
	}

	std::priority_queue<std::pair<dist_t, idx_t>> searchBaseLayer(const coord_t *x, size_t ef);

	void getNeighborsByHeuristic(std::priority_queue<std::pair<dist_t, idx_t>> &topResults, size_t NN);

	void mutuallyConnectNewElement(const coord_t *x, idx_t id, std::priority_queue<std::pair<dist_t, idx_t>> topResults);

	void addPoint(const coord_t *point, label_t label);

	std::priority_queue<std::pair<dist_t, label_t>> searchKnn(const coord_t *query_data, size_t k);
};
