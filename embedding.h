/*
 * Copyright 2023 Neon Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

typedef float    coord_t;
typedef float    dist_t;
typedef uint32_t idx_t;
typedef uint64_t label_t;

typedef struct HierarchicalNSW HierarchicalNSW;

bool hnsw_search(HierarchicalNSW* hnsw, const coord_t *point, size_t efSearch, size_t* n_results, label_t** results);
bool hnsw_add_point(HierarchicalNSW* hnsw, const coord_t *point, label_t label);
void hnsw_init(HierarchicalNSW* hnsw, size_t dim, size_t maxelements, size_t M, size_t maxM, size_t efConstruction);
int  hnsw_dimensions(HierarchicalNSW* hnsw);
size_t hnsw_count(HierarchicalNSW* hnsw);
size_t hnsw_sizeof(void);
