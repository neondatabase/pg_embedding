#include <hdf5_hl.h>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

#define ANKERL_NANOBENCH_IMPLEMENT
#include "nanobench.h"

#include "../hnswalg.cpp"

constexpr int M = 32;
constexpr int EfConstruction = 16;
constexpr int EfSearch = 64;

struct Dataset
{
    std::vector<hsize_t> shape;
    std::vector<float> data;
};

Dataset GetDataset(hid_t file_id, const char *dataset_name)
{
    int rank;
    herr_t err;
    err = H5LTget_dataset_ndims(file_id, dataset_name, &rank);
    if(err < 0)
    {
        fprintf(stderr, "Failed to get rank of dataset %s\n", dataset_name);
        exit(1);
    }

    std::vector<hsize_t> dims(rank);
    H5T_class_t class_id;
    size_t type_size;
    err = H5LTget_dataset_info(
        file_id,
        dataset_name,
        dims.data(),
        &class_id,
        &type_size);
    if(err < 0)
    {
        fprintf(stderr, "Failed to get info for dataset %s\n", dataset_name);
        exit(1);
    }
    if(class_id != H5T_FLOAT)
    {
        fprintf(stderr, "Expected float data type for dataset %s\n", dataset_name);
        exit(1);
    }

    hsize_t num_elements = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<hsize_t>());
    std::vector<float> result(num_elements);
    err = H5LTread_dataset_float(file_id, dataset_name, result.data());
    if(err < 0)
    {
        fprintf(stderr, "Failed to read dataset %s\n", dataset_name);
        exit(1);
    }
    return { std::move(dims), std::move(result) };
}

int main(int argc, const char **argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "Please specify a dataset\n");
        return 1;
    }

    hid_t file_id = H5Fopen(argv[1], H5F_ACC_RDONLY, H5P_DEFAULT);
    if(file_id < 0)
    {
        fprintf(stderr, "Invalid dataset: %s\n", argv[1]);
        return 1;
    }

    Dataset train = GetDataset(file_id, "train");

    auto maxelements = train.shape[0];
    auto maxM = M * 2;
    auto data_size = train.shape.size() * sizeof(coord_t);
    auto size_links_level0 = (maxM + 1) * sizeof(idx_t);
    auto size_data_per_element = size_links_level0 + data_size + sizeof(label_t);
    auto index_size = hnsw_sizeof() + maxelements * size_data_per_element;

    void *memory = ::operator new(index_size);
    HierarchicalNSW *hnsw = reinterpret_cast<HierarchicalNSW *>(memory);
    ankerl::nanobench::Bench().batch(train.shape[0]).run("Build", [&]()
    {
        hnsw_init(hnsw, train.shape.size(), maxelements, M, maxM, EfConstruction);
        for(size_t i = 0; i < train.data.size(); i += train.shape[1])
            hnsw_add_point(hnsw, &train.data[i], 0);
    });

    Dataset test = GetDataset(file_id, "test");

    ankerl::nanobench::Bench().batch(test.shape[0]).run("Query", [&]()
    {
        for(size_t i = 0; i < test.data.size(); i += test.shape[1])
        {
            size_t n_results;
            label_t *results;
            hnsw_search(hnsw, &test.data[i], EfSearch, &n_results, &results);
            free(results);
        }
    });

    return 0;
}
