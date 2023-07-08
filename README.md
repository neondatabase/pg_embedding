# pg_embedding

The `pg_embedding` extension enables the use of the Hierarchical Navigable Small World (HNSW) algorithm for vector similarity search in PostgreSQL.

HNSW is a graph-based approach to indexing multi-dimensional data. It constructs a multi-layered graph, where each layer is a subset of the previous one. During a search, the algorithm navigates through the graph from the top layer to the lowest layer to quickly find the nearest neighbor. 

Ths extension is based on [ivf-hnsw](https://github.com/dbaranchuk/ivf-hnsw) implementation of [HNSW](https://www.pinecone.io/learn/hnsw),
the code for the current state-of-the-art billion-scale nearest neighbor search system<sup>[[1]](#references)</sup>.

## Usage summary

```sql
CREATE EXTENSION embedding;
CREATE TABLE documents(id integer PRIMARY KEY, embedding real[]);
CREATE INDEX ON documents USING hnsw(embedding) WITH (maxelements=1000000, dims=100, m=32);
SELECT id FROM documents ORDER BY emebedding <-> ARRAY[1.0, 2.0,...] LIMIT 100;
```

## Enable the extension

Enable the `pg_embedding` extension in PostgreSQL by running the following `CREATE EXTENSION` statement.

```sql
CREATE EXTENSION embedding;
```

## Create a table for vector data

To store vector data, create a table similar to the following:

```sql
CREATE TABLE documents(id INTEGER, embedding REAL[]);
```

This statement generates a table named `documents` with a `embedding` column for storing vector data. (Your table and vector column names may differ.)

## Insert data

To insert vector data:

```sql
INSERT INTO documents(id, embedding) 
VALUES 
(1, '{1.1, 2.2, 3.3,...}'),
(2, '{4.4, 5.5, 6.6,...}'),
(3, '{7.7, 8.8, 9.9,...}');
```

## Create an HNSW index

HNSW indexes are created in memory and built on demand.

To create an HNSW index on your vector column, use a `CREATE INDEX` statement similar to the following:

```sql
CREATE INDEX ON documents USING hnsw(embedding) WITH (maxelements=1000000, dims=100, m=32);
```

## HNSW index options

- `maxelements`: Defines the maximum number of elements indexed. This is a required parameter. An "element" refers to a data point (a vector) in the dataset, which is represented as a node in the HNSW graph. Typically, you would set this option to a value able to accommodate the number of rows in your in your dataset.
- `dims`: Defines the number of dimensions in your vector data.  
- `m`: Defines the maximum number of links (also referred to as "edges") created for each node during graph construction.
- `efConstruction`: Defines the number of nearest neighbors considered during index construction. The default value is `32`. This is an optional parameter.
- `efsearch`: Defines the number of nearest neighbors considered during index search. The default value is `32`. This is an optional parameter.

## Query the indexed data

To query the indexed data for nearest neighbors, use a query similar to this:

```sql
SELECT id FROM documents ORDER BY embedding <-> array[1.1, 2.2, 3.3,...] LIMIT 100;
```

where:

- `SELECT id FROM documents` selects the `id` field from all records in the `documents` table.
- `<->` is the PostgreSQL "distance between" operator. It calculates the Euclidean distance (L2) between the query vector and each row of the dataset.
- `ORDER BY` sorts the selected records in ascending order based on the calculated distances. In other words, records with values closer to the `[1.1, 2.2, 3.3,...]` query vector will be returned first.
- `LIMIT 100` limits the result set to the first 100 records after sorting.

In summary, the query retrieves the IDs of the first 100 records from the `documents` table whose value is closest to the `[1.1, 2.2, 3.3,...]` query vector according to Euclidean distance.

## Tuning the HNSW algorithm

The `m`, `efConstruction`, and `efSearch` options allow you to tune the HNSW algorithm when creating an index.

- `m`: A higher value increases accuracy (recall) but also increases the size of the index in memory and index construction time.
- `efConstruction`: This setting influences the trade-off between index quality and construction speed. A high `efConstruction` value creates a higher quality graph, enabling more accurate search results but also means a longer index construction time.
- `efSearch`: This setting influences the trade-off between query accuracy (recall) and speed. A higher `efSearch` value increases accuracy at the cost of speed. This value should be equal to or larger than `k`, which is the number of nearest neighbors you want your search to return.

## References
- [1] Dmitry Baranchuk, Artem Babenko, Yury Malkov; Proceedings of the European Conference on Computer Vision (ECCV), 2018, pp. 202-216 <sup>[link](http://openaccess.thecvf.com/content_ECCV_2018/html/Dmitry_Baranchuk_Revisiting_the_Inverted_ECCV_2018_paper.html)</sup>
