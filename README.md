# pg_embedding

---
**IMPORTANT NOTICE:**

As of Sept 29, 2023, Neon is no longer committing to `pg_embedding`.

Support will remain in place for existing users of the extension, but we strongly encourage migrating to `pgvector`.

For migration instructions, see [Migrate from pg_embedding to pgvector](https://neon.tech/docs/extensions/pg_embedding#migrate-from-pg_embedding-to-pgvector), in the _Neon documentation_.

---

The `pg_embedding` extension enables the using the Hierarchical Navigable Small World (HNSW) algorithm for vector similarity search in PostgreSQL.

This extension is based on [ivf-hnsw](https://github.com/dbaranchuk/ivf-hnsw) implementation of HNSW
the code for the current state-of-the-art billion-scale nearest neighbor search system<sup>[[1]](#references)</sup>.

## Using the pg_embedding extension

This section describes how to use the `pg_embedding` extension with a simple example demonstrating the required statements, syntax, and options.

For information about migrating from `pgvector` to `pg_embedding`, see [Migrate from pgvector to pg_embedding](https://neon.tech/docs/extensions/pg_embedding#migrate-from-pgvector-to-pgembedding), in the _Neon documentation_.

### Usage summary

The statements in this usage summary are described in further detail in the following sections.

```sql
CREATE EXTENSION embedding;
CREATE TABLE documents(id integer PRIMARY KEY, embedding real[]);
INSERT INTO documents(id, embedding) VALUES (1, '{0,1,2}'), (2, '{1,2,3}'),  (3, '{1,1,1}');
SELECT id FROM documents ORDER BY embedding <-> ARRAY[3,3,3] LIMIT 1;
```

### Enable the extension

To enable the `pg_embedding` extension, run the following `CREATE EXTENSION` statement:

```sql
CREATE EXTENSION embedding;
```

### Create a table for your vector data

To store your vector data, create a table similar to the following:

```sql
CREATE TABLE documents(id INTEGER, embedding REAL[]);
```

This statement generates a table named `documents` with an `embedding` column for storing vector data. Your table and vector column names may differ.

### Insert data

To insert vector data, use an `INSERT` statement similar to the following:

```sql
INSERT INTO documents(id, embedding) VALUES (1, '{0,1,2}'), (2, '{1,2,3}'),  (3, '{1,1,1}');
```

## Query

The `pg_embedding` extension supports Euclidean (L2), Cosine, and Manhattan distance metrics.

Euclidean (L2) distance:

```sql
SELECT id FROM documents ORDER BY embedding <-> array[3,3,3] LIMIT 1;
```

Cosine distance:

```sql
SELECT id FROM documents ORDER BY embedding <=> array[3,3,3] LIMIT 1;
```

Manhattan distance:

```sql
SELECT id FROM documents ORDER BY embedding <~> array[3,3,3] LIMIT 1;
```

where:

- `SELECT id FROM documents` selects the `id` field from all records in the `documents` table.
- `ORDER BY` sorts the selected records in ascending order based on the calculated distances. In other words, records with values closer to the `[1.1, 2.2, 3.3]` query vector according to the distance metric will be returned first.
- `<->`, `<=>`, and `<~>` operators define the distance metric, which calculates the distance between the query vector and each row of the dataset.
- `LIMIT 1` limits the result set to one record after sorting.

In summary, the query retrieves the ID of the record from the `documents` table whose value is closest to the `[3,3,3]` query vector according to the specified distance metric.

### Create an HNSW index

To optimize search behavior, you can add an HNSW index. To create the HNSW index on your vector column, use a `CREATE INDEX` statement as shown in the following examples. The `pg_embedding` extension supports indexes for use with Euclidean, Cosine, and Manhattan distance metrics.

Euclidean (L2) distance index:

```sql
CREATE INDEX ON documents USING hnsw(embedding) WITH (dims=3, m=3, efconstruction=5, efsearch=5);
SET enable_seqscan = off;
SELECT id FROM documents ORDER BY embedding <-> array[3,3,3] LIMIT 1;
```

Cosine distance index:

```sql
CREATE INDEX ON documents USING hnsw(embedding ann_cos_ops) WITH (dims=3, m=3, efconstruction=5, efsearch=5);
SET enable_seqscan = off;
SELECT id FROM documents ORDER BY embedding <=> array[3,3,3] LIMIT 1;
```

Manhattan distance index:

```sql
CREATE INDEX ON documents USING hnsw(embedding ann_manhattan_ops) WITH (dims=3, m=3, efconstruction=5, efsearch=5);
SET enable_seqscan = off;
SELECT id FROM documents ORDER BY embedding <~> array[3,3,3] LIMIT 1;
```

### Tuning the HNSW algorithm

The following options allow you to tune the HNSW algorithm when creating an index:

- `dims`: Defines the number of dimensions in your vector data.  This is a required parameter.
- `m`: Defines the maximum number of links or "edges" created for each node during graph construction. A higher value increases accuracy (recall) but also increases the size of the index in memory and index construction time.
- `efconstruction`: Influences the trade-off between index quality and construction speed. A high `efconstruction` value creates a higher quality graph, enabling more accurate search results, but a higher value also means that index construction takes longer.
- `efsearch`: Influences the trade-off between query accuracy (recall) and speed. A higher `efsearch` value increases accuracy at the cost of speed. This value should be equal to or larger than `k`, which is the number of nearest neighbors you want your search to return (defined by the `LIMIT` clause in your `SELECT` query).

In summary, to prioritize search speed over accuracy, use lower values for `m` and `efsearch`. Conversely, to prioritize accuracy over search speed, use a higher value for `m` and `efsearch`. A higher `efconstruction` value enables more accurate search results at the cost of index build time, which is also affected by the size of your dataset.

## How HNSW search works

HNSW is a graph-based approach to indexing multi-dimensional data. It constructs a multi-layered graph, where each layer is a subset of the previous one. During a search, the algorithm navigates through the graph from the top layer to the bottom to quickly find the nearest neighbor. An HNSW graph is known for its superior performance in terms of speed and accuracy.

The search process begins at the topmost layer of the HNSW graph. From the starting node, the algorithm navigates to the nearest neighbor in the same layer. The algorithm repeats this step until it can no longer find neighbors more similar to the query vector.

Using the found node as an entry point, the algorithm moves down to the next layer in the graph and repeats the process of navigating to the nearest neighbor. The process of navigating to the nearest neighbor and moving down a layer is repeated until the algorithm reaches the bottom layer.

In the bottom layer, the algorithm continues navigating to the nearest neighbor until it can't find any nodes that are more similar to the query vector. The current node is then returned as the most similar node to the query vector.

The key idea behind HNSW is that by starting the search at the top layer and moving down through each layer, the algorithm can quickly navigate to the area of the graph that contains the node that is most similar to the query vector. This makes the search process much faster than if it had to search through every node in the graph.

## References

- [1] Dmitry Baranchuk, Artem Babenko, Yury Malkov; Proceedings of the European Conference on Computer Vision (ECCV), 2018, pp. 202-216 <sup>[link](http://openaccess.thecvf.com/content_ECCV_2018/html/Dmitry_Baranchuk_Revisiting_the_Inverted_ECCV_2018_paper.html)</sup>
