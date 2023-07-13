# pg_embedding

The `pg_embedding` extension enables the use of the Hierarchical Navigable Small World (HNSW) algorithm for vector similarity search in PostgreSQL.

This extension is based on [ivf-hnsw](https://github.com/dbaranchuk/ivf-hnsw) implementation of HNSW
the code for the current state-of-the-art billion-scale nearest neighbor search system<sup>[[1]](#references)</sup>.

## Using the pg_embedding extension

This section describes how to use the `pg_embedding` extension in Neon with a simple example that demonstrates the required statements, syntax, and options.

### Usage summary

The statements in this usage summary are described in further detail in the sections that follow.

```sql
CREATE EXTENSION embedding;
CREATE TABLE documents(id integer PRIMARY KEY, embedding real[]);
SELECT id FROM documents ORDER BY embedding <-> ARRAY[1.1, 2.2, 3.3] LIMIT 1;
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
INSERT INTO documents(id, embedding) 
VALUES (1, '{1.1, 2.2, 3.3}'),(2, '{4.4, 5.5, 6.6}');
```

## Query the nearest neighbors by L2 distance

To query the nearest neighbors by L2 distance, use a query similar to this:

```sql
SELECT id FROM documents ORDER BY embedding <-> array[1.1, 2.2, 3.3] LIMIT 1;
```

where:

- `SELECT id FROM documents` selects the `id` field from all records in the `documents` table.
- `<->`: This is a "distance between" operator. It calculates the Euclidean distance (L2) between the query vector and each row of the dataset.
- `ORDER BY` sorts the selected records in ascending order based on the calculated distances. In other words, records with values closer to the `[1.1, 2.2, 3.3]` query vector will be returned first.
- `LIMIT 1` limits the result set to one record after sorting.

In summary, the query retrieves the ID of the record from the `documents` table whose value is closest to the `[1.1, 2.2, 3.3]` query vector according to Euclidean distance.

### Create an HNSW index

To optimize search behavior, you can add an HNSW index. To create the HNSW index on your vector column, use a `CREATE INDEX` statement similar to the following:

```sql
CREATE INDEX ON documents USING hnsw(embedding) WITH (maxelements=1000, dims=3, m=8);
```

**Note:** HNSW indexes are created in memory. If released from memory, the index is rebuilt the next time it is accessed.

### HNSW index options

- `maxelements`: Defines the maximum number of elements indexed. This is a required parameter.
- `dims`: Defines the number of dimensions in your vector data.  This is a required parameter.
- `m`: Defines the maximum number of links (also referred to as "edges") created for each node during graph construction.
- `efConstruction`: Defines the number of nearest neighbors considered during index construction. The default value is `32`.
- `efsearch`: Defines the number of nearest neighbors considered during index search. The default value is `32`.

### Tuning the HNSW algorithm

The `m`, `efConstruction`, and `efSearch` options allow you to tune the HNSW algorithm when creating an index:

- `m`: This option defines the maximum number of links or "edges" created for each node during graph construction. A higher value increases accuracy (recall) but also increases the size of the index in memory and index construction time.
- `efConstruction`: This option influences the trade-off between index quality and construction speed. A high `efConstruction` value creates a higher quality graph, enabling more accurate search results, but a higher value also means that index construction takes longer.
- `efSearch`: This option influences the trade-off between query accuracy (recall) and speed. A higher `efSearch` value increases accuracy at the cost of speed. This value should be equal to or larger than `k`, which is the number of nearest neighbors you want your search to return.

In summary, to prioritize search speed over accuracy, use lower values for `m` and `efSearch`. Conversely, to prioritize accuracy over search speed, use a higher value for `m` and `efSearch`. At the cost of index build time, you can also use a higher `efConstruction` value to enable more accurate search results.

## How HNSW search works

HNSW is a graph-based approach to indexing multi-dimensional data. It constructs a multi-layered graph, where each layer is a subset of the previous one. During a search, the algorithm navigates through the graph from the top layer to the bottom to quickly find the nearest neighbor. An HNSW graph is known for its superior performance in terms of speed and accuracy.

The search process begins at the topmost layer of the HNSW graph. From the starting node, the algorithm navigates to the nearest neighbor in the same layer. The algorithm repeats this step until it can no longer find neighbors more similar to the query vector.

Using the found node as an entry point, the algorithm moves down to the next layer in the graph and repeats the process of navigating to the nearest neighbor. The process of navigating to the nearest neighbor and moving down a layer is repeated until the algorithm reaches the bottom layer.

In the bottom layer, the algorithm continues navigating to the nearest neighbor until it can't find any nodes that are more similar to the query vector. The current node is then returned as the most similar node to the query vector.

The key idea behind HNSW is that by starting the search at the top layer and moving down through each layer, the algorithm can quickly navigate to the area of the graph that contains the node that is most similar to the query vector. This makes the search process much faster than if it had to search through every node in the graph.

## References

- [1] Dmitry Baranchuk, Artem Babenko, Yury Malkov; Proceedings of the European Conference on Computer Vision (ECCV), 2018, pp. 202-216 <sup>[link](http://openaccess.thecvf.com/content_ECCV_2018/html/Dmitry_Baranchuk_Revisiting_the_Inverted_ECCV_2018_paper.html)</sup>
