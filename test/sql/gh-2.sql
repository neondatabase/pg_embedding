-- https://github.com/neondatabase/pg_embedding/issues/2

SET enable_seqscan = off;

CREATE TABLE t (val real[]);
CREATE INDEX ON t USING hnsw (val) WITH (dims=3, m=3);
SELECT * FROM t ORDER BY val <-> array[3,3,3];


DROP TABLE t;
