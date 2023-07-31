SET enable_seqscan = off;

CREATE TABLE t (val real[]);
INSERT INTO t (val) VALUES ('{0,1,2}'), ('{1,2,3}'), ('{1,1,1}'), (NULL);
CREATE INDEX ON t USING disk_hnsw (val) WITH (dims=3, m=3);

INSERT INTO t (val) VALUES (array[1,2,4]);

explain SELECT * FROM t ORDER BY val <-> array[3,3,3];
SELECT * FROM t ORDER BY val <-> array[3,3,3];
SELECT COUNT(*) FROM t;

CREATE INDEX ON t USING disk_hnsw (val ann_cos_ops) WITH (dims=3, m=3);
explain SELECT * FROM t ORDER BY val <=> array[3,3,3];
SELECT * FROM t ORDER BY val <=> array[3,3,3];

CREATE INDEX ON t USING disk_hnsw (val ann_manhattan_ops) WITH (dims=3, m=3);
explain SELECT * FROM t ORDER BY val <~> array[3,3,3];
SELECT * FROM t ORDER BY val <~> array[3,3,3];

SET enable_seqscan = on;
SELECT * FROM t ORDER BY val <-> array[3,3,3];
SELECT * FROM t ORDER BY val <=> array[3,3,3];
SELECT * FROM t ORDER BY val <~> array[3,3,3];

delete from t;
vacuum t;
INSERT INTO t (val) VALUES ('{0,1,2}'), ('{1,2,3}'), ('{1,1,1}'), (NULL), (array[1,2,4]);
SET enable_seqscan = off;
SELECT * FROM t ORDER BY val <-> array[3,3,3];
SELECT * FROM t ORDER BY val <=> array[3,3,3];
SELECT * FROM t ORDER BY val <~> array[3,3,3];



DROP TABLE t;
