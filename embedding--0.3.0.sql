-- complain if script is sourced in psql, rather than via CREATE EXTENSION
\echo Use "CREATE EXTENSION embedding" to load this file. \quit

-- functions

CREATE FUNCTION l2_distance(real[], real[]) RETURNS real
	AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION cosine_distance(real[], real[]) RETURNS real
	AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION manhattan_distance(real[], real[]) RETURNS real
	AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- operators

CREATE OPERATOR <-> (
	LEFTARG = real[], RIGHTARG = real[], PROCEDURE = l2_distance,
	COMMUTATOR = '<->'
);

CREATE OPERATOR <=> (
	LEFTARG = real[], RIGHTARG = real[], PROCEDURE = cosine_distance,
	COMMUTATOR = '<=>'
);

CREATE OPERATOR <~> (
	LEFTARG = real[], RIGHTARG = real[], PROCEDURE = manhattan_distance,
	COMMUTATOR = '<~>'
);

-- access method

CREATE FUNCTION disk_hnsw_handler(internal) RETURNS index_am_handler
	AS 'MODULE_PATHNAME' LANGUAGE C;

CREATE ACCESS METHOD disk_hnsw TYPE INDEX HANDLER disk_hnsw_handler;

COMMENT ON ACCESS METHOD disk_hnsw IS 'disk_hnsw index access method';

-- opclasses

CREATE OPERATOR CLASS ann_l2_ops
	DEFAULT FOR TYPE real[] USING disk_hnsw AS
	OPERATOR 1 <-> (real[], real[]) FOR ORDER BY float_ops,
	FUNCTION 1 l2_distance(real[], real[]);

CREATE OPERATOR CLASS ann_cos_ops
	FOR TYPE real[] USING disk_hnsw AS
	OPERATOR 1 <=> (real[], real[]) FOR ORDER BY float_ops,
	FUNCTION 1 cosine_distance(real[], real[]);

CREATE OPERATOR CLASS ann_manhattan_ops
	FOR TYPE real[] USING disk_hnsw AS
	OPERATOR 1 <~> (real[], real[]) FOR ORDER BY float_ops,
	FUNCTION 1 manhattan_distance(real[], real[]);
