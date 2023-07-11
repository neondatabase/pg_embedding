#include "postgres.h"

#include "access/amapi.h"
#include "access/generic_xlog.h"
#include "access/relation.h"
#include "access/reloptions.h"
#include "access/tableam.h"
#include "catalog/index.h"
#include "commands/vacuum.h"
#include "nodes/execnodes.h"
#include "storage/bufmgr.h"
#include "utils/guc.h"
#include "utils/selfuncs.h"

#include <math.h>
#include <float.h>

#include "embedding.h"

PG_MODULE_MAGIC;

#define HNSW_STACK_SIZE 4

/*
 * Postgres specific part of HNSW index.
 * We are not poersisting this data, but reconstruct metadata from relation options.
 * There is not protectionf from altering index option for existed index,
 * butinfoirmation stored in opaque part of HNSW page allows to check if critical
 * metadata fields are changed (dimensiopns and maxM).
 */
typedef struct {
	HnswMetadata	meta;
	Relation    	rel;
	uint64_t     	n_inserted; /* Calculated since start of operation */
	GenericXLogState* xlog_state; /* XLog state for wal logging updated pages */
	size_t			n_buffers; /* Number of simultaneously accessed buffers */
	Buffer			buffers[HNSW_STACK_SIZE];
} HnswIndex;

/*
 * This information in each HNSW page allows to detectincorrect metadata modification (ALTER INDEX)
 * which affects index format
 */
typedef struct
{
	uint16_t dims;
	uint16_t maxM;
} HnswPageOpaque;

/*
 * Options associated with HNSW index, only "dims" is mandatory
 */
typedef struct {
	int32 vl_len_;		/* varlena header (do not touch directly!) */
	int dims;
	int efConstruction;
	int efSearch;
	int M;
} HnswOptions;

static relopt_kind hnsw_relopt_kind;

typedef struct {
	HnswIndex* hnsw;
	size_t curr;
	size_t n_results;
	ItemPointer results;
} HnswScanOpaqueData;

typedef HnswScanOpaqueData* HnswScanOpaque;

#define DEFAULT_EF_CONSTRUCT 16
#define DEFAULT_EF_SEARCH    64
#define DEFAULT_M            100

static bool hnsw_add_point(HnswIndex* hnsw, coord_t const* coord, label_t label);

PGDLLEXPORT void _PG_init(void);

/*
 * Initialize index options and variables
 */
void
_PG_init(void)
{
	hnsw_relopt_kind = add_reloption_kind();
	add_int_reloption(hnsw_relopt_kind, "dims", "Number of dimensions",
					  0, 0, INT_MAX, AccessExclusiveLock);
	add_int_reloption(hnsw_relopt_kind, "m", "Number of neighbors of each vertex",
					  DEFAULT_M, 0, INT_MAX, AccessExclusiveLock);
	add_int_reloption(hnsw_relopt_kind, "efconstruction", "Number of inspected neighbors during index construction",
					  DEFAULT_EF_CONSTRUCT, 1, INT_MAX, AccessExclusiveLock);
	add_int_reloption(hnsw_relopt_kind, "efsearch", "Number of inspected neighbors during index search",
					  DEFAULT_EF_SEARCH, 1, INT_MAX, AccessExclusiveLock);
}


static void
hnsw_build_callback(Relation index, ItemPointer tid, Datum *values,
					bool *isnull, bool tupleIsAlive, void *state)
{
	HnswIndex* hnsw = (HnswIndex*) state;
	ArrayType* array;
	int n_items;
	label_t label = 0;

	/* Skip nulls */
	if (isnull[0])
		return;

	array = DatumGetArrayTypeP(values[0]);
	n_items = ArrayGetNItems(ARR_NDIM(array), ARR_DIMS(array));
	if (n_items != hnsw->meta.dim)
	{
		elog(ERROR, "Wrong number of dimensions: %d instead of %d expected",
			 n_items, (int)hnsw->meta.dim);
	}

	memcpy(&label, tid, sizeof(*tid));
	if (!hnsw_add_point(hnsw, (coord_t*)ARR_DATA_PTR(array), label))
		elog(ERROR, "HNSW index insert failed");
}

static void
hnsw_populate(HnswIndex* hnsw, Relation indexRel, Relation heapRel)
{
	IndexInfo* indexInfo = BuildIndexInfo(indexRel);
	Assert(indexInfo->ii_NumIndexAttrs == 1);
	table_index_build_scan(heapRel, indexRel, indexInfo,
						   true, true, hnsw_build_callback, (void *)hnsw, NULL);
}

static HnswIndex*
hnsw_get_index(Relation indexRel)
{
	HnswIndex* hnsw = (HnswIndex*)palloc(sizeof(HnswIndex));
	HnswOptions *opts = (HnswOptions *) indexRel->rd_options;
	if (opts == NULL || opts->dims == 0) {
		elog(ERROR, "HNSW index requires 'dims' to be specified");
	}
	hnsw->meta.dim = opts->dims;
	hnsw->meta.M = opts->M;
	hnsw->meta.maxM = hnsw->meta.M * 2;
	hnsw->meta.data_size = hnsw->meta.dim * sizeof(coord_t);
	hnsw->meta.offset_data = (hnsw->meta.maxM + 1) * sizeof(idx_t);
	hnsw->meta.offset_label = hnsw->meta.offset_data + hnsw->meta.data_size;
	hnsw->meta.size_data_per_element = hnsw->meta.offset_label + sizeof(label_t);
	hnsw->meta.elems_per_page = (BLCKSZ - MAXALIGN(SizeOfPageHeaderData) - sizeof(HnswPageOpaque)) / (hnsw->meta.size_data_per_element + sizeof(ItemIdData));
	if (hnsw->meta.elems_per_page == 0)
		elog(ERROR, "Elem,ent doesn't fit in Postgres page");
	hnsw->meta.efConstruction = opts->efConstruction;
	hnsw->meta.efSearch = opts->efSearch;
#ifdef __x86_64__
    hnsw->meta.use_avx2 = __builtin_cpu_supports("avx2");
#endif
	hnsw->meta.enterpoint_node = 0;
	hnsw->rel = indexRel;
	hnsw->n_buffers = 0;
	hnsw->xlog_state = NULL;
	hnsw->n_inserted = 0;
	return hnsw;
}

/*
 * Start or restart an index scan
 */
static IndexScanDesc
hnsw_beginscan(Relation index, int nkeys, int norderbys)
{
	IndexScanDesc scan = RelationGetIndexScan(index, nkeys, norderbys);
	HnswScanOpaque so = (HnswScanOpaque) palloc(sizeof(HnswScanOpaqueData));
	so->hnsw = hnsw_get_index(index);
	so->curr = 0;
	so->n_results = 0;
	so->results = NULL;
	scan->opaque = so;
	return scan;
}

/*
 * Start or restart an index scan
 */
static void
hnsw_rescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	if (so->results)
	{
		pfree(so->results);
		so->results = NULL;
	}
	so->curr = 0;
	if (orderbys && scan->numberOfOrderBys > 0)
		memmove(scan->orderByData, orderbys, scan->numberOfOrderBys * sizeof(ScanKeyData));
}

/*
 * Fetch the next tuple in the given scan
 */
static bool
hnsw_gettuple(IndexScanDesc scan, ScanDirection dir)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;

	/*
	 * Index can be used to scan backward, but Postgres doesn't support
	 * backward scan on operators
	 */
	Assert(ScanDirectionIsForward(dir));

	if (so->curr == 0)
	{
		Datum		value;
		ArrayType*	array;
		int         n_items;
		size_t      n_results;
		label_t*    results;

		/* Safety check */
		if (scan->orderByData == NULL)
			elog(ERROR, "cannot scan HNSW index without order");

		/* No items will match if null */
		if (scan->orderByData->sk_flags & SK_ISNULL)
			return false;

		value = scan->orderByData->sk_argument;
		array = DatumGetArrayTypeP(value);
		n_items = ArrayGetNItems(ARR_NDIM(array), ARR_DIMS(array));
		if (n_items != so->hnsw->meta.dim)
		{
			elog(ERROR, "Wrong number of dimensions: %d instead of %d expected",
				 n_items, (int)so->hnsw->meta.dim);
		}

		if (!hnsw_search(&so->hnsw->meta, (coord_t*)ARR_DATA_PTR(array), &n_results, &results))
			elog(ERROR, "HNSW index search failed");
		so->results = (ItemPointer)palloc(n_results*sizeof(ItemPointerData));
		so->n_results = n_results;
		for (size_t i = 0; i < n_results; i++)
		{
			memcpy(&so->results[i], &results[i], sizeof(so->results[i]));
		}
		free(results);
	}
	if (so->curr >= so->n_results)
	{
		return false;
	}
	else
	{
		scan->xs_heaptid = so->results[so->curr++];
		scan->xs_recheckorderby = false;
		return true;
	}
}

/*
 * End a scan and release resources
 */
static void
hnsw_endscan(IndexScanDesc scan)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	if (so->results)
		pfree(so->results);
	if (so->hnsw)
		pfree(so->hnsw);
	pfree(so);
	scan->opaque = NULL;
}


/*
 * Estimate the cost of an index scan
 */
static void
hnsw_costestimate(PlannerInfo *root, IndexPath *path, double loop_count,
				 Cost *indexStartupCost, Cost *indexTotalCost,
				 Selectivity *indexSelectivity, double *indexCorrelation
				 ,double *indexPages
)
{
	GenericCosts costs;

	/* Never use index without order */
	if (path->indexorderbys == NULL)
	{
		*indexStartupCost = DBL_MAX;
		*indexTotalCost = DBL_MAX;
		*indexSelectivity = 0;
		*indexCorrelation = 0;
		*indexPages = 0;
		return;
	}

	MemSet(&costs, 0, sizeof(costs));

	genericcostestimate(root, path, loop_count, &costs);

	/* Startup cost and total cost are same */
	*indexStartupCost = costs.indexTotalCost;
	*indexTotalCost = costs.indexTotalCost;
	*indexSelectivity = costs.indexSelectivity;
	*indexCorrelation = costs.indexCorrelation;
	*indexPages = costs.numIndexPages;
}

/*
 * Parse and validate the reloptions
 */
static bytea *
hnsw_options(Datum reloptions, bool validate)
{
	static const relopt_parse_elt tab[] = {
		{"dims", RELOPT_TYPE_INT, offsetof(HnswOptions, dims)},
		{"efconstruction", RELOPT_TYPE_INT, offsetof(HnswOptions, efConstruction)},
		{"efsearch", RELOPT_TYPE_INT, offsetof(HnswOptions, efSearch)},
		{"m", RELOPT_TYPE_INT, offsetof(HnswOptions, M)}
	};

	return (bytea *) build_reloptions(reloptions, validate,
									  hnsw_relopt_kind,
									  sizeof(HnswOptions),
									  tab, lengthof(tab));
}

/*
 * Validate catalog entries for the specified operator class
 */
static bool
hnsw_validate(Oid opclassoid)
{
	return true;
}

/*
 * Build the index for a logged table
 */
static IndexBuildResult *
hnsw_build(Relation heap, Relation index, IndexInfo *indexInfo)
{
	HnswIndex* hnsw = hnsw_get_index(index);
	IndexBuildResult* result = (IndexBuildResult *) palloc(sizeof(IndexBuildResult));
	hnsw_populate(hnsw, index, heap);
	result->heap_tuples = result->index_tuples = hnsw->n_inserted;
	pfree(hnsw);
	return result;
}

/*
 * Insert a tuple into the index
 */
static bool
hnsw_insert(Relation index, Datum *values, bool *isnull, ItemPointer heap_tid,
			Relation heap, IndexUniqueCheck checkUnique,
			bool indexUnchanged,
			IndexInfo *indexInfo)
{
	Datum value;
	ArrayType* array;
	int n_items;
	label_t label = 0;
	HnswIndex* hnsw;
	bool success;

	/* Skip nulls */
	if (isnull[0])
		return false;

	hnsw = hnsw_get_index(index);

	/* Detoast value */
	value = PointerGetDatum(PG_DETOAST_DATUM(values[0]));
	array = DatumGetArrayTypeP(value);
	n_items = ArrayGetNItems(ARR_NDIM(array), ARR_DIMS(array));
	if (n_items != hnsw->meta.dim)
	{
		elog(ERROR, "Wrong number of dimensions: %d instead of %d expected",
			 n_items, (int)hnsw->meta.dim);
	}
	memcpy(&label, heap_tid, sizeof(*heap_tid));
	success = hnsw_add_point(hnsw, (coord_t*)ARR_DATA_PTR(array), label);
	pfree(hnsw);
	return success;
}

static void hnsw_check_meta(HnswMetadata* meta, Page page)
{
	HnswPageOpaque* opq = (HnswPageOpaque*)PageGetSpecialPointer(page);
	if (opq->dims != (uint16_t)meta->dim ||
		opq->maxM != (uint16_t)meta->maxM)
	{
		elog(ERROR, "Inconsistency with HNSW index metadata: only ef_construction and ef_search options of HNSW index may be altered");
	}
}



static bool hnsw_add_point(HnswIndex* hnsw, coord_t const* coord, label_t label)
{
	BlockNumber rel_size = RelationGetNumberOfBlocks(hnsw->rel);
	GenericXLogState *state = NULL;
	OffsetNumber ins_offs = 0;
	bool extend = rel_size == 0;
	idx_t cur_c;
	Buffer buf;
	Page page;
	HnswPageOpaque* opq;
	char item[BLCKSZ];

	memset(item, 0, hnsw->meta.offset_data);
	memcpy(item + hnsw->meta.offset_data, coord, hnsw->meta.offset_label - hnsw->meta.offset_data);
	memcpy(item + hnsw->meta.offset_label, &label, sizeof(label_t));

	while (true)
	{
		buf = ReadBuffer(hnsw->rel, extend ? P_NEW : rel_size-1);
		LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);

		state = GenericXLogStart(hnsw->rel);

		if (extend)
		{
			Assert(BufferGetBlockNumber(buf) == rel_size);
			page = GenericXLogRegisterBuffer(state, buf, GENERIC_XLOG_FULL_IMAGE);
			PageInit(page, BufferGetPageSize(buf), sizeof(HnswPageOpaque));
			opq = (HnswPageOpaque*)PageGetSpecialPointer(page);
			opq->dims = (uint16_t)hnsw->meta.dim;
			opq->maxM = (uint16_t)hnsw->meta.maxM;
			rel_size += 1;
		}
		else
		{
			page = GenericXLogRegisterBuffer(state, buf, 0);
			hnsw_check_meta(&hnsw->meta, page);
		}

		ins_offs = PageAddItem(page, (Item)item, hnsw->meta.size_data_per_element, InvalidOffsetNumber, false, false);
		if (ins_offs == InvalidOffsetNumber)
		{
			if (extend)
				elog(ERROR, "Failed to append item to the page");
			GenericXLogAbort(state);
			UnlockReleaseBuffer(buf);
			extend = true;
		}
		else
			break;
	}
	MarkBufferDirty(buf);
	GenericXLogFinish(state);
	UnlockReleaseBuffer(buf);

	hnsw->n_inserted += 1;

	cur_c = (rel_size-1)*hnsw->meta.elems_per_page + ins_offs - FirstOffsetNumber;

	return hnsw_bind_point(&hnsw->meta, coord, cur_c);
}


void hnsw_begin_read(HnswMetadata* meta, idx_t idx, idx_t** indexes, coord_t** coords, label_t* label)
{
	HnswIndex* hnsw = (HnswIndex*)meta;
	BlockNumber blkno = idx/meta->elems_per_page;
	Page page;
	Item item;
	ItemId item_id;
	Buffer buf;

	if (hnsw->n_buffers >= HNSW_STACK_SIZE)
		elog(ERROR, "HNSW stack overflow");

	buf = ReadBuffer(hnsw->rel, blkno);

	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);

	hnsw_check_meta(meta, page);

	item_id = PageGetItemId(page, FirstOffsetNumber + idx % meta->elems_per_page);
	item = PageGetItem(page, item_id);

	hnsw->buffers[hnsw->n_buffers++] = buf;

	if (indexes)
		*indexes = (idx_t*)item;

	if (coords)
		*coords = (coord_t*)((char*)item + meta->offset_data);

	if (label)
		memcpy(label, (char*)item + meta->offset_label, sizeof(*label));
}

void hnsw_end_read(HnswMetadata* meta)
{
	HnswIndex* hnsw = (HnswIndex*)meta;
	if (hnsw->n_buffers == 0)
		elog(ERROR, "HNSW stack is empty");
	UnlockReleaseBuffer(hnsw->buffers[--hnsw->n_buffers]);
}

void hnsw_begin_write(HnswMetadata* meta, idx_t idx, idx_t** indexes, coord_t** coords, label_t* label)
{
	HnswIndex* hnsw = (HnswIndex*)meta;
	BlockNumber blkno = idx/meta->elems_per_page;
	Page page;
	ItemId item_id;
	Item item;
	Buffer buf;

	buf = ReadBuffer(hnsw->rel, blkno);

	if (hnsw->xlog_state)
		elog(ERROR, "More than two concurrent write operations");

	if (hnsw->n_buffers >= HNSW_STACK_SIZE)
		elog(ERROR, "HNSW stack overflow");

	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	hnsw->xlog_state = GenericXLogStart(hnsw->rel);
	page = GenericXLogRegisterBuffer(hnsw->xlog_state, buf, 0);

	hnsw_check_meta(meta, page);

	item_id = PageGetItemId(page, FirstOffsetNumber + idx % meta->elems_per_page);
	item = PageGetItem(page, item_id);

	hnsw->buffers[hnsw->n_buffers++] = buf;

	if (indexes)
		*indexes = (idx_t*)item;

	if (coords)
		*coords = (coord_t*)((char*)item + meta->offset_data);

	if (label)
		memcpy(label, (char*)item + meta->offset_label, sizeof(*label));
}

void hnsw_end_write(HnswMetadata* meta)
{
	HnswIndex* hnsw = (HnswIndex*)meta;

	if (hnsw->n_buffers == 0)
		elog(ERROR, "HNSW stack is empty");

	if (hnsw->xlog_state == NULL)
		elog(ERROR, "No buffer is updated");

	hnsw->n_buffers -= 1;
	MarkBufferDirty(hnsw->buffers[hnsw->n_buffers]);
	GenericXLogFinish(hnsw->xlog_state);
	UnlockReleaseBuffer(hnsw->buffers[hnsw->n_buffers]);
	hnsw->xlog_state = NULL;
}


/*
 * Build the index for an unlogged table
 */
static void
hnsw_buildempty(Relation index)
{
}

/*
 * Clean up after a VACUUM operation
 */
static IndexBulkDeleteResult *
hnsw_vacuumcleanup(IndexVacuumInfo *info, IndexBulkDeleteResult *stats)
{
	Relation	rel = info->index;

	if (stats == NULL)
		return NULL;

	stats->num_pages = RelationGetNumberOfBlocks(rel);

	return stats;
}

/*
 * Bulk delete tuples from the index
 */
static IndexBulkDeleteResult *
hnsw_bulkdelete(IndexVacuumInfo *info, IndexBulkDeleteResult *stats,
				IndexBulkDeleteCallback callback, void *callback_state)
{
	if (stats == NULL)
		stats = (IndexBulkDeleteResult *) palloc0(sizeof(IndexBulkDeleteResult));
	return stats;
}

/*
 * Define index handler
 *
 * See https://www.postgresql.org/docs/current/index-api.html
 */
PGDLLEXPORT PG_FUNCTION_INFO_V1(hnsw_handler);
Datum
hnsw_handler(PG_FUNCTION_ARGS)
{
	IndexAmRoutine *amroutine = makeNode(IndexAmRoutine);

	amroutine->amstrategies = 0;
	amroutine->amsupport = 0;
	amroutine->amoptsprocnum = 0;
	amroutine->amcanorder = false;
	amroutine->amcanorderbyop = true;
	amroutine->amcanbackward = false;	/* can change direction mid-scan */
	amroutine->amcanunique = false;
	amroutine->amcanmulticol = false;
	amroutine->amoptionalkey = true;
	amroutine->amsearcharray = false;
	amroutine->amsearchnulls = false;
	amroutine->amstorage = false;
	amroutine->amclusterable = false;
	amroutine->ampredlocks = false;
	amroutine->amcanparallel = false;
	amroutine->amcaninclude = false;
	amroutine->amusemaintenanceworkmem = false; /* not used during VACUUM */
	amroutine->amparallelvacuumoptions = VACUUM_OPTION_PARALLEL_BULKDEL;
	amroutine->amkeytype = InvalidOid;

	/* Interface functions */
	amroutine->ambuild = hnsw_build;
	amroutine->ambuildempty = hnsw_buildempty;
	amroutine->aminsert = hnsw_insert;
	amroutine->ambulkdelete = hnsw_bulkdelete;
	amroutine->amvacuumcleanup = hnsw_vacuumcleanup;
	amroutine->amcanreturn = NULL;	/* tuple not included in heapsort */
	amroutine->amcostestimate = hnsw_costestimate;
	amroutine->amoptions = hnsw_options;
	amroutine->amproperty = NULL;	/* TODO AMPROP_DISTANCE_ORDERABLE */
	amroutine->ambuildphasename = NULL;
	amroutine->amvalidate = hnsw_validate;
	amroutine->amadjustmembers = NULL;
	amroutine->ambeginscan = hnsw_beginscan;
	amroutine->amrescan = hnsw_rescan;
	amroutine->amgettuple = hnsw_gettuple;
	amroutine->amgetbitmap = NULL;
	amroutine->amendscan = hnsw_endscan;
	amroutine->ammarkpos = NULL;
	amroutine->amrestrpos = NULL;

	/* Interface functions to support parallel index scans */
	amroutine->amestimateparallelscan = NULL;
	amroutine->aminitparallelscan = NULL;
	amroutine->amparallelrescan = NULL;

	PG_RETURN_POINTER(amroutine);
}

/*
 * Get the L2 distance between vectors
 */
PGDLLEXPORT PG_FUNCTION_INFO_V1(l2_distance);
Datum
l2_distance(PG_FUNCTION_ARGS)
{
	ArrayType  *a = PG_GETARG_ARRAYTYPE_P(0);
	ArrayType  *b = PG_GETARG_ARRAYTYPE_P(1);
	int         a_dim = ArrayGetNItems(ARR_NDIM(a), ARR_DIMS(a));
	int         b_dim = ArrayGetNItems(ARR_NDIM(b), ARR_DIMS(b));
	dist_t 		distance = 0.0;
	dist_t		diff;
	coord_t	   *ax = (coord_t*)ARR_DATA_PTR(a);
	coord_t	   *bx = (coord_t*)ARR_DATA_PTR(b);

	if (a_dim != b_dim)
	{
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("different array dimensions %d and %d", a_dim, b_dim)));
	}

	#pragma clang loop vectorize(enable)
	for (int i = 0; i < a_dim; i++)
	{
		diff = ax[i] - bx[i];
		distance += diff * diff;
	}

	PG_RETURN_FLOAT4((dist_t)sqrt(distance));
}
