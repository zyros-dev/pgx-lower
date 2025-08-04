#ifndef COMPILER_RUNTIME_ALL_WRAPPERS_H
#define COMPILER_RUNTIME_ALL_WRAPPERS_H

#include "compiler/runtime/helpers.h"

namespace pgx_lower::compiler::runtime {

// Arrow runtime wrappers
struct ArrowColumnBuilder {
    static RuntimeCallGenerator create(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_arrow_column_builder_create", builder, loc);
    }
    static RuntimeCallGenerator finish(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_arrow_column_builder_finish", builder, loc);
    }
    static RuntimeCallGenerator merge(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_arrow_column_builder_merge", builder, loc);
    }
};

struct ArrowTable {
    static RuntimeCallGenerator createEmpty(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_arrow_table_create_empty", builder, loc);
    }
    static RuntimeCallGenerator addColumn(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_arrow_table_add_column", builder, loc);
    }
};

// Buffer Iterator
struct BufferIterator {
    static RuntimeCallGenerator iterate(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_buffer_iterator_iterate", builder, loc);
    }
};

// Entry Lock
struct EntryLock {
    static RuntimeCallGenerator initialize(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_entry_lock_initialize", builder, loc);
    }
    static RuntimeCallGenerator lock(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_entry_lock_lock", builder, loc);
    }
    static RuntimeCallGenerator unlock(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_entry_lock_unlock", builder, loc);
    }
};

// Execution tracing
struct ExecutionStepTracing {
    static RuntimeCallGenerator start(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_execution_step_tracing_start", builder, loc);
    }
    static RuntimeCallGenerator end(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_execution_step_tracing_end", builder, loc);
    }
};

// Growing Buffer Allocator
struct GrowingBufferAllocator {
    static RuntimeCallGenerator getDefaultAllocator(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_growing_buffer_allocator_get_default", builder, loc);
    }
    static RuntimeCallGenerator getGroupAllocator(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_growing_buffer_allocator_get_group", builder, loc);
    }
};

// Hash structures
struct HashIndexAccess {
    static RuntimeCallGenerator lookup(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_hash_index_access_lookup", builder, loc);
    }
};

struct HashIndexedView {
    static RuntimeCallGenerator build(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_hash_indexed_view_build", builder, loc);
    }
};

struct HashIndexIteration {
    static RuntimeCallGenerator hasNext(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_hash_index_iteration_has_next", builder, loc);
    }
    static RuntimeCallGenerator consumeRecordBatch(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_hash_index_iteration_consume_record_batch", builder, loc);
    }
};

struct HashMultiMap {
    static RuntimeCallGenerator create(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_hash_multi_map_create", builder, loc);
    }
    static RuntimeCallGenerator insertEntry(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_hash_multi_map_insert_entry", builder, loc);
    }
    static RuntimeCallGenerator insertValue(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_hash_multi_map_insert_value", builder, loc);
    }
};

struct Hashtable {
    static RuntimeCallGenerator create(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_hashtable_create", builder, loc);
    }
    static RuntimeCallGenerator createIterator(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_hashtable_create_iterator", builder, loc);
    }
    static RuntimeCallGenerator insert(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_hashtable_insert", builder, loc);
    }
    static RuntimeCallGenerator merge(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_hashtable_merge", builder, loc);
    }
};

// Heap
struct Heap {
    static RuntimeCallGenerator create(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_heap_create", builder, loc);
    }
    static RuntimeCallGenerator getBuffer(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_heap_get_buffer", builder, loc);
    }
    static RuntimeCallGenerator insert(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_heap_insert", builder, loc);
    }
    static RuntimeCallGenerator merge(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_heap_merge", builder, loc);
    }
};

// Pre-aggregation
struct PreAggregationHashtable {
    static RuntimeCallGenerator createIterator(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_pre_aggregation_hashtable_create_iterator", builder, loc);
    }
    static RuntimeCallGenerator merge(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_pre_aggregation_hashtable_merge", builder, loc);
    }
};

struct PreAggregationHashtableFragment {
    static RuntimeCallGenerator create(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_pre_aggregation_hashtable_fragment_create", builder, loc);
    }
    static RuntimeCallGenerator insert(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_pre_aggregation_hashtable_fragment_insert", builder, loc);
    }
};

// Relation helpers
struct RelationHelper {
    static RuntimeCallGenerator accessHashIndex(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_relation_helper_access_hash_index", builder, loc);
    }
};

// Segment tree
struct SegmentTreeView {
    static RuntimeCallGenerator build(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_segment_tree_view_build", builder, loc);
    }
    static RuntimeCallGenerator lookup(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_segment_tree_view_lookup", builder, loc);
    }
};

// Simple state
struct SimpleState {
    static RuntimeCallGenerator create(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_simple_state_create", builder, loc);
    }
    static RuntimeCallGenerator merge(mlir::OpBuilder& builder, mlir::Location loc) {
        return RuntimeCallGenerator("pgx_simple_state_merge", builder, loc);
    }
};

} // namespace pgx_lower::compiler::runtime

#endif // COMPILER_RUNTIME_ALL_WRAPPERS_H