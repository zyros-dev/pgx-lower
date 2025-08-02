#include "../Headers/SubOpToControlFlowPatterns.h"
#include "../Headers/SubOpToControlFlowUtilities.h"

namespace pgx_lower {
namespace compiler {
namespace dialect {
namespace subop_to_cf {

// EntryStorageHelper is defined in SubOpToControlFlowUtilities.cpp


//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Check if atomic store is supported on current platform
static bool checkAtomicStore(mlir::Operation* op) {
   //on x86, stores are always atomic (if aligned)
#ifdef __x86_64__
   return true;
#else
   return !op->hasAttr("atomic");
#endif
}

//===----------------------------------------------------------------------===//
// Gather Operations - Data Loading Patterns
//===----------------------------------------------------------------------===//

/// Default gather operation lowering for state entry references
class DefaultGatherOpLowering : public SubOpTupleStreamConsumerConversionPattern<subop::GatherOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::GatherOp>::SubOpTupleStreamConsumerConversionPattern;

   LogicalResult matchAndRewrite(subop::GatherOp gatherOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      auto referenceType = mlir::cast<subop::StateEntryReference>(gatherOp.getRef().getColumn().type);
      EntryStorageHelper storageHelper(gatherOp, referenceType.getMembers(), referenceType.hasLock(), typeConverter);
      storageHelper.loadIntoColumns(gatherOp.getMapping(), mapping, mapping.resolve(gatherOp, gatherOp.getRef()), rewriter, gatherOp->getLoc());
      rewriter.replaceTupleStream(gatherOp, mapping);
      return success();
   }
};

/// Continuous reference gather operation lowering
class ContinuousRefGatherOpLowering : public SubOpTupleStreamConsumerConversionPattern<subop::GatherOp, 2> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::GatherOp, 2>::SubOpTupleStreamConsumerConversionPattern;

   LogicalResult matchAndRewrite(subop::GatherOp gatherOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      auto continuousRefEntryType = mlir::dyn_cast_or_null<subop::ContinuousEntryRefType>(gatherOp.getRef().getColumn().type);
      if (!continuousRefEntryType) { return failure(); }
      llvm::SmallVector<mlir::Value> unPackedReference;
      rewriter.createOrFold<util::UnPackOp>(unPackedReference, gatherOp->getLoc(), mapping.resolve(gatherOp, gatherOp.getRef()));
      EntryStorageHelper storageHelper(gatherOp, continuousRefEntryType.getMembers(), continuousRefEntryType.hasLock(), typeConverter);
      auto ptrType = storageHelper.getRefType();
      auto baseRef = rewriter.create<util::BufferGetRef>(gatherOp->getLoc(), ptrType, unPackedReference[1]);
      auto elementRef = rewriter.create<util::ArrayElementPtrOp>(gatherOp->getLoc(), ptrType, baseRef, unPackedReference[0]);
      storageHelper.loadIntoColumns(gatherOp.getMapping(), mapping, elementRef, rewriter, gatherOp->getLoc());
      rewriter.replaceTupleStream(gatherOp, mapping);
      return success();
   }
};

/// Hash map reference gather operation lowering
class HashMapRefGatherOpLowering : public SubOpTupleStreamConsumerConversionPattern<subop::GatherOp, 2> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::GatherOp, 2>::SubOpTupleStreamConsumerConversionPattern;

   LogicalResult matchAndRewrite(subop::GatherOp gatherOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      auto refType = gatherOp.getRef().getColumn().type;
      auto referenceType = mlir::dyn_cast_or_null<subop::HashMapEntryRefType>(refType);
      if (!referenceType) { return failure(); }
      auto keyMembers = referenceType.getHashMap().getKeyMembers();
      auto valMembers = referenceType.getHashMap().getValueMembers();
      auto loc = gatherOp->getLoc();
      EntryStorageHelper keyStorageHelper(gatherOp, keyMembers, false, typeConverter);
      EntryStorageHelper valStorageHelper(gatherOp, valMembers, referenceType.hasLock(), typeConverter);
      auto ref = mapping.resolve(gatherOp, gatherOp.getRef());
      auto keyRef = rewriter.create<util::TupleElementPtrOp>(loc, keyStorageHelper.getRefType(), ref, 0);
      auto valRef = rewriter.create<util::TupleElementPtrOp>(loc, valStorageHelper.getRefType(), ref, 1);
      keyStorageHelper.loadIntoColumns(gatherOp.getMapping(), mapping, keyRef, rewriter, loc);
      valStorageHelper.loadIntoColumns(gatherOp.getMapping(), mapping, valRef, rewriter, loc);
      rewriter.replaceTupleStream(gatherOp, mapping);
      return success();
   }
};

/// Pre-aggregation hash table reference gather operation lowering
class PreAggrHtRefGatherOpLowering : public SubOpTupleStreamConsumerConversionPattern<subop::GatherOp, 2> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::GatherOp, 2>::SubOpTupleStreamConsumerConversionPattern;

   LogicalResult matchAndRewrite(subop::GatherOp gatherOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      auto refType = gatherOp.getRef().getColumn().type;
      auto referenceType = mlir::dyn_cast_or_null<subop::PreAggrHTEntryRefType>(refType);
      if (!referenceType) { return failure(); }
      auto keyMembers = referenceType.getHashMap().getKeyMembers();
      auto valMembers = referenceType.getHashMap().getValueMembers();
      auto loc = gatherOp->getLoc();
      EntryStorageHelper keyStorageHelper(gatherOp, keyMembers, false, typeConverter);
      EntryStorageHelper valStorageHelper(gatherOp, valMembers, referenceType.hasLock(), typeConverter);
      auto ref = mapping.resolve(gatherOp, gatherOp.getRef());
      auto keyRef = rewriter.create<util::TupleElementPtrOp>(loc, keyStorageHelper.getRefType(), ref, 0);
      auto valRef = rewriter.create<util::TupleElementPtrOp>(loc, valStorageHelper.getRefType(), ref, 1);
      keyStorageHelper.loadIntoColumns(gatherOp.getMapping(), mapping, keyRef, rewriter, loc);
      valStorageHelper.loadIntoColumns(gatherOp.getMapping(), mapping, valRef, rewriter, loc);
      rewriter.replaceTupleStream(gatherOp, mapping);
      return success();
   }
};

/// Hash multi-map reference gather operation lowering
class HashMultiMapRefGatherOpLowering : public SubOpTupleStreamConsumerConversionPattern<subop::GatherOp, 2> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::GatherOp, 2>::SubOpTupleStreamConsumerConversionPattern;

   LogicalResult matchAndRewrite(subop::GatherOp gatherOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      auto refType = gatherOp.getRef().getColumn().type;
      if (!mlir::isa<subop::HashMultiMapEntryRefType>(refType)) { return failure(); }
      auto hashMultiMap = mlir::cast<subop::HashMultiMapEntryRefType>(refType).getHashMultimap();
      auto keyMembers = hashMultiMap.getKeyMembers();
      auto valMembers = hashMultiMap.getValueMembers();
      auto loc = gatherOp->getLoc();
      EntryStorageHelper keyStorageHelper(gatherOp, keyMembers, false, typeConverter);
      EntryStorageHelper valStorageHelper(gatherOp, valMembers, hashMultiMap.hasLock(), typeConverter);
      auto packed = mapping.resolve(gatherOp, gatherOp.getRef());
      llvm::SmallVector<mlir::Value> unpacked;
      rewriter.createOrFold<util::UnPackOp>(unpacked, loc, packed);
      keyStorageHelper.loadIntoColumns(gatherOp.getMapping(), mapping, unpacked[0], rewriter, loc);
      valStorageHelper.loadIntoColumns(gatherOp.getMapping(), mapping, unpacked[1], rewriter, loc);
      rewriter.replaceTupleStream(gatherOp, mapping);
      return success();
   }
};

/// External hash index reference gather operation lowering
/// Specialized for PostgreSQL tuple field access
class ExternalHashIndexRefGatherOpLowering : public SubOpTupleStreamConsumerConversionPattern<subop::GatherOp, 2> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::GatherOp, 2>::SubOpTupleStreamConsumerConversionPattern;

   LogicalResult matchAndRewrite(subop::GatherOp gatherOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      auto refType = gatherOp.getRef().getColumn().type;
      if (!mlir::isa<subop::ExternalHashIndexEntryRefType>(refType)) { return failure(); }
      auto columns = mlir::cast<subop::ExternalHashIndexEntryRefType>(refType).getMembers();
      auto tableRefVal = mapping.resolve(gatherOp, gatherOp.getRef());
      llvm::SmallVector<mlir::Value> unPacked;
      rewriter.createOrFold<util::UnPackOp>(unPacked, gatherOp->getLoc(), tableRefVal);
      auto currRow = unPacked[0];
      llvm::SmallVector<mlir::Value> unPackedColumns;
      rewriter.createOrFold<util::UnPackOp>(unPackedColumns, gatherOp->getLoc(), unPacked[1]);
      
      for (size_t i = 0; i < columns.getTypes().size(); i++) {
         auto memberName = mlir::cast<mlir::StringAttr>(columns.getNames()[i]).str();
         if (gatherOp.getMapping().contains(memberName)) {
            auto columnDefAttr = mlir::cast<tuples::ColumnDefAttr>(gatherOp.getMapping().get(memberName));
            auto colArray = unPackedColumns[i];
            auto type = columnDefAttr.getColumn().type;
            // PostgreSQL: Replace Arrow loading with PostgreSQL tuple field access
            // In PostgreSQL context, currRow should be the tuple pointer
            // Field index is i, field name is memberName
            mlir::Value fieldIndex = rewriter.create<arith::ConstantIndexOp>(gatherOp->getLoc(), i);
            mlir::StringAttr fieldNameAttr = rewriter.getStringAttr(memberName);
            mlir::Value loaded = rewriter.create<db::LoadPostgreSQLOp>(gatherOp->getLoc(), type, currRow, fieldIndex, fieldNameAttr);
            mapping.define(columnDefAttr, loaded);
         }
      }
      rewriter.replaceTupleStream(gatherOp, mapping);
      return success();
   }
};

//===----------------------------------------------------------------------===//
// Scatter Operations - Data Storing Patterns
//===----------------------------------------------------------------------===//

/// Continuous reference scatter operation lowering
class ContinuousRefScatterOpLowering : public SubOpTupleStreamConsumerConversionPattern<subop::ScatterOp, 2> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::ScatterOp, 2>::SubOpTupleStreamConsumerConversionPattern;

   LogicalResult matchAndRewrite(subop::ScatterOp scatterOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      if (!checkAtomicStore(scatterOp)) return failure();
      auto continuousRefEntryType = mlir::dyn_cast_or_null<subop::ContinuousEntryRefType>(scatterOp.getRef().getColumn().type);
      if (!continuousRefEntryType) { return failure(); }
      llvm::SmallVector<mlir::Value> unpackedReference;
      rewriter.createOrFold<util::UnPackOp>(unpackedReference, scatterOp->getLoc(), mapping.resolve(scatterOp, scatterOp.getRef()));
      EntryStorageHelper storageHelper(scatterOp, continuousRefEntryType.getMembers(), continuousRefEntryType.hasLock(), typeConverter);
      auto ptrType = storageHelper.getRefType();
      auto baseRef = rewriter.create<util::BufferGetRef>(scatterOp->getLoc(), ptrType, unpackedReference[1]);
      auto elementRef = rewriter.create<util::ArrayElementPtrOp>(scatterOp->getLoc(), ptrType, baseRef, unpackedReference[0]);
      auto values = storageHelper.getValueMap(elementRef, rewriter, scatterOp->getLoc());
      for (auto x : scatterOp.getMapping()) {
         values.set(x.getName().str(), mapping.resolve(scatterOp, mlir::cast<tuples::ColumnRefAttr>(x.getValue())));
      }
      values.store();
      rewriter.eraseOp(scatterOp);
      return success();
   }
};

/// Generic scatter operation lowering for state entry references
class ScatterOpLowering : public SubOpTupleStreamConsumerConversionPattern<subop::ScatterOp, 1> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::ScatterOp>::SubOpTupleStreamConsumerConversionPattern;

   LogicalResult matchAndRewrite(subop::ScatterOp scatterOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      if (!checkAtomicStore(scatterOp)) return failure();
      auto referenceType = mlir::cast<subop::StateEntryReference>(scatterOp.getRef().getColumn().type);
      auto columns = referenceType.getMembers();
      EntryStorageHelper storageHelper(scatterOp, columns, referenceType.hasLock(), typeConverter);
      auto ref = mapping.resolve(scatterOp, scatterOp.getRef());
      auto values = storageHelper.getValueMap(ref, rewriter, scatterOp->getLoc());
      for (auto x : scatterOp.getMapping()) {
         values.set(x.getName().str(), mapping.resolve(scatterOp, mlir::cast<tuples::ColumnRefAttr>(x.getValue())));
      }
      values.store();
      rewriter.eraseOp(scatterOp);
      return success();
   }
};

/// Hash multi-map scatter operation lowering
class HashMultiMapScatterOp : public SubOpTupleStreamConsumerConversionPattern<subop::ScatterOp, 1> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::ScatterOp>::SubOpTupleStreamConsumerConversionPattern;

   LogicalResult matchAndRewrite(subop::ScatterOp scatterOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      auto hashMultiMapEntryRef = mlir::dyn_cast_or_null<subop::HashMultiMapEntryRefType>(scatterOp.getRef().getColumn().type);
      if (!hashMultiMapEntryRef) return failure();
      auto columns = hashMultiMapEntryRef.getHashMultimap().getValueMembers();
      EntryStorageHelper storageHelper(scatterOp, columns, hashMultiMapEntryRef.hasLock(), typeConverter);
      llvm::SmallVector<mlir::Value> unPacked;
      rewriter.createOrFold<util::UnPackOp>(unPacked, scatterOp.getLoc(), mapping.resolve(scatterOp, scatterOp.getRef()));
      auto ref = unPacked[1];
      auto values = storageHelper.getValueMap(ref, rewriter, scatterOp->getLoc());
      for (auto x : scatterOp.getMapping()) {
         values.set(x.getName().str(), mapping.resolve(scatterOp, mlir::cast<tuples::ColumnRefAttr>(x.getValue())));
      }
      values.store();
      rewriter.eraseOp(scatterOp);
      return success();
   }
};

} // namespace subop_to_cf
} // namespace dialect  
} // namespace compiler
} // namespace pgx_lower