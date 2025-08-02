#include "../Headers/SubOpToControlFlowPatterns.h"
#include "../Headers/SubOpToControlFlowUtilities.h"

namespace pgx_lower {
namespace compiler {
namespace dialect {
namespace subop_to_cf {

// Using declarations for utility functions
using subop_to_control_flow::implementBufferIteration;
using subop_to_control_flow::implementBufferIterationRuntime;
using subop_to_control_flow::getHtKVType;
using subop_to_control_flow::getHtEntryType;
using subop_to_control_flow::getHashMultiMapEntryType;
using subop_to_control_flow::getHashMultiMapValueType;
using subop_to_control_flow::hashKeys;
using subop_to_control_flow::unpackTypes;
using subop_to_control_flow::inlineBlock;

//===----------------------------------------------------------------------===//
// Simple State Lookup Operation
//===----------------------------------------------------------------------===//

class LookupSimpleStateLowering : public SubOpTupleStreamConsumerConversionPattern<subop::LookupOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::LookupOp>::SubOpTupleStreamConsumerConversionPattern;
   LogicalResult matchAndRewrite(subop::LookupOp lookupOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      if (!mlir::isa<subop::SimpleStateType>(lookupOp.getState().getType())) return failure();
      mapping.define(lookupOp.getRef(), adaptor.getState());
      rewriter.replaceTupleStream(lookupOp, mapping);
      return mlir::success();
   }
};

//===----------------------------------------------------------------------===//
// Hash Indexed View Lookup Operation
//===----------------------------------------------------------------------===//

class LookupHashIndexedViewLowering : public SubOpTupleStreamConsumerConversionPattern<subop::LookupOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::LookupOp>::SubOpTupleStreamConsumerConversionPattern;
   LogicalResult matchAndRewrite(subop::LookupOp lookupOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      if (!mlir::isa<subop::HashIndexedViewType>(lookupOp.getState().getType())) return failure();
      auto loc = lookupOp->getLoc();
      mlir::Value hash = mapping.resolve(lookupOp, lookupOp.getKeys())[0];
      auto* context = getContext();
      auto indexType = rewriter.getIndexType();
      auto htType = util::RefType::get(context, util::RefType::get(context, rewriter.getI8Type()));

      Value castedPointer = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(context, TupleType::get(context, {htType, indexType})), adaptor.getState());

      auto loaded = rewriter.create<util::LoadOp>(loc, mlir::cast<util::RefType>(castedPointer.getType()).getElementType(), castedPointer, Value());
      auto unpacked = rewriter.create<util::UnPackOp>(loc, loaded);
      Value ht = unpacked.getResult(0);
      Value htMask = unpacked.getResult(1);
      Value buckedPos = rewriter.create<arith::AndIOp>(loc, htMask, hash);
      Value ptr = rewriter.create<util::LoadOp>(loc, util::RefType::get(getContext(), rewriter.getI8Type()), ht, buckedPos);
      //optimization
      Value refValid = rewriter.create<util::PtrTagMatches>(loc, rewriter.getI1Type(), ptr, hash);
      ptr = rewriter.create<util::UnTagPtr>(loc, ptr.getType(), ptr);
      Value matches = rewriter.create<util::PackOp>(loc, ValueRange{ptr, hash, refValid});

      mapping.define(lookupOp.getRef(), matches);
      rewriter.replaceTupleStream(lookupOp, mapping);
      return mlir::success();
   }
};

//===----------------------------------------------------------------------===//
// Segment Tree View Lookup Operation
//===----------------------------------------------------------------------===//

class LookupSegmentTreeViewLowering : public SubOpTupleStreamConsumerConversionPattern<subop::LookupOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::LookupOp>::SubOpTupleStreamConsumerConversionPattern;
   LogicalResult matchAndRewrite(subop::LookupOp lookupOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      if (!mlir::isa<subop::SegmentTreeViewType>(lookupOp.getState().getType())) return failure();

      auto valueMembers = mlir::cast<subop::SegmentTreeViewType>(lookupOp.getState().getType()).getValueMembers();
      mlir::TupleType stateType = mlir::TupleType::get(getContext(), unpackTypes(valueMembers.getTypes()));

      auto loc = lookupOp->getLoc();
      llvm::SmallVector<mlir::Value> unpackedLeft;
      llvm::SmallVector<mlir::Value> unpackedRight;

      rewriter.createOrFold<util::UnPackOp>(unpackedLeft, loc, mapping.resolve(lookupOp, lookupOp.getKeys())[0]);
      rewriter.createOrFold<util::UnPackOp>(unpackedRight, loc, mapping.resolve(lookupOp, lookupOp.getKeys())[1]);
      auto idxLeft = unpackedLeft[0];
      auto idxRight = unpackedRight[0];
      mlir::Value ref;
      rewriter.atStartOf(&rewriter.getCurrentStreamLoc()->getParentOfType<mlir::func::FuncOp>().getFunctionBody().front(), [&](SubOpRewriter& rewriter) {
         ref = rewriter.create<util::AllocaOp>(lookupOp->getLoc(), util::RefType::get(typeConverter->convertType(stateType)), mlir::Value());
      });
      rt::SegmentTreeView::lookup(rewriter, loc)({adaptor.getState(), ref, idxLeft, idxRight});
      mapping.define(lookupOp.getRef(), ref);
      rewriter.replaceTupleStream(lookupOp, mapping);
      return mlir::success();
   }
};

//===----------------------------------------------------------------------===//
// Pure Hash Map Lookup Operation
//===----------------------------------------------------------------------===//

class PureLookupHashMapLowering : public SubOpTupleStreamConsumerConversionPattern<subop::LookupOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::LookupOp>::SubOpTupleStreamConsumerConversionPattern;
   LogicalResult matchAndRewrite(subop::LookupOp lookupOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      if (!mlir::isa<subop::HashMapType>(lookupOp.getState().getType())) return failure();
      subop::HashMapType htStateType = mlir::cast<subop::HashMapType>(lookupOp.getState().getType());
      EntryStorageHelper keyStorageHelper(lookupOp, htStateType.getKeyMembers(), false, typeConverter);
      EntryStorageHelper valStorageHelper(lookupOp, htStateType.getValueMembers(), htStateType.hasLock(), typeConverter);
      auto lookupKey = mapping.resolve(lookupOp, lookupOp.getKeys());
      mlir::Value hash = hashKeys(lookupKey, rewriter, lookupOp->getLoc());
      auto equalFnBuilder = [&lookupOp](mlir::OpBuilder& rewriter, EntryStorageHelper::LazyValueMap left, std::vector<Value> right) -> Value {
         std::vector<mlir::Value> arguments;
         for (const auto& pair : left) { arguments.push_back(pair.second); }
         for (const auto& value : right) { arguments.push_back(value); }
         auto res = inlineBlock(&lookupOp.getEqFn().front(), rewriter, arguments);
         return res[0];
      };
      auto loc = lookupOp->getLoc();
      Value hashed = hash;
      auto* context = rewriter.getContext();
      auto entryType = getHtEntryType(htStateType, *typeConverter);
      auto i8PtrType = util::RefType::get(context, IntegerType::get(context, 8));
      auto i8PtrPtrType = util::RefType::get(context, i8PtrType);

      auto idxType = rewriter.getIndexType();
      auto idxPtrType = util::RefType::get(rewriter.getContext(), idxType);
      auto kvType = getHtKVType(htStateType, *typeConverter);
      auto kvPtrType = util::RefType::get(context, kvType);
      auto keyPtrType = keyStorageHelper.getRefType();
      auto entryPtrType = util::RefType::get(context, entryType);
      auto htType = util::RefType::get(context, entryPtrType);

      Value castedState = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(getContext(), mlir::TupleType::get(getContext(), {htType, rewriter.getIndexType()})), adaptor.getState());
      Value htAddress = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(rewriter.getContext(), htType), castedState, 0);
      Value htMaskAddress = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(rewriter.getContext(), rewriter.getIndexType()), castedState, 1);
      Value ht = rewriter.create<util::LoadOp>(loc, htType, htAddress);
      Value htMask = rewriter.create<util::LoadOp>(loc, rewriter.getIndexType(), htMaskAddress);

      Value trueValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
      Value falseValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));

      //position = hash & hashTableMask
      Value position = rewriter.create<arith::AndIOp>(loc, htMask, hashed);
      // ptr = &hashtable[position]
      Type bucketPtrType = util::RefType::get(context, entryType);
      Value ptr = rewriter.create<util::LoadOp>(loc, bucketPtrType, ht, position);
      Value tagMatches = rewriter.create<util::PtrTagMatches>(loc, rewriter.getI1Type(), ptr, hashed);
      ptr = rewriter.create<util::UnTagPtr>(loc, ptr.getType(), ptr);
      auto ifOpOuter = rewriter.create<mlir::scf::IfOp>(
         loc, tagMatches, [&](mlir::OpBuilder& builder1, mlir::Location loc) {
            auto whileOp = rewriter.create<scf::WhileOp>(loc, bucketPtrType, ValueRange({ptr}));
            Block* before = new Block;
            whileOp.getBefore().push_back(before);
            before->addArgument(bucketPtrType, loc);
            Block* after = new Block;
            whileOp.getAfter().push_back(after);
            after->addArgument(bucketPtrType, loc);
            rewriter.atStartOf(before, [&](SubOpRewriter& rewriter) {
               Value currEntryPtr = before->getArgument(0);
               Value ptr = currEntryPtr;
               //    if (*ptr != nullptr){
               Value cmp = rewriter.create<util::IsRefValidOp>(loc, rewriter.getI1Type(), currEntryPtr);
               auto ifOp = rewriter.create<scf::IfOp>(
                  loc, cmp, [&](OpBuilder& b, Location loc) {
               Value hashAddress=rewriter.create<util::TupleElementPtrOp>(loc, idxPtrType, currEntryPtr, 1);
               Value entryHash = rewriter.create<util::LoadOp>(loc, idxType, hashAddress);
               Value hashMatches = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, entryHash,hashed);
               auto ifOpH = b.create<scf::IfOp>(
                  loc, hashMatches, [&](OpBuilder& b, Location loc) {
                     Value kvAddress = rewriter.create<util::TupleElementPtrOp>(loc, kvPtrType, currEntryPtr, 2);
                     Value keyRef = rewriter.create<util::TupleElementPtrOp>(loc, keyPtrType, kvAddress, 0);
                     auto keyValues=keyStorageHelper.getValueMap(keyRef,rewriter,loc);
                     Value keyMatches = equalFnBuilder(b, keyValues, lookupKey);
                     auto ifOp2 = b.create<scf::IfOp>(
                        loc, keyMatches, [&](OpBuilder& b, Location loc) {


                           b.create<scf::YieldOp>(loc, ValueRange{falseValue, ptr}); }, [&](OpBuilder& b, Location loc) {
                           Value nextPtrAddr=rewriter.create<util::TupleElementPtrOp>(loc, i8PtrPtrType, currEntryPtr, 0);
                           //          ptr = &entry.next
                           Value newEntryPtr=b.create<util::LoadOp>(loc,nextPtrAddr,mlir::Value());
                           Value newPtr=b.create<util::GenericMemrefCastOp>(loc, bucketPtrType,newEntryPtr);
                           //          yield ptr,done=false
                           b.create<scf::YieldOp>(loc, ValueRange{trueValue, newPtr }); });
                     b.create<scf::YieldOp>(loc, ifOp2.getResults());
                  }, [&](OpBuilder& b, Location loc) {
                     Value nextPtrAddr=rewriter.create<util::TupleElementPtrOp>(loc, i8PtrPtrType, currEntryPtr, 0);
                     //          ptr = &entry.next
                     Value newEntryPtr=b.create<util::LoadOp>(loc,nextPtrAddr,mlir::Value());
                     Value newPtr=b.create<util::GenericMemrefCastOp>(loc, bucketPtrType,newEntryPtr);
                     //          yield ptr,done=false
                     b.create<scf::YieldOp>(loc, ValueRange{trueValue, newPtr });});
               b.create<scf::YieldOp>(loc, ifOpH.getResults()); }, [&](OpBuilder& b, Location loc) { b.create<scf::YieldOp>(loc, ValueRange{falseValue, ptr}); });

               Value done = ifOp.getResult(0);
               Value newPtr = ifOp.getResult(1);
               rewriter.create<scf::ConditionOp>(loc, done, ValueRange({newPtr}));
            });
            rewriter.atStartOf(after, [&](SubOpRewriter& rewriter) {
               rewriter.create<scf::YieldOp>(loc, after->getArguments());
            });
            rewriter.create<scf::YieldOp>(loc, ValueRange{whileOp.getResult(0)}); },
         [&](mlir::OpBuilder& builder1, mlir::Location loc) {
            Value invalidPtr = rewriter.create<util::InvalidRefOp>(loc, bucketPtrType);
            rewriter.create<scf::YieldOp>(loc, ValueRange{invalidPtr});
         });

      Value currEntryPtr = ifOpOuter.getResult(0);
      currEntryPtr = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(getContext(), rewriter.getI8Type()), currEntryPtr);
      mapping.define(lookupOp.getRef(), currEntryPtr);
      rewriter.replaceTupleStream(lookupOp, mapping);
      return mlir::success();
   }
};

//===----------------------------------------------------------------------===//
// Pure Pre-Aggregation Hash Table Lookup Operation
//===----------------------------------------------------------------------===//

class PureLookupPreAggregationHtLowering : public SubOpTupleStreamConsumerConversionPattern<subop::LookupOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::LookupOp>::SubOpTupleStreamConsumerConversionPattern;
   LogicalResult matchAndRewrite(subop::LookupOp lookupOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      if (!mlir::isa<subop::PreAggrHtType>(lookupOp.getState().getType())) return failure();
      subop::PreAggrHtType htStateType = mlir::cast<subop::PreAggrHtType>(lookupOp.getState().getType());
      EntryStorageHelper keyStorageHelper(lookupOp, htStateType.getKeyMembers(), false, typeConverter);
      EntryStorageHelper valStorageHelper(lookupOp, htStateType.getValueMembers(), htStateType.hasLock(), typeConverter);
      auto lookupKey = mapping.resolve(lookupOp, lookupOp.getKeys());

      mlir::Value hash = hashKeys(lookupKey, rewriter, lookupOp->getLoc());
      ASSERT_WITH_OP(!lookupOp.getEqFn().empty(), lookupOp, "LookupOp must have an equality function");
      auto equalFnBuilder = [&lookupOp](mlir::OpBuilder& rewriter, EntryStorageHelper::LazyValueMap left, std::vector<Value> right) -> Value {
         std::vector<mlir::Value> arguments;
         for (const auto& pair : left) { arguments.push_back(pair.second); }
         for (const auto& value : right) { arguments.push_back(value); }
         auto res = inlineBlock(&lookupOp.getEqFn().front(), rewriter, arguments);
         return res[0];
      };
      auto loc = lookupOp->getLoc();
      Value hashed = hash;
      auto* context = rewriter.getContext();
      auto entryType = getHtEntryType(htStateType, *typeConverter);
      auto i8PtrType = util::RefType::get(context, IntegerType::get(context, 8));
      auto i8PtrPtrType = util::RefType::get(context, i8PtrType);

      auto idxType = rewriter.getIndexType();
      auto idxPtrType = util::RefType::get(rewriter.getContext(), idxType);
      auto kvType = getHtKVType(htStateType, *typeConverter);
      auto kvPtrType = util::RefType::get(context, kvType);
      auto keyPtrType = keyStorageHelper.getRefType();
      Type bucketPtrType = util::RefType::get(context, entryType);

      mlir::Value partition = rewriter.create<mlir::arith::AndIOp>(loc, hashed, rewriter.create<mlir::arith::ConstantIndexOp>(loc, 64 - 1));
      Type partitionHtType = mlir::TupleType::get(rewriter.getContext(), {util::RefType::get(context, bucketPtrType), rewriter.getIndexType()});
      Value preaggregationHt = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(context, partitionHtType), adaptor.getState());
      Value partitionHt = rewriter.create<util::LoadOp>(loc, preaggregationHt, partition);
      auto unpacked = rewriter.create<util::UnPackOp>(loc, partitionHt).getResults();
      Value ht = unpacked[0];
      Value htMask = unpacked[1];
      Value position = rewriter.create<arith::AndIOp>(loc, htMask, rewriter.create<arith::ShRUIOp>(loc, hashed, rewriter.create<mlir::arith::ConstantIndexOp>(loc, 6)));
      Value ptr = rewriter.create<util::LoadOp>(loc, bucketPtrType, ht, position);
      Value tagMatches = rewriter.create<util::PtrTagMatches>(loc, rewriter.getI1Type(), ptr, hashed);
      ptr = rewriter.create<util::UnTagPtr>(loc, ptr.getType(), ptr);
      Value trueValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
      Value falseValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
      auto ifOpOuter = rewriter.create<mlir::scf::IfOp>(
         loc, tagMatches, [&](mlir::OpBuilder& builder1, mlir::Location loc) {
            auto whileOp = rewriter.create<scf::WhileOp>(loc, bucketPtrType, ValueRange({ptr}));
            Block* before = new Block;
            whileOp.getBefore().push_back(before);
            before->addArgument(bucketPtrType, loc);
            Block* after = new Block;
            whileOp.getAfter().push_back(after);
            after->addArgument(bucketPtrType, loc);
            rewriter.atStartOf(before, [&](SubOpRewriter& rewriter) {
               Value currEntryPtr = before->getArgument(0);
               Value ptr = currEntryPtr;
               //    if (*ptr != nullptr){
               Value cmp = rewriter.create<util::IsRefValidOp>(loc, rewriter.getI1Type(), currEntryPtr);
               auto ifOp = rewriter.create<scf::IfOp>(
                  loc, cmp, [&](OpBuilder& b, Location loc) {
                     Value hashAddress=rewriter.create<util::TupleElementPtrOp>(loc, idxPtrType, currEntryPtr, 1);
                     Value entryHash = rewriter.create<util::LoadOp>(loc, idxType, hashAddress);
                     Value hashMatches = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, entryHash,hashed);
                     auto ifOpH = b.create<scf::IfOp>(
                        loc, hashMatches, [&](OpBuilder& b, Location loc) {
                           Value kvAddress = rewriter.create<util::TupleElementPtrOp>(loc, kvPtrType, currEntryPtr, 2);
                           Value keyRef = rewriter.create<util::TupleElementPtrOp>(loc, keyPtrType, kvAddress, 0);
                           auto keyValues=keyStorageHelper.getValueMap(keyRef,rewriter,loc);
                           Value keyMatches = equalFnBuilder(b, keyValues, lookupKey);
                           auto ifOp2 = b.create<scf::IfOp>(
                              loc, keyMatches, [&](OpBuilder& b, Location loc) {


                                 b.create<scf::YieldOp>(loc, ValueRange{falseValue, ptr}); }, [&](OpBuilder& b, Location loc) {
                                 Value nextPtrAddr=rewriter.create<util::TupleElementPtrOp>(loc, i8PtrPtrType, currEntryPtr, 0);
                                 //          ptr = &entry.next
                                 Value newEntryPtr=b.create<util::LoadOp>(loc,nextPtrAddr,mlir::Value());
                                 Value newPtr=b.create<util::GenericMemrefCastOp>(loc, bucketPtrType,newEntryPtr);
                                 //          yield ptr,done=false
                                 b.create<scf::YieldOp>(loc, ValueRange{trueValue, newPtr }); });
                           b.create<scf::YieldOp>(loc, ifOp2.getResults());
                        }, [&](OpBuilder& b, Location loc) {
                           Value nextPtrAddr=rewriter.create<util::TupleElementPtrOp>(loc, i8PtrPtrType, currEntryPtr, 0);
                           //          ptr = &entry.next
                           Value newEntryPtr=b.create<util::LoadOp>(loc,nextPtrAddr,mlir::Value());
                           Value newPtr=b.create<util::GenericMemrefCastOp>(loc, bucketPtrType,newEntryPtr);
                           //          yield ptr,done=false
                           b.create<scf::YieldOp>(loc, ValueRange{trueValue, newPtr });});
                     b.create<scf::YieldOp>(loc, ifOpH.getResults()); }, [&](OpBuilder& b, Location loc) { b.create<scf::YieldOp>(loc, ValueRange{falseValue, ptr}); });

               Value done = ifOp.getResult(0);
               Value newPtr = ifOp.getResult(1);
               rewriter.create<scf::ConditionOp>(loc, done, ValueRange({newPtr}));
            });
            rewriter.atStartOf(after, [&](SubOpRewriter& rewriter) {
               rewriter.create<scf::YieldOp>(loc, after->getArguments());
            });
            rewriter.create<scf::YieldOp>(loc, ValueRange{whileOp.getResult(0)}); },
         [&](mlir::OpBuilder& builder1, mlir::Location loc) {
            Value invalidPtr = rewriter.create<util::InvalidRefOp>(loc, bucketPtrType);
            rewriter.create<scf::YieldOp>(loc, ValueRange{invalidPtr});
         });

      Value currEntryPtr = ifOpOuter.getResult(0);
      currEntryPtr = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(getContext(), rewriter.getI8Type()), currEntryPtr);
      mapping.define(lookupOp.getRef(), currEntryPtr);
      rewriter.replaceTupleStream(lookupOp, mapping);
      return mlir::success();
   }
};

//===----------------------------------------------------------------------===//
// Hash Multi-Map Lookup Operation
//===----------------------------------------------------------------------===//

class LookupHashMultiMapLowering : public SubOpTupleStreamConsumerConversionPattern<subop::LookupOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::LookupOp>::SubOpTupleStreamConsumerConversionPattern;
   LogicalResult matchAndRewrite(subop::LookupOp lookupOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      subop::HashMultiMapType hashMultiMapType = mlir::dyn_cast_or_null<subop::HashMultiMapType>(lookupOp.getState().getType());
      if (!hashMultiMapType) return failure();
      EntryStorageHelper keyStorageHelper(lookupOp, hashMultiMapType.getKeyMembers(), false, typeConverter);
      EntryStorageHelper valStorageHelper(lookupOp, hashMultiMapType.getValueMembers(), hashMultiMapType.hasLock(), typeConverter);
      auto lookupKey = mapping.resolve(lookupOp, lookupOp.getKeys());

      mlir::Value hash = hashKeys(lookupKey, rewriter, lookupOp->getLoc());
      auto equalFnBuilder = [&lookupOp](mlir::OpBuilder& rewriter, EntryStorageHelper::LazyValueMap left, std::vector<Value> right) -> Value {
         std::vector<mlir::Value> arguments;
         for (const auto& pair : left) { arguments.push_back(pair.second); }
         for (const auto& value : right) { arguments.push_back(value); }
         auto res = inlineBlock(&lookupOp.getEqFn().front(), rewriter, arguments);
         return res[0];
      };
      auto loc = lookupOp->getLoc();
      Value hashed = hash;
      auto* context = rewriter.getContext();
      auto entryType = getHashMultiMapEntryType(hashMultiMapType, *typeConverter);
      auto i8PtrType = util::RefType::get(context, IntegerType::get(context, 8));
      auto i8PtrPtrType = util::RefType::get(context, i8PtrType);

      auto idxType = rewriter.getIndexType();
      auto idxPtrType = util::RefType::get(rewriter.getContext(), idxType);
      auto keyPtrType = keyStorageHelper.getRefType();
      auto entryPtrType = util::RefType::get(context, entryType);
      auto htType = util::RefType::get(context, entryPtrType);

      Value castedState = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(getContext(), mlir::TupleType::get(getContext(), {htType, rewriter.getIndexType()})), adaptor.getState());
      Value htAddress = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(rewriter.getContext(), htType), castedState, 0);
      Value htMaskAddress = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(rewriter.getContext(), rewriter.getIndexType()), castedState, 1);
      Value ht = rewriter.create<util::LoadOp>(loc, htType, htAddress);
      Value htMask = rewriter.create<util::LoadOp>(loc, rewriter.getIndexType(), htMaskAddress);

      Value trueValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
      Value falseValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));

      //position = hash & hashTableMask
      Value position = rewriter.create<arith::AndIOp>(loc, htMask, hashed);
      // ptr = &hashtable[position]
      Type bucketPtrType = util::RefType::get(context, entryType);
      Value ptr = rewriter.create<util::LoadOp>(loc, bucketPtrType, ht, position);
      Value tagMatches = rewriter.create<util::PtrTagMatches>(loc, rewriter.getI1Type(), ptr, hashed);
      ptr = rewriter.create<util::UnTagPtr>(loc, ptr.getType(), ptr);
      auto ifOpOuter = rewriter.create<mlir::scf::IfOp>(
         loc, tagMatches, [&](mlir::OpBuilder& builder1, mlir::Location loc) {

      auto whileOp = rewriter.create<scf::WhileOp>(loc, bucketPtrType, ValueRange({ptr}));
      Block* before = new Block;
      whileOp.getBefore().push_back(before);
      before->addArgument(bucketPtrType, loc);
      Block* after = new Block;
      whileOp.getAfter().push_back(after);
      after->addArgument(bucketPtrType, loc);
      rewriter.atStartOf(before, [&](SubOpRewriter& rewriter) {
         Value ptr = before->getArgument(0);

         Value currEntryPtr = ptr;
         //    if (*ptr != nullptr){
         Value cmp = rewriter.create<util::IsRefValidOp>(loc, rewriter.getI1Type(), currEntryPtr);
         auto ifOp = rewriter.create<scf::IfOp>(
            loc, cmp, [&](OpBuilder& b, Location loc) {
               Value hashAddress=rewriter.create<util::TupleElementPtrOp>(loc, idxPtrType, currEntryPtr, 1);
               Value entryHash = rewriter.create<util::LoadOp>(loc, idxType, hashAddress);
               Value hashMatches = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, entryHash,hashed);
               auto ifOpH = b.create<scf::IfOp>(
                  loc, hashMatches, [&](OpBuilder& b, Location loc) {
                     Value keyRef = rewriter.create<util::TupleElementPtrOp>(loc, keyPtrType, currEntryPtr, 3);
                     auto keyValues=keyStorageHelper.getValueMap(keyRef,rewriter,loc);
                     Value keyMatches = equalFnBuilder(b, keyValues, lookupKey);
                     auto ifOp2 = b.create<scf::IfOp>(
                        loc,  keyMatches, [&](OpBuilder& b, Location loc) {
                           b.create<scf::YieldOp>(loc, ValueRange{falseValue, ptr}); }, [&](OpBuilder& b, Location loc) {
                           Value nextPtrAddr=rewriter.create<util::TupleElementPtrOp>(loc, i8PtrPtrType, currEntryPtr, 0);
                           //          ptr = &entry.next
                           Value newEntryPtr=b.create<util::LoadOp>(loc,nextPtrAddr,mlir::Value());
                           Value newPtr=b.create<util::GenericMemrefCastOp>(loc, bucketPtrType,newEntryPtr);
                           //          yield ptr,done=false
                           b.create<scf::YieldOp>(loc, ValueRange{trueValue, newPtr }); });
                     b.create<scf::YieldOp>(loc, ifOp2.getResults());
                  }, [&](OpBuilder& b, Location loc) {
                     Value nextPtrAddr=rewriter.create<util::TupleElementPtrOp>(loc, i8PtrPtrType, currEntryPtr, 0);
                     //          ptr = &entry.next
                     Value newEntryPtr=b.create<util::LoadOp>(loc,nextPtrAddr,mlir::Value());
                     Value newPtr=b.create<util::GenericMemrefCastOp>(loc, bucketPtrType,newEntryPtr);
                     //          yield ptr,done=false
                     b.create<scf::YieldOp>(loc, ValueRange{trueValue, newPtr });});
               b.create<scf::YieldOp>(loc, ifOpH.getResults()); }, [&](OpBuilder& b, Location loc) { b.create<scf::YieldOp>(loc, ValueRange{falseValue, ptr}); });

         Value done = ifOp.getResult(0);
         Value newPtr = ifOp.getResult(1);
         rewriter.create<scf::ConditionOp>(loc, done, ValueRange({newPtr}));
      });
      rewriter.atStartOf(after, [&](SubOpRewriter& rewriter) {
         rewriter.create<scf::YieldOp>(loc, after->getArguments());
      });
            rewriter.create<scf::YieldOp>(loc, ValueRange{whileOp.getResult(0)}); },
         [&](mlir::OpBuilder& builder1, mlir::Location loc) {
            Value invalidPtr = rewriter.create<util::InvalidRefOp>(loc, bucketPtrType);
            rewriter.create<scf::YieldOp>(loc, ValueRange{invalidPtr});
         });

      Value currEntryPtr = ifOpOuter.getResult(0);
      mapping.define(lookupOp.getRef(), currEntryPtr);
      rewriter.replaceTupleStream(lookupOp, mapping);
      return mlir::success();
   }
};

//===----------------------------------------------------------------------===//
// Multi-Map Insert Operation
//===----------------------------------------------------------------------===//

class InsertMultiMapLowering : public SubOpTupleStreamConsumerConversionPattern<subop::InsertOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::InsertOp>::SubOpTupleStreamConsumerConversionPattern;
   LogicalResult matchAndRewrite(subop::InsertOp insertOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      subop::HashMultiMapType htStateType = mlir::dyn_cast_or_null<subop::HashMultiMapType>(insertOp.getState().getType());
      if (!htStateType) return failure();
      EntryStorageHelper keyStorageHelper(insertOp, htStateType.getKeyMembers(), false, typeConverter);
      EntryStorageHelper valStorageHelper(insertOp, htStateType.getValueMembers(), htStateType.hasLock(), typeConverter);
      auto loc = insertOp->getLoc();
      std::vector<mlir::Value> lookupKey = keyStorageHelper.resolve(insertOp, insertOp.getMapping(), mapping);

      mlir::Value hash = hashKeys(lookupKey, rewriter, loc);
      mlir::Value hashTable = adaptor.getState();
      auto equalFnBuilder = [&insertOp](mlir::OpBuilder& rewriter, EntryStorageHelper::LazyValueMap left, std::vector<Value> right) -> Value {
         std::vector<mlir::Value> arguments;
         for (const auto& pair : left) { arguments.push_back(pair.second); }
         for (const auto& value : right) { arguments.push_back(value); }
         auto res = inlineBlock(&insertOp.getEqFn().front(), rewriter, arguments);
         return res[0];
      };
      Value hashed = hash;

      auto* context = rewriter.getContext();
      auto entryType = getHashMultiMapEntryType(htStateType, *typeConverter);
      auto i8PtrType = util::RefType::get(context, IntegerType::get(context, 8));
      auto i8PtrPtrType = util::RefType::get(context, i8PtrType);

      auto idxType = rewriter.getIndexType();
      auto idxPtrType = util::RefType::get(rewriter.getContext(), idxType);
      auto keyPtrType = keyStorageHelper.getRefType();
      auto entryPtrType = util::RefType::get(context, entryType);
      auto htType = util::RefType::get(context, entryPtrType);

      Value castedState = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(getContext(), mlir::TupleType::get(getContext(), {htType, rewriter.getIndexType()})), adaptor.getState());
      Value htAddress = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(rewriter.getContext(), htType), castedState, 0);
      Value htMaskAddress = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(rewriter.getContext(), rewriter.getIndexType()), castedState, 1);
      Value ht = rewriter.create<util::LoadOp>(loc, htType, htAddress);
      Value htMask = rewriter.create<util::LoadOp>(loc, rewriter.getIndexType(), htMaskAddress);

      Value trueValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
      Value falseValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));

      //position = hash & hashTableMask
      Value position = rewriter.create<arith::AndIOp>(loc, htMask, hashed);
      // ptr = &hashtable[position]
      Type bucketPtrType = util::RefType::get(context, entryType);
      Value firstPtr = rewriter.create<util::LoadOp>(loc, bucketPtrType, ht, position);
      Value tagMatches = rewriter.create<util::PtrTagMatches>(loc, rewriter.getI1Type(), firstPtr, hashed);
      firstPtr = rewriter.create<arith::SelectOp>(loc, tagMatches, rewriter.create<util::UnTagPtr>(loc, firstPtr.getType(), firstPtr), rewriter.create<util::InvalidRefOp>(loc, bucketPtrType));

      auto whileOp = rewriter.create<scf::WhileOp>(loc, bucketPtrType, ValueRange({firstPtr}));
      Block* before = new Block;
      before->addArgument(bucketPtrType, loc);
      whileOp.getBefore().push_back(before);
      Block* after = new Block;
      after->addArgument(bucketPtrType, loc);
      whileOp.getAfter().push_back(after);

      rewriter.atStartOf(before, [&](SubOpRewriter& rewriter) {
         Value ptr = before->getArgument(0);

         Value currEntryPtr = ptr;
         //    if (*ptr != nullptr){
         Value cmp = rewriter.create<util::IsRefValidOp>(loc, rewriter.getI1Type(), currEntryPtr);
         auto ifOp = rewriter.create<scf::IfOp>(
            loc, cmp, [&](OpBuilder& b, Location loc) {
               Value hashAddress=rewriter.create<util::TupleElementPtrOp>(loc, idxPtrType, currEntryPtr, 1);
               Value entryHash = rewriter.create<util::LoadOp>(loc, idxType, hashAddress);
               Value hashMatches = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, entryHash,hashed);
               auto ifOpH = b.create<scf::IfOp>(
                  loc, hashMatches, [&](OpBuilder& b, Location loc) {
                     Value keyRef=rewriter.create<util::TupleElementPtrOp>(loc, keyPtrType, currEntryPtr, 3);
                     auto keyValues=keyStorageHelper.getValueMap(keyRef,rewriter,loc);
                     Value keyMatches = equalFnBuilder(b, keyValues, lookupKey);
                     auto ifOp2 = b.create<scf::IfOp>(
                        loc, keyMatches, [&](OpBuilder& b, Location loc) {
                           Value valRef=rt::HashMultiMap::insertValue(rewriter,loc)({hashTable,currEntryPtr})[0];
                           valRef=rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(getContext(),mlir::TupleType::get(getContext(),{i8PtrType,valStorageHelper.getStorageType()})),valRef);
                           valRef=rewriter.create<util::TupleElementPtrOp>(loc, valStorageHelper.getRefType(), valRef, 1);
                           valStorageHelper.storeFromColumns(insertOp.getMapping(),mapping,valRef,rewriter,loc);
                           b.create<scf::YieldOp>(loc, ValueRange{falseValue, ptr}); }, [&](OpBuilder& b, Location loc) {
                           Value nextPtrAddr=rewriter.create<util::TupleElementPtrOp>(loc, i8PtrPtrType, currEntryPtr, 0);
                           //          ptr = &entry.next
                           Value newEntryPtr=b.create<util::LoadOp>(loc,nextPtrAddr,mlir::Value());
                           Value newPtr=b.create<util::GenericMemrefCastOp>(loc, bucketPtrType,newEntryPtr);
                           //          yield ptr,done=false
                           b.create<scf::YieldOp>(loc, ValueRange{trueValue, newPtr }); });
                     b.create<scf::YieldOp>(loc, ifOp2.getResults());
                  }, [&](OpBuilder& b, Location loc) {

                     Value nextPtrAddr=rewriter.create<util::TupleElementPtrOp>(loc, i8PtrPtrType, currEntryPtr, 0);
                     //          ptr = &entry.next
                     Value newEntryPtr=b.create<util::LoadOp>(loc,nextPtrAddr,mlir::Value());
                     Value newPtr=b.create<util::GenericMemrefCastOp>(loc, bucketPtrType,newEntryPtr);
                     //          yield ptr,done=false
                     b.create<scf::YieldOp>(loc, ValueRange{trueValue, newPtr });});
               b.create<scf::YieldOp>(loc, ifOpH.getResults()); }, [&](OpBuilder& b, Location loc) {
               Value entryRef=rt::HashMultiMap::insertEntry(b,loc)({hashTable,hashed})[0];
               Value entryRefCasted= rewriter.create<util::GenericMemrefCastOp>(loc, bucketPtrType, entryRef);
               Value keyRef=rewriter.create<util::TupleElementPtrOp>(loc, keyPtrType, entryRefCasted, 3);

               keyStorageHelper.storeFromColumns(insertOp.getMapping(),mapping,keyRef,rewriter,loc);
               Value valRef=rt::HashMultiMap::insertValue(rewriter,loc)({hashTable,entryRef})[0];
               valRef=rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(getContext(),mlir::TupleType::get(getContext(),{i8PtrType,valStorageHelper.getStorageType()})),valRef);
               valRef=rewriter.create<util::TupleElementPtrOp>(loc, valStorageHelper.getRefType(), valRef, 1);
               valStorageHelper.storeFromColumns(insertOp.getMapping(),mapping,valRef,rewriter,loc);
               b.create<scf::YieldOp>(loc, ValueRange{falseValue, entryRefCasted}); });
         //       if(compare(entry.key,key)){

         Value done = ifOp.getResult(0);
         Value newPtr = ifOp.getResult(1);
         rewriter.create<scf::ConditionOp>(loc, done, ValueRange({newPtr}));
      });
      rewriter.atStartOf(after, [&](SubOpRewriter& rewriter) {
         rewriter.create<scf::YieldOp>(loc, after->getArguments());
      });
      rewriter.eraseOp(insertOp);
      return mlir::success();
   }
};

//===----------------------------------------------------------------------===//
// Pre-Aggregation Hash Table Fragment Lookup Operation
//===----------------------------------------------------------------------===//

class LookupPreAggrHtFragment : public SubOpTupleStreamConsumerConversionPattern<subop::LookupOrInsertOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::LookupOrInsertOp>::SubOpTupleStreamConsumerConversionPattern;
   LogicalResult matchAndRewrite(subop::LookupOrInsertOp lookupOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      if (!mlir::isa<subop::PreAggrHtFragmentType>(lookupOp.getState().getType())) return failure();
      subop::PreAggrHtFragmentType fragmentType = mlir::cast<subop::PreAggrHtFragmentType>(lookupOp.getState().getType());
      EntryStorageHelper keyStorageHelper(lookupOp, fragmentType.getKeyMembers(), false, typeConverter);
      EntryStorageHelper valStorageHelper(lookupOp, fragmentType.getValueMembers(), fragmentType.hasLock(), typeConverter);
      auto lookupKey = mapping.resolve(lookupOp, lookupOp.getKeys());
      mlir::Value hash = hashKeys(lookupKey, rewriter, lookupOp->getLoc());
      auto equalFnBuilder = [&lookupOp](mlir::OpBuilder& rewriter, EntryStorageHelper::LazyValueMap left, std::vector<Value> right) -> Value {
         std::vector<mlir::Value> arguments;
         for (const auto& pair : left) { arguments.push_back(pair.second); }
         for (const auto& value : right) { arguments.push_back(value); }
         auto res = inlineBlock(&lookupOp.getEqFn().front(), rewriter, arguments);
         return res[0];
      };
      auto initValBuilder = [&lookupOp, this](SubOpRewriter& rewriter) -> std::vector<mlir::Value> {
         std::vector<mlir::Value> res;
         rewriter.inlineBlock<tuples::ReturnOpAdaptor>(&lookupOp.getInitFn().front(), {}, [&](tuples::ReturnOpAdaptor adaptor) {
            res = std::vector<mlir::Value>{adaptor.getResults().begin(), adaptor.getResults().end()};
         });
         for (size_t i = 0; i < res.size(); i++) {
            auto convertedType = typeConverter->convertType(res[i].getType());
            if (res[i].getType() != convertedType) {
               res[i] = rewriter.create<mlir::UnrealizedConversionCastOp>(lookupOp->getLoc(), convertedType, res[i]).getResult(0);
            }
         }
         return res;
      };
      auto loc = lookupOp->getLoc();
      Value hashed = hash;

      auto* context = rewriter.getContext();
      auto entryType = getHtEntryType(fragmentType, *typeConverter);

      auto idxType = rewriter.getIndexType();
      auto idxPtrType = util::RefType::get(rewriter.getContext(), idxType);
      auto kvType = getHtKVType(fragmentType, *typeConverter);
      auto kvPtrType = util::RefType::get(context, kvType);
      auto keyPtrType = keyStorageHelper.getRefType();
      auto valPtrType = valStorageHelper.getRefType();

      Value ht = adaptor.getState();
      Value htMask = rewriter.create<arith::ConstantIndexOp>(loc, 1024 - 1);
      Value falseValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));

      //position = hash & hashTableMask
      Value position = rewriter.create<arith::AndIOp>(loc, htMask, rewriter.create<arith::ShRUIOp>(loc, hashed, rewriter.create<mlir::arith::ConstantIndexOp>(loc, 6)));
      // ptr = &hashtable[position]
      Type bucketPtrType = util::RefType::get(context, entryType);
      Value currEntryPtr = rewriter.create<util::LoadOp>(loc, bucketPtrType, ht, position);
      ;
      //    if (*ptr != nullptr){
      Value cmp = rewriter.create<util::IsRefValidOp>(loc, rewriter.getI1Type(), currEntryPtr);
      auto ifOp = rewriter.create<scf::IfOp>(
         loc, cmp, [&](OpBuilder& b, Location loc) {
            Value hashAddress=rewriter.create<util::TupleElementPtrOp>(loc, idxPtrType, currEntryPtr, 1);
            Value entryHash = rewriter.create<util::LoadOp>(loc, idxType, hashAddress);
            Value hashMatches = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, entryHash,hashed);
            auto ifOpH = b.create<scf::IfOp>(
               loc, hashMatches, [&](OpBuilder& b, Location loc) {
                  Value kvAddress = rewriter.create<util::TupleElementPtrOp>(loc, kvPtrType, currEntryPtr, 2);
                  Value keyRef = rewriter.create<util::TupleElementPtrOp>(loc, keyPtrType, kvAddress, 0);
                  auto keyValues=keyStorageHelper.getValueMap(keyRef,rewriter,loc);
                  Value keyMatches = equalFnBuilder(b, keyValues, lookupKey);
                  b.create<scf::YieldOp>(loc, mlir::ValueRange{keyMatches});
               }, [&](OpBuilder& b, Location loc) {  b.create<scf::YieldOp>(loc, falseValue);});
            b.create<scf::YieldOp>(loc, ifOpH.getResults()); }, [&](OpBuilder& b, Location loc) { b.create<scf::YieldOp>(loc, ValueRange{falseValue}); });
      auto ifOp2 = rewriter.create<scf::IfOp>(
         loc, ifOp.getResults()[0], [&](OpBuilder& b, Location loc) { b.create<scf::YieldOp>(loc, currEntryPtr); },
         [&](OpBuilder& b, Location loc) {
            auto initialVals = initValBuilder(rewriter);
            Value entryRef = rt::PreAggregationHashtableFragment::insert(b, loc)({adaptor.getState(), hashed})[0];
            Value entryRefCasted = rewriter.create<util::GenericMemrefCastOp>(loc, bucketPtrType, entryRef);
            Value kvRef = rewriter.create<util::TupleElementPtrOp>(loc, kvPtrType, entryRefCasted, 2);
            Value keyRef = rewriter.create<util::TupleElementPtrOp>(loc, keyPtrType, kvRef, 0);
            Value valRef = rewriter.create<util::TupleElementPtrOp>(loc, valPtrType, kvRef, 1);
            keyStorageHelper.storeOrderedValues(keyRef, lookupKey, rewriter, loc);
            valStorageHelper.storeOrderedValues(valRef, initialVals, rewriter, loc);
            if (fragmentType.hasLock()) {
               rt::EntryLock::initialize(rewriter, loc)({valStorageHelper.getLockPointer(valRef, rewriter, loc)});
            }
            b.create<scf::YieldOp>(loc, ValueRange{entryRefCasted});
         });
      currEntryPtr = ifOp2.getResult(0);
      Value kvAddress = rewriter.create<util::TupleElementPtrOp>(loc, kvPtrType, currEntryPtr, 2);
      mapping.define(lookupOp.getRef(), rewriter.create<util::TupleElementPtrOp>(lookupOp->getLoc(), util::RefType::get(getContext(), kvType.getType(1)), kvAddress, 1));
      rewriter.replaceTupleStream(lookupOp, mapping);
      return mlir::success();
   }
};

//===----------------------------------------------------------------------===//
// Hash Map Lookup with Insert Operation
//===----------------------------------------------------------------------===//

class LookupHashMapLowering : public SubOpTupleStreamConsumerConversionPattern<subop::LookupOrInsertOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::LookupOrInsertOp>::SubOpTupleStreamConsumerConversionPattern;
   LogicalResult matchAndRewrite(subop::LookupOrInsertOp lookupOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      if (!mlir::isa<subop::HashMapType>(lookupOp.getState().getType())) return failure();
      subop::HashMapType htStateType = mlir::cast<subop::HashMapType>(lookupOp.getState().getType());
      EntryStorageHelper keyStorageHelper(lookupOp, htStateType.getKeyMembers(), false, typeConverter);
      EntryStorageHelper valStorageHelper(lookupOp, htStateType.getValueMembers(), htStateType.hasLock(), typeConverter);
      auto lookupKey = mapping.resolve(lookupOp, lookupOp.getKeys());

      mlir::Value hash = hashKeys(lookupKey, rewriter, lookupOp->getLoc());
      mlir::Value hashTable = adaptor.getState();
      auto equalFnBuilder = [&lookupOp](mlir::OpBuilder& rewriter, EntryStorageHelper::LazyValueMap left, std::vector<Value> right) -> Value {
         std::vector<mlir::Value> arguments;
         for (const auto& pair : left) { arguments.push_back(pair.second); }
         for (const auto& value : right) { arguments.push_back(value); }
         auto res = inlineBlock(&lookupOp.getEqFn().front(), rewriter, arguments);
         return res[0];
      };
      auto initValBuilder = [&lookupOp, this](SubOpRewriter& rewriter) -> std::vector<mlir::Value> {
         std::vector<mlir::Value> res;
         rewriter.inlineBlock<tuples::ReturnOpAdaptor>(&lookupOp.getInitFn().front(), {}, [&](tuples::ReturnOpAdaptor adaptor) {
            res = std::vector<mlir::Value>{adaptor.getResults().begin(), adaptor.getResults().end()};
         });
         for (size_t i = 0; i < res.size(); i++) {
            auto convertedType = typeConverter->convertType(res[i].getType());
            if (res[i].getType() != convertedType) {
               res[i] = rewriter.create<mlir::UnrealizedConversionCastOp>(lookupOp->getLoc(), convertedType, res[i]).getResult(0);
            }
         }
         return res;
      };
      auto loc = lookupOp->getLoc();
      Value hashed = hash;

      auto* context = rewriter.getContext();
      auto entryType = getHtEntryType(htStateType, *typeConverter);
      auto i8PtrType = util::RefType::get(context, IntegerType::get(context, 8));
      auto i8PtrPtrType = util::RefType::get(context, i8PtrType);

      auto idxType = rewriter.getIndexType();
      auto idxPtrType = util::RefType::get(rewriter.getContext(), idxType);
      auto kvType = getHtKVType(htStateType, *typeConverter);
      auto kvPtrType = util::RefType::get(context, kvType);
      auto keyPtrType = keyStorageHelper.getRefType();
      auto valPtrType = valStorageHelper.getRefType();
      auto entryPtrType = util::RefType::get(context, entryType);
      auto htType = util::RefType::get(context, entryPtrType);

      Value castedState = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(getContext(), mlir::TupleType::get(getContext(), {htType, rewriter.getIndexType()})), adaptor.getState());
      Value htAddress = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(rewriter.getContext(), htType), castedState, 0);
      Value htMaskAddress = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(rewriter.getContext(), rewriter.getIndexType()), castedState, 1);
      Value ht = rewriter.create<util::LoadOp>(loc, htType, htAddress);
      Value htMask = rewriter.create<util::LoadOp>(loc, rewriter.getIndexType(), htMaskAddress);

      Value trueValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
      Value falseValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));

      //position = hash & hashTableMask
      Value position = rewriter.create<arith::AndIOp>(loc, htMask, hashed);
      // ptr = &hashtable[position]
      Type bucketPtrType = util::RefType::get(context, entryType);
      Value firstPtr = rewriter.create<util::LoadOp>(loc, bucketPtrType, ht, position);
      Value tagMatches = rewriter.create<util::PtrTagMatches>(loc, rewriter.getI1Type(), firstPtr, hashed);
      firstPtr = rewriter.create<arith::SelectOp>(loc, tagMatches, rewriter.create<util::UnTagPtr>(loc, firstPtr.getType(), firstPtr), rewriter.create<util::InvalidRefOp>(loc, firstPtr.getType()));

      auto whileOp = rewriter.create<scf::WhileOp>(loc, bucketPtrType, ValueRange({firstPtr}));
      Block* before = new Block;
      before->addArgument(bucketPtrType, loc);
      whileOp.getBefore().push_back(before);
      Block* after = new Block;
      after->addArgument(bucketPtrType, loc);
      whileOp.getAfter().push_back(after);

      rewriter.atStartOf(before, [&](SubOpRewriter& rewriter) {
         Value ptr = before->getArgument(0);

         Value currEntryPtr = ptr;
         //    if (*ptr != nullptr){
         Value cmp = rewriter.create<util::IsRefValidOp>(loc, rewriter.getI1Type(), currEntryPtr);
         auto ifOp = rewriter.create<scf::IfOp>(
            loc, cmp, [&](OpBuilder& b, Location loc) {
               Value hashAddress=rewriter.create<util::TupleElementPtrOp>(loc, idxPtrType, currEntryPtr, 1);
               Value entryHash = rewriter.create<util::LoadOp>(loc, idxType, hashAddress);
               Value hashMatches = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, entryHash,hashed);
               auto ifOpH = b.create<scf::IfOp>(
                  loc, hashMatches, [&](OpBuilder& b, Location loc) {
                     Value kvAddress = rewriter.create<util::TupleElementPtrOp>(loc, kvPtrType, currEntryPtr, 2);
                     Value keyRef = rewriter.create<util::TupleElementPtrOp>(loc, keyPtrType, kvAddress, 0);
                     auto keyValues=keyStorageHelper.getValueMap(keyRef,rewriter,loc);
                     Value keyMatches = equalFnBuilder(b, keyValues, lookupKey);
                     auto ifOp2 = b.create<scf::IfOp>(
                        loc,  keyMatches, [&](OpBuilder& b, Location loc) {


                           b.create<scf::YieldOp>(loc, ValueRange{falseValue, ptr}); }, [&](OpBuilder& b, Location loc) {
                           Value nextPtrAddr=rewriter.create<util::TupleElementPtrOp>(loc, i8PtrPtrType, currEntryPtr, 0);
                           //          ptr = &entry.next
                           Value newEntryPtr=b.create<util::LoadOp>(loc,nextPtrAddr,mlir::Value());
                           Value newPtr=b.create<util::GenericMemrefCastOp>(loc, bucketPtrType,newEntryPtr);
                           //          yield ptr,done=false
                           b.create<scf::YieldOp>(loc, ValueRange{trueValue, newPtr }); });
                     b.create<scf::YieldOp>(loc, ifOp2.getResults());
                  }, [&](OpBuilder& b, Location loc) {
                     Value nextPtrAddr=rewriter.create<util::TupleElementPtrOp>(loc, i8PtrPtrType, currEntryPtr, 0);
                     //          ptr = &entry.next
                     Value newEntryPtr=b.create<util::LoadOp>(loc,nextPtrAddr,mlir::Value());
                     Value newPtr=b.create<util::GenericMemrefCastOp>(loc, bucketPtrType,newEntryPtr);
                     //          yield ptr,done=false
                     b.create<scf::YieldOp>(loc, ValueRange{trueValue, newPtr });});
               b.create<scf::YieldOp>(loc, ifOpH.getResults()); }, [&](OpBuilder& b, Location loc) {
               auto initialVals = initValBuilder(rewriter);
               //       %newEntry = ...
               Value entryRef=rt::Hashtable::insert(b,loc)({hashTable,hashed})[0];
               Value entryRefCasted= rewriter.create<util::GenericMemrefCastOp>(loc, bucketPtrType, entryRef);
               Value kvRef = rewriter.create<util::TupleElementPtrOp>(loc, kvPtrType, entryRefCasted, 2);
               Value keyRef = rewriter.create<util::TupleElementPtrOp>(loc, keyPtrType, kvRef, 0);
               Value valRef = rewriter.create<util::TupleElementPtrOp>(loc, valPtrType, kvRef, 1);
               keyStorageHelper.storeOrderedValues(keyRef,lookupKey,rewriter,loc);
               valStorageHelper.storeOrderedValues(valRef,initialVals,rewriter,loc);

               b.create<scf::YieldOp>(loc, ValueRange{falseValue, entryRefCasted}); });
         //       if(compare(entry.key,key)){

         Value done = ifOp.getResult(0);
         Value newPtr = ifOp.getResult(1);
         rewriter.create<scf::ConditionOp>(loc, done, ValueRange({newPtr}));
      });
      rewriter.atStartOf(after, [&](SubOpRewriter& rewriter) {
         rewriter.create<scf::YieldOp>(loc, after->getArguments());
      });
      Value currEntryPtr = whileOp.getResult(0);
      Value kvAddress = rewriter.create<util::TupleElementPtrOp>(loc, kvPtrType, currEntryPtr, 2);
      mapping.define(lookupOp.getRef(), rewriter.create<util::TupleElementPtrOp>(lookupOp->getLoc(), util::RefType::get(getContext(), kvType.getType(1)), kvAddress, 1));
      rewriter.replaceTupleStream(lookupOp, mapping);
      return mlir::success();
   }
};

//===----------------------------------------------------------------------===//
// External Hash Index Lookup Operation
//===----------------------------------------------------------------------===//

class LookupExternalHashIndexLowering : public SubOpTupleStreamConsumerConversionPattern<subop::LookupOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::LookupOp>::SubOpTupleStreamConsumerConversionPattern;
   LogicalResult matchAndRewrite(subop::LookupOp lookupOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      if (!mlir::isa<subop::ExternalHashIndexType>(lookupOp.getState().getType())) return failure();

      auto loc = lookupOp->getLoc();

      // Calculate hash value and perform lookup in external index hashmap
      auto hashValue = rewriter.create<db::Hash>(loc, rewriter.create<util::PackOp>(loc, mapping.resolve(lookupOp, lookupOp.getKeys())));
      mlir::Value list = rt::HashIndexAccess::lookup(rewriter, loc)({adaptor.getState(), hashValue})[0];

      mapping.define(lookupOp.getRef(), list);
      rewriter.replaceTupleStream(lookupOp, mapping);
      return mlir::success();
   }
};

} // namespace subop_to_cf
} // namespace dialect
} // namespace compiler
} // namespace pgx_lower