#include "../Headers/SubOpToControlFlowCommon.h"
#include "../Headers/SubOpToControlFlowPatterns.h"
#include "../Headers/SubOpToControlFlowRewriter.h"
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
// Table Reference Operations
//===----------------------------------------------------------------------===//

class TableRefGatherOpLowering : public SubOpTupleStreamConsumerConversionPattern<subop::GatherOp, 2> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::GatherOp, 2>::SubOpTupleStreamConsumerConversionPattern;

   LogicalResult matchAndRewrite(subop::GatherOp gatherOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      auto refType = gatherOp.getRef().getColumn().type;
      if (!mlir::isa<subop::TableEntryRefType>(refType)) { return failure(); }
      auto columns = mlir::cast<subop::TableEntryRefType>(refType).getMembers();
      auto tableRefVal = mapping.resolve(gatherOp, gatherOp.getRef());
      llvm::SmallVector<mlir::Value> unpacked;
      rewriter.createOrFold<util::UnPackOp>(unpacked, gatherOp->getLoc(), tableRefVal);
      auto currRow = unpacked[0];
      llvm::SmallVector<mlir::Value> unPackedColumns;
      rewriter.createOrFold<util::UnPackOp>(unPackedColumns, gatherOp->getLoc(), unpacked[1]);
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
// Table Materialization Operations
//===----------------------------------------------------------------------===//

class MaterializeTableLowering : public SubOpTupleStreamConsumerConversionPattern<subop::MaterializeOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::MaterializeOp>::SubOpTupleStreamConsumerConversionPattern;

   LogicalResult matchAndRewrite(subop::MaterializeOp materializeOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      if (!mlir::isa<subop::ResultTableType>(materializeOp.getState().getType())) return failure();
      auto stateType = mlir::cast<subop::ResultTableType>(materializeOp.getState().getType());
      mlir::Value loaded = rewriter.create<util::LoadOp>(materializeOp->getLoc(), adaptor.getState());
      auto columnBuilders = rewriter.create<util::UnPackOp>(materializeOp->getLoc(), loaded);
      for (size_t i = 0; i < stateType.getMembers().getTypes().size(); i++) {
         auto memberName = mlir::cast<mlir::StringAttr>(stateType.getMembers().getNames()[i]).str();
         auto attribute = mlir::cast<tuples::ColumnRefAttr>(materializeOp.getMapping().get(memberName));
         auto val = mapping.resolve(materializeOp, attribute);
         // TODO Phase 5: Replace Arrow operations with PostgreSQL equivalents
         // auto asArrayBuilder = rewriter.create<arrow::BuilderFromPtr>(materializeOp->getLoc(), columnBuilders.getResult(i));
         // rewriter.create<db::AppendArrowOp>(materializeOp->getLoc(), asArrayBuilder, val);
      }
      rewriter.eraseOp(materializeOp);
      return mlir::success();
   }
};

class MaterializeHeapLowering : public SubOpTupleStreamConsumerConversionPattern<subop::MaterializeOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::MaterializeOp>::SubOpTupleStreamConsumerConversionPattern;

   LogicalResult matchAndRewrite(subop::MaterializeOp materializeOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      auto heapType = mlir::dyn_cast_or_null<subop::HeapType>(materializeOp.getState().getType());
      if (!heapType) return failure();
      EntryStorageHelper storageHelper(materializeOp, heapType.getMembers(), heapType.hasLock(), typeConverter);
      mlir::Value ref;
      rewriter.atStartOf(&rewriter.getCurrentStreamLoc()->getParentOfType<mlir::func::FuncOp>().getFunctionBody().front(), [&](SubOpRewriter& rewriter) {
         ref = rewriter.create<util::AllocaOp>(materializeOp->getLoc(), util::RefType::get(storageHelper.getStorageType()), mlir::Value());
      });
      storageHelper.storeFromColumns(materializeOp.getMapping(), mapping, ref, rewriter, materializeOp->getLoc());
      rt::Heap::insert(rewriter, materializeOp->getLoc())({adaptor.getState(), ref});
      rewriter.eraseOp(materializeOp);
      return mlir::success();
   }
};

class MaterializeVectorLowering : public SubOpTupleStreamConsumerConversionPattern<subop::MaterializeOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::MaterializeOp>::SubOpTupleStreamConsumerConversionPattern;

   LogicalResult matchAndRewrite(subop::MaterializeOp materializeOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      auto bufferType = mlir::dyn_cast_or_null<subop::BufferType>(materializeOp.getState().getType());
      if (!bufferType) return failure();
      EntryStorageHelper storageHelper(materializeOp, bufferType.getMembers(), bufferType.hasLock(), typeConverter);
      mlir::Value ref = rt::GrowingBuffer::insert(rewriter, materializeOp->getLoc())({adaptor.getState()})[0];
      storageHelper.storeFromColumns(materializeOp.getMapping(), mapping, ref, rewriter, materializeOp->getLoc());
      rewriter.eraseOp(materializeOp);
      return mlir::success();
   }
};

//===----------------------------------------------------------------------===//
// Table Scanning Operations
//===----------------------------------------------------------------------===//

class ScanRefsTableLowering : public SubOpConversionPattern<subop::ScanRefsOp> {
   public:
   using SubOpConversionPattern<subop::ScanRefsOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::ScanRefsOp scanOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      if (!mlir::isa<subop::TableType>(scanOp.getState().getType())) return failure();
      auto loc = scanOp->getLoc();
      auto refType = mlir::cast<subop::TableEntryRefType>(scanOp.getRef().getColumn().type);
      std::string memberMapping = "[";
      std::vector<mlir::Type> accessedColumnTypes;
      auto members = refType.getMembers();
      for (auto i = 0ul; i < members.getTypes().size(); i++) {
         auto type = mlir::cast<mlir::TypeAttr>(members.getTypes()[i]).getValue();
         auto name = mlir::cast<mlir::StringAttr>(members.getNames()[i]).str();
         accessedColumnTypes.push_back(type);
         if (memberMapping.length() > 1) {
            memberMapping += ",";
         }
         memberMapping += "\"" + name + "\"";
      }
      memberMapping += "]";
      mlir::Value memberMappingValue = rewriter.create<util::CreateConstVarLen>(scanOp->getLoc(), util::VarLen32Type::get(rewriter.getContext()), memberMapping);
      mlir::Value iterator = rt::DataSourceIteration::init(rewriter, scanOp->getLoc())({adaptor.getState(), memberMappingValue})[0];
      ColumnMapping mapping;

      auto* ctxt = rewriter.getContext();
      auto i16T = mlir::IntegerType::get(rewriter.getContext(), 16);
      std::vector<mlir::Type> recordBatchTypes{rewriter.getIndexType(), rewriter.getIndexType(), util::RefType::get(i16T), util::RefType::get(rewriter.getI16Type())};
      auto recordBatchInfoRepr = mlir::TupleType::get(ctxt, recordBatchTypes);
      ModuleOp parentModule = scanOp->getParentOfType<ModuleOp>();
      mlir::func::FuncOp funcOp;
      static size_t funcIds;
      auto ptrType = util::RefType::get(getContext(), IntegerType::get(getContext(), 8));
      rewriter.atStartOf(parentModule.getBody(), [&](SubOpRewriter& rewriter) {
         funcOp = rewriter.create<mlir::func::FuncOp>(parentModule.getLoc(), "scan_func" + std::to_string(funcIds++), mlir::FunctionType::get(getContext(), TypeRange{ptrType, ptrType}, TypeRange()));
      });
      auto* funcBody = new Block;
      mlir::Value recordBatchPointer = funcBody->addArgument(ptrType, loc);
      mlir::Value contextPtr = funcBody->addArgument(ptrType, loc);
      funcOp.getBody().push_back(funcBody);
      auto ptr = rewriter.storeStepRequirements();
      rewriter.atStartOf(funcBody, [&](SubOpRewriter& rewriter) {
         rewriter.loadStepRequirements(contextPtr, *typeConverter);
         recordBatchPointer = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(getContext(), recordBatchInfoRepr), recordBatchPointer);
         mlir::Value lenRef = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(rewriter.getIndexType()), recordBatchPointer, 0);
         mlir::Value offsetRef = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(rewriter.getIndexType()), recordBatchPointer, 1);
         //mlir::Value selVecRef = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(util::RefType::get(i16T)), recordBatchPointer, 2);
         mlir::Value ptrRef = rewriter.create<util::TupleElementPtrOp>(loc, util::RefType::get(util::RefType::get(rewriter.getI16Type())), recordBatchPointer, 3);
         mlir::Value ptrToColumns = rewriter.create<util::LoadOp>(loc, ptrRef);
         std::vector<mlir::Value> arrays;
         for (size_t i = 0; i < accessedColumnTypes.size(); i++) {
            auto ci = rewriter.create<mlir::arith::ConstantIndexOp>(loc, i);
            auto array = rewriter.create<util::LoadOp>(loc, ptrToColumns, ci);
            arrays.push_back(array);
         }
         auto arraysVal = rewriter.create<util::PackOp>(loc, arrays);
         auto start = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
         auto end = rewriter.create<util::LoadOp>(loc, lenRef);
         auto globalOffset = rewriter.create<util::LoadOp>(loc, offsetRef);
         auto c1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
         auto forOp2 = rewriter.create<mlir::scf::ForOp>(loc, start, end, c1, mlir::ValueRange{});
         rewriter.atStartOf(forOp2.getBody(), [&](SubOpRewriter& rewriter) {
            auto withOffset = rewriter.create<mlir::arith::AddIOp>(loc, forOp2.getInductionVar(), globalOffset);
            auto currentRecord = rewriter.create<util::PackOp>(loc, mlir::ValueRange{withOffset, arraysVal});
            mapping.define(scanOp.getRef(), currentRecord);
            rewriter.replaceTupleStream(scanOp, mapping);
         });
         rewriter.create<mlir::func::ReturnOp>(loc);
      });
      Value functionPointer = rewriter.create<mlir::func::ConstantOp>(loc, funcOp.getFunctionType(), SymbolRefAttr::get(rewriter.getStringAttr(funcOp.getSymName())));
      Value parallelConst = rewriter.create<mlir::arith::ConstantIntOp>(loc, scanOp->hasAttr("parallel"), rewriter.getI1Type());
      rt::DataSourceIteration::iterate(rewriter, scanOp->getLoc())({iterator, parallelConst, functionPointer, ptr});
      return success();
   }
};

//===----------------------------------------------------------------------===//
// Table Creation Operations
//===----------------------------------------------------------------------===//

class CreateFromResultTableLowering : public SubOpConversionPattern<subop::CreateFrom> {
   using SubOpConversionPattern<subop::CreateFrom>::SubOpConversionPattern;
   LogicalResult matchAndRewrite(subop::CreateFrom createOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      auto resultTableType = mlir::dyn_cast<subop::ResultTableType>(createOp.getState().getType());
      if (!resultTableType) return failure();
      mlir::Value loaded = rewriter.create<util::LoadOp>(createOp->getLoc(), adaptor.getState());
      auto columnBuilders = rewriter.create<util::UnPackOp>(createOp->getLoc(), loaded);
      auto loc = createOp->getLoc();
      mlir::Value table = rt::ArrowTable::createEmpty(rewriter, loc)({})[0];
      for (auto i = 0ul; i < columnBuilders.getNumResults(); i++) {
         auto columnBuilder = columnBuilders.getResult(i);
         auto column = rt::ArrowColumnBuilder::finish(rewriter, loc)({columnBuilder})[0];
         mlir::Value columnName = rewriter.create<util::CreateConstVarLen>(loc, util::VarLen32Type::get(getContext()), mlir::cast<mlir::StringAttr>(createOp.getColumns()[i]));
         table = rt::ArrowTable::addColumn(rewriter, loc)({table, columnName, column})[0];
      }
      rewriter.replaceOp(createOp, table);
      return success();
   }
};

class CreateTableLowering : public SubOpConversionPattern<subop::GenericCreateOp> {
   std::string arrowDescrFromType(mlir::Type type) const {
      if (type.isIndex()) {
         return "int[64]";
      } else if (isIntegerType(type, 1)) {
         return "bool";
      } else if (auto intWidth = getIntegerWidth(type, false)) {
         return "int[" + std::to_string(intWidth) + "]";
      } else if (auto uIntWidth = getIntegerWidth(type, true)) {
         return "uint[" + std::to_string(uIntWidth) + "]";
      } else if (auto floatType = mlir::dyn_cast_or_null<mlir::FloatType>(type)) {
         return "float[" + std::to_string(floatType.getWidth()) + "]";
      } else if (auto decimalType = mlir::dyn_cast_or_null<db::DecimalType>(type)) {
         auto prec = std::min(decimalType.getP(), 38);
         return "decimal[" + std::to_string(prec) + "," + std::to_string(decimalType.getS()) + "]";
      } else if (auto dateType = mlir::dyn_cast_or_null<db::DateType>(type)) {
         return dateType.getUnit() == db::DateUnitAttr::day ? "date[32]" : "date[64]";
      } else if (auto timestampType = mlir::dyn_cast_or_null<db::TimestampType>(type)) {
         return "timestamp[" + std::to_string(static_cast<uint32_t>(timestampType.getUnit())) + "]";
      } else if (mlir::isa<db::StringType>(type)) {
         return "string";
      } else if (auto charType = mlir::dyn_cast_or_null<db::CharType>(type)) {
         if (charType.getLen() <= 1) {
            return "fixed_sized[4]";
         } else {
            return "string";
         }
      }
      assert(false);
      return "";
   }

   public:
   using SubOpConversionPattern<subop::GenericCreateOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::GenericCreateOp createOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      if (!mlir::isa<subop::ResultTableType>(createOp.getType())) return failure();
      auto tableType = mlir::cast<subop::ResultTableType>(createOp.getType());
      std::string descr;
      std::vector<mlir::Value> columnBuilders;
      auto loc = createOp->getLoc();
      for (size_t i = 0; i < tableType.getMembers().getTypes().size(); i++) {
         auto type = mlir::cast<mlir::TypeAttr>(tableType.getMembers().getTypes()[i]).getValue();
         auto baseType = getBaseType(type);
         mlir::Value typeDescr = rewriter.create<util::CreateConstVarLen>(loc, util::VarLen32Type::get(getContext()), arrowDescrFromType(baseType));
         Value columnBuilder = rt::ArrowColumnBuilder::create(rewriter, loc)({typeDescr})[0];
         columnBuilders.push_back(columnBuilder);
      }
      mlir::Value tpl = rewriter.create<util::PackOp>(createOp->getLoc(), columnBuilders);
      auto tplSize = rewriter.create<util::SizeOfOp>(createOp->getLoc(), rewriter.getIndexType(), tpl.getType());
      mlir::Value ref = rt::ExecutionContext::allocStateRaw(rewriter, loc)({tplSize})[0];
      ref = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(tpl.getType()), ref);
      rewriter.create<util::StoreOp>(createOp->getLoc(), tpl, ref, mlir::Value());
      rewriter.replaceOp(createOp, ref);
      return mlir::success();
   }
};

//===----------------------------------------------------------------------===//
// External Table Operations
//===----------------------------------------------------------------------===//

class GetExternalTableLowering : public SubOpConversionPattern<subop::GetExternalOp> {
   public:
   using SubOpConversionPattern<subop::GetExternalOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::GetExternalOp op, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      if (!mlir::isa<subop::TableType>(op.getType())) return failure();
      mlir::Value description = rewriter.create<util::CreateConstVarLen>(op->getLoc(), util::VarLen32Type::get(rewriter.getContext()), op.getDescrAttr());
      rewriter.replaceOp(op, rt::DataSource::get(rewriter, op->getLoc())({description})[0]);
      return mlir::success();
   }
};

//===----------------------------------------------------------------------===//
// Generate Operations
//===----------------------------------------------------------------------===//

class GenerateLowering : public SubOpConversionPattern<subop::GenerateOp> {
   public:
   using SubOpConversionPattern<subop::GenerateOp>::SubOpConversionPattern;

   LogicalResult matchAndRewrite(subop::GenerateOp generateOp, OpAdaptor adaptor, SubOpRewriter& rewriter) const override {
      ColumnMapping mapping;
      std::vector<subop::GenerateEmitOp> emitOps;
      generateOp.getRegion().walk([&](subop::GenerateEmitOp emitOp) {
         emitOps.push_back(emitOp);
      });
      std::vector<mlir::Value> streams;
      for (auto emitOp : emitOps) {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPointAfter(emitOp);
         ColumnMapping mapping;
         mapping.define(generateOp.getGeneratedColumns(), emitOp.getValues());
         mlir::Value newInFlight = rewriter.createInFlight(mapping);
         streams.push_back(newInFlight);
         rewriter.eraseOp(emitOp);
      }

      rewriter.inlineBlock<tuples::ReturnOpAdaptor>(&generateOp.getRegion().front(), {}, [](auto x) {});
      for (auto [inflight, stream] : llvm::zip(streams, generateOp.getStreams())) {
         stream.replaceAllUsesWith(inflight);
      }
      rewriter.eraseOp(generateOp);

      return success();
   }
};

} // namespace subop_to_cf
} // namespace dialect
} // namespace compiler
} // namespace pgx_lower