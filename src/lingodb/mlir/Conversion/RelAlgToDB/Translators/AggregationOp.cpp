#include "lingodb/mlir/Conversion/RelAlgToDB/OrderedAttributes.h"
#include "lingodb/mlir/Conversion/RelAlgToDB/Translator.h"
#include "lingodb/mlir/Dialect/DB/IR/DBOps.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "lingodb/mlir/Dialect/util/UtilOps.h"
#include <llvm/ADT/TypeSwitch.h>
#include "lingodb/mlir/Dialect/DSA/IR/DSAOps.h"
#include "pgx-lower/utility/logging.h"

class AggregationTranslator : public mlir::relalg::Translator {
   mlir::relalg::AggregationOp aggregationOp;
   mlir::Value aggrHt;

   mlir::TupleType keyTupleType;
   mlir::TupleType valTupleType;

   mlir::relalg::OrderedAttributes key;
   mlir::relalg::OrderedAttributes val;

   std::vector<std::function<std::pair<const mlir::relalg::Column*, mlir::Value>(mlir::ValueRange, mlir::OpBuilder& builder)>> finalizeFunctions;
   std::vector<std::function<std::vector<mlir::Value>(mlir::ValueRange, mlir::ValueRange, mlir::OpBuilder& builder)>> aggregationFunctions;
   std::vector<mlir::Value> defaultValues;
   std::vector<mlir::Type> aggrTypes;

   public:
   AggregationTranslator(mlir::relalg::AggregationOp aggregationOp) : mlir::relalg::Translator(aggregationOp), aggregationOp(aggregationOp) {
   }

    // PGX-LOWER: Postgres handles type casts inside of aggregations differently to lingodb. Noteably, when you do aggregation
    // on an in32 the output result is expected to be an int64. This is the proper way too apprach this - we add type casts
    // inside of the aggregation operation.
   static mlir::Value castToAggregationType(mlir::OpBuilder& builder, const mlir::Location loc,
                                            mlir::Value sourceVal,
                                            const mlir::Value targetVal, mlir::Type resultingType) {
       if (targetVal.getType() == sourceVal.getType()) {
           return sourceVal;
       }

       auto getBaseType = [](mlir::Type type) -> mlir::Type {
           if (const auto nullableType = type.dyn_cast<mlir::db::NullableType>()) {
               return nullableType.getType();
           }
           return type;
       };

       auto baseResultType = getBaseType(resultingType);
       const auto baseSourceType = getBaseType(sourceVal.getType());
       const auto sourceIsNullable = sourceVal.getType().isa<mlir::db::NullableType>();
       const auto resultIsNullable = resultingType.isa<mlir::db::NullableType>();
       auto castVal = sourceVal;

       if (baseResultType != baseSourceType) {
           if (sourceIsNullable) {
               auto extracted = builder.create<mlir::db::NullableGetVal>(loc, sourceVal);
               auto casted = builder.create<mlir::db::CastOp>(loc, baseResultType, extracted);
               auto isNull = builder.create<mlir::db::IsNullOp>(loc, builder.getI1Type(), sourceVal);
               castVal = builder.create<mlir::db::AsNullableOp>(loc, resultingType, casted, isNull);
           } else {
               castVal = builder.create<mlir::db::CastOp>(loc, baseResultType, sourceVal);
               if (resultIsNullable) {
                   castVal = builder.create<mlir::db::AsNullableOp>(loc, resultingType, castVal);
               }
           }
       } else if (resultIsNullable && !sourceIsNullable) {
           castVal = builder.create<mlir::db::AsNullableOp>(loc, resultingType, sourceVal);
       }

       return castVal;
   }

   mlir::Value compareKeys(mlir::OpBuilder& rewriter, mlir::Value left, mlir::Value right,mlir::Location loc) {
      mlir::Value equal = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getI1Type(), rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
      auto leftUnpacked = rewriter.create<mlir::util::UnPackOp>(loc, left);
      auto rightUnpacked = rewriter.create<mlir::util::UnPackOp>(loc, right);
      for (size_t i = 0; i < leftUnpacked.getNumResults(); i++) {
         mlir::Value compared;
         auto currLeftType = leftUnpacked->getResult(i).getType();
         auto currRightType = rightUnpacked.getResult(i).getType();
         auto currLeftNullableType = currLeftType.dyn_cast_or_null<mlir::db::NullableType>();
         auto currRightNullableType = currRightType.dyn_cast_or_null<mlir::db::NullableType>();
         if (currLeftNullableType || currRightNullableType) {
            ::mlir::Value isNull1 = rewriter.create<mlir::db::IsNullOp>(loc, rewriter.getI1Type(), leftUnpacked->getResult(i));
            ::mlir::Value isNull2 = rewriter.create<mlir::db::IsNullOp>(loc, rewriter.getI1Type(), rightUnpacked->getResult(i));

            // Print: null flags
            auto extIsNull1 = rewriter.create<mlir::arith::ExtUIOp>(loc, rewriter.getI32Type(), isNull1);
            auto extIsNull2 = rewriter.create<mlir::arith::ExtUIOp>(loc, rewriter.getI32Type(), isNull2);
            auto ptrType = mlir::util::RefType::get(rewriter.getContext(), rewriter.getI8Type());
            auto nullPtr = rewriter.create<mlir::util::UndefOp>(loc, ptrType);
            rewriter.create<mlir::db::RuntimeCall>(loc, mlir::TypeRange{}, "PrintI32", mlir::ValueRange{nullPtr, extIsNull1});
            rewriter.create<mlir::db::RuntimeCall>(loc, mlir::TypeRange{}, "PrintI32", mlir::ValueRange{nullPtr, extIsNull2});

            ::mlir::Value anyNull = rewriter.create<mlir::arith::OrIOp>(loc, isNull1, isNull2);
            ::mlir::Value bothNull = rewriter.create<mlir::arith::AndIOp>(loc, isNull1, isNull2);

            // Print: anyNull and bothNull
            auto extAnyNull = rewriter.create<mlir::arith::ExtUIOp>(loc, rewriter.getI32Type(), anyNull);
            auto extBothNull = rewriter.create<mlir::arith::ExtUIOp>(loc, rewriter.getI32Type(), bothNull);
            rewriter.create<mlir::db::RuntimeCall>(loc, mlir::TypeRange{}, "PrintI32", mlir::ValueRange{nullPtr, extAnyNull});
            rewriter.create<mlir::db::RuntimeCall>(loc, mlir::TypeRange{}, "PrintI32", mlir::ValueRange{nullPtr, extBothNull});

            auto ifOp = rewriter.create<mlir::scf::IfOp>(loc, mlir::TypeRange{rewriter.getI1Type()}, anyNull);
            ifOp.getThenRegion().emplaceBlock();
            ifOp.getElseRegion().emplaceBlock();
            {
               mlir::OpBuilder::InsertionGuard guard(rewriter);
               rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
               // Then branch
               rewriter.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{bothNull});
            }
            {
               mlir::OpBuilder::InsertionGuard guard(rewriter);
               rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
               // Else branch
               ::mlir::Value left = rewriter.create<mlir::db::NullableGetVal>(loc, leftUnpacked->getResult(i));
               ::mlir::Value right = rewriter.create<mlir::db::NullableGetVal>(loc, rightUnpacked->getResult(i));

               // Print: extracted values (only for i32)
               auto ptrType2 = mlir::util::RefType::get(rewriter.getContext(), rewriter.getI8Type());
               auto nullPtr2 = rewriter.create<mlir::util::UndefOp>(loc, ptrType2);
               if (left.getType().isInteger(32) && right.getType().isInteger(32)) {
                  rewriter.create<mlir::db::RuntimeCall>(loc, mlir::TypeRange{}, "PrintI32", mlir::ValueRange{nullPtr2, left});
                  rewriter.create<mlir::db::RuntimeCall>(loc, mlir::TypeRange{}, "PrintI32", mlir::ValueRange{nullPtr2, right});
               }

               ::mlir::Value cmpRes = rewriter.create<mlir::db::CmpOp>(loc, mlir::db::DBCmpPredicate::eq, left, right);

               // Print: comparison result
               auto extCmpRes = rewriter.create<mlir::arith::ExtUIOp>(loc, rewriter.getI32Type(), cmpRes);
               rewriter.create<mlir::db::RuntimeCall>(loc, mlir::TypeRange{}, "PrintI32", mlir::ValueRange{nullPtr2, extCmpRes});

               rewriter.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{cmpRes});
            }
            compared = ifOp.getResult(0);
         } else {
            compared = rewriter.create<mlir::db::CmpOp>(loc, mlir::db::DBCmpPredicate::eq, leftUnpacked->getResult(i), rightUnpacked.getResult(i));
         }

         // Print: compared result
         auto extCompared = rewriter.create<mlir::arith::ExtUIOp>(loc, rewriter.getI32Type(), compared);
         auto ptrType3 = mlir::util::RefType::get(rewriter.getContext(), rewriter.getI8Type());
         auto nullPtr3 = rewriter.create<mlir::util::UndefOp>(loc, ptrType3);
         rewriter.create<mlir::db::RuntimeCall>(loc, mlir::TypeRange{}, "PrintI32", mlir::ValueRange{nullPtr3, extCompared});

         mlir::Value localEqual = rewriter.create<mlir::arith::AndIOp>(loc, rewriter.getI1Type(), mlir::ValueRange({equal, compared}));

         // Print: localEqual (accumulated equality)
         auto extLocalEqual = rewriter.create<mlir::arith::ExtUIOp>(loc, rewriter.getI32Type(), localEqual);
         rewriter.create<mlir::db::RuntimeCall>(loc, mlir::TypeRange{}, "PrintI32", mlir::ValueRange{nullPtr3, extLocalEqual});

         equal = localEqual;
      }
      return equal;
   }
   virtual void consume(mlir::relalg::Translator* child, mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context) override {
      mlir::Value packedKey = key.pack(context, builder, aggregationOp->getLoc());
      mlir::Value packedVal = val.pack(context, builder, aggregationOp->getLoc());

      auto reduceOp = builder.create<mlir::dsa::HashtableInsert>(aggregationOp->getLoc(), aggrHt, packedKey, packedVal);

      auto scope = context.createScope();

      auto aggrTupleType = mlir::TupleType::get(builder.getContext(), aggrTypes);
      keyTupleType = key.getTupleType(builder.getContext());
      valTupleType = val.getTupleType(builder.getContext());
      aggrTupleType = mlir::TupleType::get(builder.getContext(), aggrTypes);
      if (!keyTupleType.getTypes().empty()) {
         {
            mlir::Block* aggrBuilderBlock = new mlir::Block;
            reduceOp.getEqual().push_back(aggrBuilderBlock);
            aggrBuilderBlock->addArguments({packedKey.getType(), packedKey.getType()}, {aggregationOp->getLoc(), aggregationOp->getLoc()});
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(aggrBuilderBlock);
            auto yieldOp = builder.create<mlir::dsa::YieldOp>(aggregationOp->getLoc());
            builder.setInsertionPointToStart(aggrBuilderBlock);
            mlir::Value matches = compareKeys(builder, aggrBuilderBlock->getArgument(0), aggrBuilderBlock->getArgument(1),aggregationOp->getLoc());
            builder.create<mlir::dsa::YieldOp>(aggregationOp->getLoc(), matches);
            yieldOp.erase();
         }
         {
            mlir::Block* aggrBuilderBlock = new mlir::Block;
            reduceOp.getHash().push_back(aggrBuilderBlock);
            aggrBuilderBlock->addArguments({packedKey.getType()}, {aggregationOp->getLoc()});
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(aggrBuilderBlock);
            mlir::Value hashed = builder.create<mlir::db::Hash>(aggregationOp->getLoc(), builder.getIndexType(), aggrBuilderBlock->getArgument(0));
            builder.create<mlir::dsa::YieldOp>(aggregationOp->getLoc(), hashed);
         }
      }
      mlir::Block* aggrBuilderBlock = new mlir::Block;
      reduceOp.getReduce().push_back(aggrBuilderBlock);
      aggrBuilderBlock->addArguments({aggrTupleType, valTupleType}, {aggregationOp->getLoc(), aggregationOp->getLoc()});
      mlir::OpBuilder builder2(builder.getContext());
      builder2.setInsertionPointToStart(aggrBuilderBlock);
      auto unpackedCurr = builder2.create<mlir::util::UnPackOp>(aggregationOp->getLoc(), aggrBuilderBlock->getArgument(0))->getResults();
      mlir::ValueRange unpackedNew;
      if (valTupleType.getTypes().size() > 0) {
         unpackedNew = builder2.create<mlir::util::UnPackOp>(aggregationOp->getLoc(), aggrBuilderBlock->getArgument(1)).getResults();
      }
      std::vector<mlir::Value> valuesx;
      for (auto aggrFn : aggregationFunctions) {
         auto vec = aggrFn(unpackedCurr, unpackedNew, builder2);
         valuesx.insert(valuesx.end(), vec.begin(), vec.end());
      }

      mlir::Value packedx = builder2.create<mlir::util::PackOp>(aggregationOp->getLoc(), valuesx);

      builder2.create<mlir::dsa::YieldOp>(aggregationOp->getLoc(), packedx);
   }

   mlir::Attribute getMaxValueAttr(mlir::Type type) {
      auto* context = aggregationOp->getContext();
      mlir::OpBuilder builder(context);
      mlir::Attribute maxValAttr = ::llvm::TypeSwitch<::mlir::Type, mlir::Attribute>(type)

                                      .Case<::mlir::db::DecimalType>([&](::mlir::db::DecimalType t) {
                                         if (t.getP() < 19) {
                                            return (mlir::Attribute) builder.getI64IntegerAttr(std::numeric_limits<int64_t>::max());
                                         }
                                         std::vector<uint64_t> parts = {0xFFFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF};
                                         return (mlir::Attribute) builder.getIntegerAttr(mlir::IntegerType::get(context, 128), mlir::APInt(128, parts));
                                      })
                                      .Case<::mlir::IntegerType>([&](::mlir::IntegerType) {
                                         return builder.getI64IntegerAttr(std::numeric_limits<int64_t>::max());
                                      })
                                      .Case<::mlir::FloatType>([&](::mlir::FloatType t) {
                                         if (t.getWidth() == 32) {
                                            return (mlir::Attribute) builder.getF32FloatAttr(std::numeric_limits<float>::max());
                                         } else if (t.getWidth() == 64) {
                                            return (mlir::Attribute) builder.getF64FloatAttr(std::numeric_limits<double>::max());
                                         } else {
                                            assert(false && "should not happen");
                                            return mlir::Attribute();
                                         }
                                      })
                                      .Default([&](::mlir::Type) { return builder.getI64IntegerAttr(std::numeric_limits<int64_t>::max()); });
      return maxValAttr;
   }
   void analyze(::mlir::OpBuilder& builder) {
      key = mlir::relalg::OrderedAttributes::fromRefArr(aggregationOp.getGroupByCols());

      auto counterType = builder.getI64Type();
      mlir::relalg::ReturnOp terminator = mlir::cast<mlir::relalg::ReturnOp>(aggregationOp.getAggrFunc().front().getTerminator());

      for (size_t i = 0; i < aggregationOp.getComputedCols().size(); i++) {
         auto* destAttr = &aggregationOp.getComputedCols()[i].cast<mlir::relalg::ColumnDefAttr>().getColumn();
         ::mlir::Value computedVal = terminator.getResults()[i];
         if (auto aggrFn = mlir::dyn_cast_or_null<mlir::relalg::AggrFuncOp>(computedVal.getDefiningOp())) {
            auto loc = aggrFn->getLoc();
            auto* attr = &aggrFn.getAttr().getColumn();
            auto attrIsNullable = attr->type.isa<mlir::db::NullableType>();
            size_t currValIdx = val.insert(attr);
            mlir::Type resultingType = destAttr->type;
            size_t currDestIdx = aggrTypes.size();

            if (aggrFn.getFn() == mlir::relalg::AggrFunc::sum) {
               finalizeFunctions.push_back([currDestIdx = currDestIdx, destAttr = destAttr](mlir::ValueRange range, mlir::OpBuilder& builder) { return std::make_pair(destAttr, range[currDestIdx]); });
               aggrTypes.push_back(resultingType);
               mlir::Value initVal;
               if (resultingType.isa<mlir::db::NullableType>()) {
                  initVal = builder.create<mlir::db::NullOp>(loc, resultingType);
               } else {
                  initVal = builder.create<mlir::db::ConstantOp>(loc, getBaseType(resultingType), builder.getI64IntegerAttr(0));
               }
               defaultValues.push_back(initVal);
               aggregationFunctions.push_back([loc, currDestIdx = currDestIdx, currValIdx = currValIdx, attrIsNullable, resultingType = resultingType](mlir::ValueRange aggr, mlir::ValueRange val, mlir::OpBuilder& builder) {
                  std::vector<mlir::Value> res;
                  mlir::Value currVal = aggr[currDestIdx];
                  mlir::Value newVal = val[currValIdx];

                  const auto castNewVal = castToAggregationType(builder, loc, newVal, currVal, resultingType);
                  mlir::Value added = builder.create<mlir::db::AddOp>(loc, resultingType, currVal, castNewVal);
                  mlir::Value updatedVal = added;
                  if (attrIsNullable) {
                     mlir::Value isNull1 = builder.create<mlir::db::IsNullOp>(loc, builder.getI1Type(), castNewVal);
                     updatedVal = builder.create<mlir::arith::SelectOp>(loc, isNull1, currVal, added);
                  }
                  if (resultingType.isa<mlir::db::NullableType>()) {
                     mlir::Value isNull = builder.create<mlir::db::IsNullOp>(loc, builder.getI1Type(), currVal);
                     res.push_back(builder.create<mlir::arith::SelectOp>(loc, isNull, castNewVal, updatedVal));
                  } else {
                     res.push_back(updatedVal);
                  }
                  return res;
               });
            } else if (aggrFn.getFn() == mlir::relalg::AggrFunc::min) {
               finalizeFunctions.push_back([currDestIdx = currDestIdx, destAttr = destAttr](mlir::ValueRange range, mlir::OpBuilder& builder) { return std::make_pair(destAttr, range[currDestIdx]); });
               aggrTypes.push_back(resultingType);
               mlir::Value initVal;
               if (resultingType.isa<mlir::db::NullableType>()) {
                  initVal = builder.create<mlir::db::NullOp>(aggregationOp.getLoc(), resultingType);
               } else {
                  initVal = builder.create<mlir::db::ConstantOp>(aggregationOp.getLoc(), getBaseType(resultingType), getMaxValueAttr(resultingType));
               }
               defaultValues.push_back(initVal);
               aggregationFunctions.push_back([loc, currDestIdx = currDestIdx, resultingType = resultingType, currValIdx = currValIdx](mlir::ValueRange aggr, mlir::ValueRange val, mlir::OpBuilder& builder) {
                  std::vector<mlir::Value> res;
                  mlir::Value currVal = aggr[currDestIdx];
                  mlir::Value newVal = val[currValIdx];

                  auto castNewVal = castToAggregationType(builder, loc, newVal, currVal, resultingType);
                  mlir::Value newLtCurr = builder.create<mlir::db::CmpOp>(loc, mlir::db::DBCmpPredicate::lt, castNewVal, currVal);
                  mlir::Value newLtCurrT = builder.create<mlir::db::DeriveTruth>(loc, newLtCurr);
                  mlir::Value selected = builder.create<mlir::arith::SelectOp>(loc, newLtCurrT, castNewVal, currVal);

                  if (resultingType.isa<mlir::db::NullableType>()) {
                     mlir::Value isNull = builder.create<mlir::db::IsNullOp>(loc, builder.getI1Type(), currVal);
                     res.push_back(builder.create<mlir::arith::SelectOp>(loc, isNull, castNewVal, selected));
                  } else {
                     res.push_back(selected);
                  }
                  return res;
               });
            } else if (aggrFn.getFn() == mlir::relalg::AggrFunc::max) {
               finalizeFunctions.push_back([currDestIdx = currDestIdx, destAttr = destAttr](mlir::ValueRange range, mlir::OpBuilder& builder) { return std::make_pair(destAttr, range[currDestIdx]); });
               aggrTypes.push_back(resultingType);
               mlir::Value initVal;
               if (resultingType.isa<mlir::db::NullableType>()) {
                  initVal = builder.create<mlir::db::NullOp>(aggregationOp.getLoc(), resultingType);
               } else {
                  initVal = builder.create<mlir::db::ConstantOp>(aggregationOp.getLoc(), getBaseType(resultingType), builder.getI64IntegerAttr(0));
               }
               defaultValues.push_back(initVal);
               aggregationFunctions.push_back([loc, currDestIdx = currDestIdx, attrIsNullable, resultingType = resultingType, currValIdx = currValIdx](mlir::ValueRange aggr, mlir::ValueRange val, mlir::OpBuilder& builder) {
                  std::vector<mlir::Value> res;
                  mlir::Value currVal = aggr[currDestIdx];
                  mlir::Value newVal = val[currValIdx];

                  auto castNewVal = castToAggregationType(builder, loc, newVal, currVal, resultingType);
                  mlir::Value currGtNew = builder.create<mlir::db::CmpOp>(loc, mlir::db::DBCmpPredicate::gt, currVal, castNewVal);
                  mlir::Value currGTNewT = builder.create<mlir::db::DeriveTruth>(loc, currGtNew);
                  mlir::Value selected = builder.create<mlir::arith::SelectOp>(loc, currGTNewT, currVal, castNewVal);
                  mlir::Value updatedVal = selected;

                  if (attrIsNullable) {
                     mlir::Value isNull1 = builder.create<mlir::db::IsNullOp>(loc, builder.getI1Type(), castNewVal);
                     updatedVal = builder.create<mlir::arith::SelectOp>(loc, isNull1, currVal, selected);
                  }
                  if (resultingType.isa<mlir::db::NullableType>()) {
                     mlir::Value isNull = builder.create<mlir::db::IsNullOp>(loc, builder.getI1Type(), currVal);
                     res.push_back(builder.create<mlir::arith::SelectOp>(loc, isNull, castNewVal, updatedVal));
                  } else {
                     res.push_back(updatedVal);
                  }
                  return res;
               });
            } else if (aggrFn.getFn() == mlir::relalg::AggrFunc::avg) {
               aggrTypes.push_back(resultingType);
               aggrTypes.push_back(counterType);
                // PGX-LOWER: Using a proper base constant here instead of only an int64
               auto zeroAttr = mlir::Attribute();
               auto baseType = getBaseType(resultingType);
               if (baseType.isa<mlir::db::DecimalType>()) {
                  zeroAttr = builder.getIntegerAttr(mlir::IntegerType::get(builder.getContext(), 128), mlir::APInt(128, 0));
               } else if (baseType.isa<mlir::FloatType>()) {
                  auto floatType = baseType.cast<mlir::FloatType>();
                  if (floatType.getWidth() == 32) {
                     zeroAttr = builder.getF32FloatAttr(0.0f);
                  } else {
                     zeroAttr = builder.getF64FloatAttr(0.0);
                  }
               } else if (baseType.isa<mlir::IntegerType>()) {
                  zeroAttr = builder.getI64IntegerAttr(0);
               } else {
                   PGX_WARNING("Unsupported base type in averaging operation");
               }
               // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
               mlir::Value initVal = builder.create<mlir::db::ConstantOp>(aggregationOp.getLoc(), baseType, zeroAttr);
               mlir::Value initCounterVal = builder.create<mlir::db::ConstantOp>(aggregationOp.getLoc(), counterType, builder.getI64IntegerAttr(0));
               mlir::Value defaultVal = resultingType.isa<mlir::db::NullableType>() ? builder.create<mlir::db::AsNullableOp>(aggregationOp.getLoc(), resultingType, initVal) : initVal;
               defaultValues.push_back(defaultVal);
               defaultValues.push_back(initCounterVal);
               finalizeFunctions.push_back([loc, currDestIdx = currDestIdx, destAttr = destAttr, resultingType = resultingType](mlir::ValueRange range, mlir::OpBuilder builder) {
                  mlir::Value casted=builder.create<mlir::db::CastOp>(loc, getBaseType(resultingType), range[currDestIdx+1]);
                  if(resultingType.isa<mlir::db::NullableType>()&&casted.getType()!=resultingType){
                     casted=builder.create<mlir::db::AsNullableOp>(loc, resultingType, casted);
                  }
                  mlir::Value average=builder.create<mlir::db::DivOp>(loc, resultingType, range[currDestIdx], casted);
                  return std::make_pair(destAttr, average); });
               aggregationFunctions.push_back([loc, currDestIdx = currDestIdx, currValIdx = currValIdx, attrIsNullable, resultingType = resultingType, counterType = counterType](mlir::ValueRange aggr, mlir::ValueRange val, mlir::OpBuilder& builder) {
                  std::vector<mlir::Value> res;
                  auto one = builder.create<mlir::db::ConstantOp>(loc, counterType, builder.getI64IntegerAttr(1));
                  mlir::Value currVal = aggr[currDestIdx];
                  mlir::Value newVal = val[currValIdx];

                  mlir::Value castNewVal = castToAggregationType(builder, loc, newVal, currVal, resultingType);
                  mlir::Value added1 = builder.create<mlir::db::AddOp>(loc, resultingType, currVal, castNewVal);
                  mlir::Value added2 = builder.create<mlir::db::AddOp>(loc, counterType, aggr[currDestIdx + 1], one);
                  if (attrIsNullable) {
                     mlir::Value isNull1 = builder.create<mlir::db::IsNullOp>(loc, builder.getI1Type(), castNewVal);
                     res.push_back(builder.create<mlir::arith::SelectOp>(loc, isNull1, currVal, added1));
                     res.push_back(builder.create<mlir::arith::SelectOp>(loc, isNull1, aggr[currDestIdx + 1], added2));
                  } else {
                     res.push_back(added1);
                     res.push_back(added2);
                  }

                  return res;
               });
            } else if (aggrFn.getFn() == mlir::relalg::AggrFunc::count) {
               size_t currDestIdx = aggrTypes.size();
               auto initCounterVal = builder.create<mlir::db::ConstantOp>(aggregationOp.getLoc(), counterType, builder.getI64IntegerAttr(0));
               defaultValues.push_back(initCounterVal);
               aggrTypes.push_back(resultingType);
               finalizeFunctions.push_back([currDestIdx = currDestIdx, destAttr = destAttr](mlir::ValueRange range, mlir::OpBuilder& builder) { return std::make_pair(destAttr, range[currDestIdx]); });

               aggregationFunctions.push_back([loc, currDestIdx = currDestIdx, attrIsNullable, currValIdx = currValIdx, counterType = counterType, resultingType = resultingType](mlir::ValueRange aggr, mlir::ValueRange val, mlir::OpBuilder& builder) {
                  std::vector<mlir::Value> res;
                  auto one = builder.create<mlir::db::ConstantOp>(loc, counterType, builder.getI64IntegerAttr(1));
                  mlir::Value value = builder.create<mlir::db::AddOp>(loc, resultingType, aggr[currDestIdx], one);
                  if (attrIsNullable) {
                     mlir::Value isNull2 = builder.create<mlir::db::IsNullOp>(loc, builder.getI1Type(), val[currValIdx]);
                     mlir::Value tmp = builder.create<mlir::arith::SelectOp>(loc, isNull2, aggr[currDestIdx], value);
                     value = tmp;
                  }

                  res.push_back(value);
                  return res;
               });
            } else if (aggrFn.getFn() == mlir::relalg::AggrFunc::any) {
               size_t currDestIdx = aggrTypes.size();
               auto initVal = builder.create<mlir::util::UndefOp>(aggregationOp.getLoc(), resultingType);
               defaultValues.push_back(initVal);
               aggrTypes.push_back(resultingType);
               finalizeFunctions.push_back([currDestIdx = currDestIdx, destAttr = destAttr](mlir::ValueRange range, mlir::OpBuilder& builder) { return std::make_pair(destAttr, range[currDestIdx]); });

               aggregationFunctions.push_back([currValIdx = currValIdx](mlir::ValueRange aggr, mlir::ValueRange val, mlir::OpBuilder& builder) {
                  std::vector<mlir::Value> res;
                  res.push_back(val[currValIdx]);
                  return res;
               });
            }
         }
         if (auto countOp = mlir::dyn_cast_or_null<mlir::relalg::CountRowsOp>(computedVal.getDefiningOp())) {
            auto loc = countOp->getLoc();

            size_t currDestIdx = aggrTypes.size();
            aggrTypes.push_back(counterType);
            auto initCounterVal = builder.create<mlir::db::ConstantOp>(aggregationOp.getLoc(), counterType, builder.getI64IntegerAttr(0));
            defaultValues.push_back(initCounterVal);
            finalizeFunctions.push_back([currDestIdx = currDestIdx, destAttr = destAttr](mlir::ValueRange range, mlir::OpBuilder& builder) { return std::make_pair(destAttr, range[currDestIdx]); });

            aggregationFunctions.push_back([loc, currDestIdx = currDestIdx, counterType = counterType](mlir::ValueRange aggr, mlir::ValueRange val, mlir::OpBuilder& builder) {
               std::vector<mlir::Value> res;
               auto one = builder.create<mlir::db::ConstantOp>(loc, counterType, builder.getI64IntegerAttr(1));
               mlir::Value added2 = builder.create<mlir::db::AddOp>(loc, counterType, aggr[currDestIdx], one);
               res.push_back(added2);
               return res;
            });
         }
      };
   }
   virtual void produce(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      auto scope = context.createScope();
         analyze(builder);
         auto aggrTupleType = mlir::TupleType::get(builder.getContext(), aggrTypes);
         auto initTuple = builder.create<mlir::util::PackOp>(aggregationOp->getLoc(), aggrTupleType, defaultValues);
         keyTupleType = key.getTupleType(builder.getContext());
         valTupleType = val.getTupleType(builder.getContext());

         // Extract HashtableSpecification pointer from aggregationOp attributes
         uint64_t specPtrValue = 0;
         if (auto specAttr = aggregationOp->getAttrOfType<mlir::IntegerAttr>("hashtable_spec")) {
            specPtrValue = specAttr.getValue().getZExtValue();
            PGX_LOG(RELALG_LOWER, DEBUG, "Extracted HashtableSpecification pointer 0x%lx from aggregationOp",
                    specPtrValue);
         }

         auto createDSOp = builder.create<mlir::dsa::CreateDS>(
             aggregationOp.getLoc(),
             mlir::dsa::AggregationHashtableType::get(builder.getContext(), keyTupleType, aggrTupleType),
             initTuple);

         if (specPtrValue != 0) {
            auto specAttr = builder.getIntegerAttr(
                builder.getIntegerType(64),  // signless 64-bit
                specPtrValue
            );
            createDSOp->setAttr("spec_ptr", specAttr);
            PGX_LOG(RELALG_LOWER, DEBUG, "Stored spec pointer 0x%lx as attribute on CreateDS",
                    specPtrValue);
         }

         aggrHt = createDSOp.getResult();

      auto iterEntryType = mlir::TupleType::get(builder.getContext(), {keyTupleType, aggrTupleType});
      children[0]->produce(context, builder);

      {
         auto forOp2 = builder.create<mlir::dsa::ForOp>(aggregationOp->getLoc(), mlir::TypeRange{}, aggrHt, mlir::Value(), mlir::ValueRange{});
         mlir::Block* block2 = new mlir::Block;
         block2->addArgument(iterEntryType, aggregationOp->getLoc());
         forOp2.getBodyRegion().push_back(block2);
         mlir::OpBuilder builder2(forOp2.getBodyRegion());
         auto unpacked = builder2.create<mlir::util::UnPackOp>(aggregationOp->getLoc(), forOp2.getInductionVar()).getResults();
         mlir::ValueRange unpackedKey;
         if (!keyTupleType.getTypes().empty()) {
            unpackedKey = builder2.create<mlir::util::UnPackOp>(aggregationOp->getLoc(), unpacked[0]).getResults();
         }
         auto unpackedAggr = builder2.create<mlir::util::UnPackOp>(aggregationOp->getLoc(), unpacked[1]).getResults();

         for (auto fn : finalizeFunctions) {
            auto [attr, val] = fn(unpackedAggr, builder2);
            context.setValueForAttribute(scope, attr, val);
         }
         key.setValuesForColumns(context, scope, unpackedKey);
         consumer->consume(this, builder2, context);
         builder2.create<mlir::dsa::YieldOp>(aggregationOp->getLoc(), mlir::ValueRange{});
      }
      builder.create<mlir::dsa::FreeOp>(aggregationOp->getLoc(), aggrHt);
   }
   virtual void done() override {
   }
   virtual ~AggregationTranslator() {}
};

std::unique_ptr<mlir::relalg::Translator> mlir::relalg::Translator::createAggregationTranslator(mlir::relalg::AggregationOp sortOp) {
   return std::make_unique<AggregationTranslator>(sortOp);
}