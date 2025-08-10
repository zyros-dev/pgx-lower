#include "mlir/Conversion/RelAlgToDB/OrderedAttributes.h"
#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/util/UtilOps.h"
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/DSA/IR/DSAOps.h>

class AggregationTranslator : public pgx::mlir::relalg::Translator {
   pgx::mlir::relalg::AggregationOp aggregationOp;
   ::mlir::Value aggrHt;

   mlir::TupleType keyTupleType;
   mlir::TupleType valTupleType;

   pgx::mlir::relalg::OrderedAttributes key;
   pgx::mlir::relalg::OrderedAttributes val;

   std::vector<std::function<std::pair<const pgx::mlir::relalg::Column*, ::mlir::Value>(::mlir::ValueRange, ::mlir::OpBuilder& builder)>> finalizeFunctions;
   std::vector<std::function<std::vector<::mlir::Value>(::mlir::ValueRange, ::mlir::ValueRange, ::mlir::OpBuilder& builder)>> aggregationFunctions;
   std::vector<::mlir::Value> defaultValues;
   std::vector<::mlir::Type> aggrTypes;

   public:
   AggregationTranslator(pgx::mlir::relalg::AggregationOp aggregationOp) : pgx::mlir::relalg::Translator(aggregationOp), aggregationOp(aggregationOp) {
   }

   ::mlir::Value compareKeys(::mlir::OpBuilder& rewriter, ::mlir::Value left, ::mlir::Value right,::mlir::Location loc) {
      ::mlir::Value equal = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getI1Type(), rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
      auto leftUnpacked = rewriter.create<pgx::mlir::util::UnPackOp>(loc, left);
      auto rightUnpacked = rewriter.create<pgx::mlir::util::UnPackOp>(loc, right);
      for (size_t i = 0; i < leftUnpacked.getNumResults(); i++) {
         ::mlir::Value compared;
         auto currLeftType = leftUnpacked->getResult(i).getType();
         auto currRightType = rightUnpacked.getResult(i).getType();
         auto currLeftNullableType = currLeftType.dyn_cast_or_null<pgx::mlir::db::NullableType>();
         auto currRightNullableType = currRightType.dyn_cast_or_null<pgx::mlir::db::NullableType>();
         if (currLeftNullableType || currRightNullableType) {
            ::mlir::Value isNull1 = rewriter.create<pgx::mlir::db::IsNullOp>(loc, rewriter.getI1Type(), leftUnpacked->getResult(i));
            ::mlir::Value isNull2 = rewriter.create<pgx::mlir::db::IsNullOp>(loc, rewriter.getI1Type(), rightUnpacked->getResult(i));
            ::mlir::Value anyNull = rewriter.create<mlir::arith::OrIOp>(loc, isNull1, isNull2);
            ::mlir::Value bothNull = rewriter.create<mlir::arith::AndIOp>(loc, isNull1, isNull2);
            compared = rewriter.create<mlir::scf::IfOp>(
                                  loc, rewriter.getI1Type(), anyNull, [&](::mlir::OpBuilder& b, ::mlir::Location loc) { b.create<mlir::scf::YieldOp>(loc, bothNull); },
                                  [&](::mlir::OpBuilder& b, ::mlir::Location loc) {
                                     ::mlir::Value left = rewriter.create<pgx::mlir::db::NullableGetVal>(loc, leftUnpacked->getResult(i));
                                     ::mlir::Value right = rewriter.create<pgx::mlir::db::NullableGetVal>(loc, rightUnpacked->getResult(i));
                                     ::mlir::Value cmpRes = rewriter.create<pgx::mlir::db::CmpOp>(loc, pgx::mlir::db::DBCmpPredicate::eq, left, right);
                                     b.create<mlir::scf::YieldOp>(loc, cmpRes);
                                  })
                          .getResult(0);
         } else {
            compared = rewriter.create<pgx::mlir::db::CmpOp>(loc, pgx::mlir::db::DBCmpPredicate::eq, leftUnpacked->getResult(i), rightUnpacked.getResult(i));
         }
         ::mlir::Value localEqual = rewriter.create<mlir::arith::AndIOp>(loc, rewriter.getI1Type(), ::mlir::ValueRange({equal, compared}));
         equal = localEqual;
      }
      return equal;
   }
   virtual void consume(pgx::mlir::relalg::Translator* child, ::mlir::OpBuilder& builder, pgx::mlir::relalg::TranslatorContext& context) override {
      ::mlir::Value packedKey = key.pack(context, builder, aggregationOp->getLoc());
      ::mlir::Value packedVal = val.pack(context, builder, aggregationOp->getLoc());

      auto reduceOp = builder.create<pgx::mlir::dsa::HashtableInsert>(aggregationOp->getLoc(), aggrHt, packedKey, packedVal);

      auto scope = context.createScope();

      auto aggrTupleType = mlir::TupleType::get(builder.getContext(), aggrTypes);
      keyTupleType = key.getTupleType(builder.getContext());
      valTupleType = val.getTupleType(builder.getContext());
      aggrTupleType = mlir::TupleType::get(builder.getContext(), aggrTypes);
      if (!keyTupleType.getTypes().empty()) {
         {
            ::mlir::Block* aggrBuilderBlock = new ::mlir::Block;
            reduceOp.equal().push_back(aggrBuilderBlock);
            aggrBuilderBlock->addArguments({packedKey.getType(), packedKey.getType()}, {aggregationOp->getLoc(), aggregationOp->getLoc()});
            ::mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(aggrBuilderBlock);
            auto yieldOp = builder.create<pgx::mlir::dsa::YieldOp>(aggregationOp->getLoc());
            builder.setInsertionPointToStart(aggrBuilderBlock);
            ::mlir::Value matches = compareKeys(builder, aggrBuilderBlock->getArgument(0), aggrBuilderBlock->getArgument(1),aggregationOp->getLoc());
            builder.create<pgx::mlir::dsa::YieldOp>(aggregationOp->getLoc(), matches);
            yieldOp.erase();
         }
         {
            ::mlir::Block* aggrBuilderBlock = new ::mlir::Block;
            reduceOp.hash().push_back(aggrBuilderBlock);
            aggrBuilderBlock->addArguments({packedKey.getType()}, {aggregationOp->getLoc()});
            ::mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(aggrBuilderBlock);
            ::mlir::Value hashed = builder.create<pgx::mlir::db::Hash>(aggregationOp->getLoc(), builder.getIndexType(), aggrBuilderBlock->getArgument(0));
            builder.create<pgx::mlir::dsa::YieldOp>(aggregationOp->getLoc(), hashed);
         }
      }
      ::mlir::Block* aggrBuilderBlock = new ::mlir::Block;
      reduceOp.reduce().push_back(aggrBuilderBlock);
      aggrBuilderBlock->addArguments({aggrTupleType, valTupleType}, {aggregationOp->getLoc(), aggregationOp->getLoc()});
      ::mlir::OpBuilder builder2(builder.getContext());
      builder2.setInsertionPointToStart(aggrBuilderBlock);
      auto unpackedCurr = builder2.create<pgx::mlir::util::UnPackOp>(aggregationOp->getLoc(), aggrBuilderBlock->getArgument(0))->getResults();
      ::mlir::ValueRange unpackedNew;
      if (valTupleType.getTypes().size() > 0) {
         unpackedNew = builder2.create<pgx::mlir::util::UnPackOp>(aggregationOp->getLoc(), aggrBuilderBlock->getArgument(1)).getResults();
      }
      std::vector<::mlir::Value> valuesx;
      for (auto aggrFn : aggregationFunctions) {
         auto vec = aggrFn(unpackedCurr, unpackedNew, builder2);
         valuesx.insert(valuesx.end(), vec.begin(), vec.end());
      }

      ::mlir::Value packedx = builder2.create<pgx::mlir::util::PackOp>(aggregationOp->getLoc(), valuesx);

      builder2.create<pgx::mlir::dsa::YieldOp>(aggregationOp->getLoc(), packedx);
   }

   ::mlir::Attribute getMaxValueAttr(::mlir::Type type) {
      auto* context = aggregationOp->getContext();
      ::mlir::OpBuilder builder(context);
      ::mlir::Attribute maxValAttr = ::llvm::TypeSwitch<::mlir::Type, ::mlir::Attribute>(type)

                                      .Case<::pgx::mlir::db::DecimalType>([&](::pgx::mlir::db::DecimalType t) {
                                         if (t.getP() < 19) {
                                            return (::mlir::Attribute) builder.getI64IntegerAttr(std::numeric_limits<int64_t>::max());
                                         }
                                         std::vector<uint64_t> parts = {0xFFFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF};
                                         return (::mlir::Attribute) builder.getIntegerAttr(mlir::IntegerType::get(context, 128), mlir::APInt(128, parts));
                                      })
                                      .Case<::mlir::IntegerType>([&](::mlir::IntegerType) {
                                         return builder.getI64IntegerAttr(std::numeric_limits<int64_t>::max());
                                      })
                                      .Case<::mlir::FloatType>([&](::mlir::FloatType t) {
                                         if (t.getWidth() == 32) {
                                            return (::mlir::Attribute) builder.getF32FloatAttr(std::numeric_limits<float>::max());
                                         } else if (t.getWidth() == 64) {
                                            return (::mlir::Attribute) builder.getF64FloatAttr(std::numeric_limits<double>::max());
                                         } else {
                                            assert(false && "should not happen");
                                            return ::mlir::Attribute();
                                         }
                                      })
                                      .Default([&](::mlir::Type) { return builder.getI64IntegerAttr(std::numeric_limits<int64_t>::max()); });
      return maxValAttr;
   }
   void analyze(::mlir::OpBuilder& builder) {
      key = pgx::mlir::relalg::OrderedAttributes::fromRefArr(aggregationOp.group_by_colsAttr());

      auto counterType = builder.getI64Type();
      pgx::mlir::relalg::ReturnOp terminator = mlir::cast<pgx::mlir::relalg::ReturnOp>(aggregationOp.aggr_func().front().getTerminator());

      for (size_t i = 0; i < aggregationOp.computed_cols().size(); i++) {
         auto* destAttr = &aggregationOp.computed_cols()[i].cast<pgx::mlir::relalg::ColumnDefAttr>().getColumn();
         ::mlir::Value computedVal = terminator.results()[i];
         if (auto aggrFn = mlir::dyn_cast_or_null<pgx::mlir::relalg::AggrFuncOp>(computedVal.getDefiningOp())) {
            auto loc = aggrFn->getLoc();
            auto* attr = &aggrFn.getAttr().getColumn();
            auto attrIsNullable = attr->type.isa<pgx::mlir::db::NullableType>();
            size_t currValIdx = val.insert(attr);
            ::mlir::Type resultingType = destAttr->type;
            size_t currDestIdx = aggrTypes.size();

            if (aggrFn.fn() == pgx::mlir::relalg::AggrFunc::sum) {
               finalizeFunctions.push_back([currDestIdx = currDestIdx, destAttr = destAttr](::mlir::ValueRange range, ::mlir::OpBuilder& builder) { return std::make_pair(destAttr, range[currDestIdx]); });
               aggrTypes.push_back(resultingType);
               ::mlir::Value initVal;
               if (resultingType.isa<pgx::mlir::db::NullableType>()) {
                  initVal = builder.create<pgx::mlir::db::NullOp>(loc, resultingType);
               } else {
                  initVal = builder.create<pgx::mlir::db::ConstantOp>(loc, getBaseType(resultingType), builder.getI64IntegerAttr(0));
               }
               defaultValues.push_back(initVal);
               aggregationFunctions.push_back([loc, currDestIdx = currDestIdx, currValIdx = currValIdx, attrIsNullable, resultingType = resultingType](::mlir::ValueRange aggr, ::mlir::ValueRange val, ::mlir::OpBuilder& builder) {
                  std::vector<::mlir::Value> res;
                  ::mlir::Value currVal = aggr[currDestIdx];
                  ::mlir::Value newVal = val[currValIdx];
                  ::mlir::Value added = builder.create<pgx::mlir::db::AddOp>(loc, resultingType, currVal, newVal);
                  ::mlir::Value updatedVal = added;
                  if (attrIsNullable) {
                     ::mlir::Value isNull1 = builder.create<pgx::mlir::db::IsNullOp>(loc, builder.getI1Type(), newVal);
                     updatedVal = builder.create<mlir::arith::SelectOp>(loc, isNull1, currVal, added);
                  }
                  if (resultingType.isa<pgx::mlir::db::NullableType>()) {
                     ::mlir::Value casted = newVal;
                     if (currVal.getType() != newVal.getType()) {
                        casted = builder.create<pgx::mlir::db::AsNullableOp>(loc, currVal.getType(), newVal);
                     }
                     ::mlir::Value isNull = builder.create<pgx::mlir::db::IsNullOp>(loc, builder.getI1Type(), currVal);
                     res.push_back(builder.create<mlir::arith::SelectOp>(loc, isNull, casted, updatedVal));
                  } else {
                     res.push_back(updatedVal);
                  }
                  return res;
               });
            } else if (aggrFn.fn() == pgx::mlir::relalg::AggrFunc::min) {
               finalizeFunctions.push_back([currDestIdx = currDestIdx, destAttr = destAttr](::mlir::ValueRange range, ::mlir::OpBuilder& builder) { return std::make_pair(destAttr, range[currDestIdx]); });
               aggrTypes.push_back(resultingType);
               ::mlir::Value initVal;
               if (resultingType.isa<pgx::mlir::db::NullableType>()) {
                  initVal = builder.create<pgx::mlir::db::NullOp>(aggregationOp.getLoc(), resultingType);
               } else {
                  initVal = builder.create<pgx::mlir::db::ConstantOp>(aggregationOp.getLoc(), getBaseType(resultingType), getMaxValueAttr(resultingType));
               }
               defaultValues.push_back(initVal);
               aggregationFunctions.push_back([loc, currDestIdx = currDestIdx, resultingType = resultingType, currValIdx = currValIdx](::mlir::ValueRange aggr, ::mlir::ValueRange val, ::mlir::OpBuilder& builder) {
                  std::vector<::mlir::Value> res;
                  ::mlir::Value currVal = aggr[currDestIdx];
                  ::mlir::Value newVal = val[currValIdx];
                  ::mlir::Value newLtCurr = builder.create<pgx::mlir::db::CmpOp>(loc, pgx::mlir::db::DBCmpPredicate::lt, newVal,currVal);
                  ::mlir::Value casted = newVal;
                  if (newVal.getType() != currVal.getType()) {
                     casted = builder.create<pgx::mlir::db::AsNullableOp>(loc, currVal.getType(), newVal);
                  }
                  ::mlir::Value newLtCurrT = builder.create<pgx::mlir::db::DeriveTruth>(loc, newLtCurr);

                  ::mlir::Value added = builder.create<mlir::arith::SelectOp>(loc, newLtCurrT, casted,currVal);
                  if (resultingType.isa<pgx::mlir::db::NullableType>()) {
                     ::mlir::Value isNull = builder.create<pgx::mlir::db::IsNullOp>(loc, builder.getI1Type(), currVal);
                     res.push_back(builder.create<mlir::arith::SelectOp>(loc, isNull, casted, added));
                  } else {
                     res.push_back(added);
                  }
                  return res;
               });
            } else if (aggrFn.fn() == pgx::mlir::relalg::AggrFunc::max) {
               finalizeFunctions.push_back([currDestIdx = currDestIdx, destAttr = destAttr](::mlir::ValueRange range, ::mlir::OpBuilder& builder) { return std::make_pair(destAttr, range[currDestIdx]); });
               aggrTypes.push_back(resultingType);
               ::mlir::Value initVal;
               if (resultingType.isa<pgx::mlir::db::NullableType>()) {
                  initVal = builder.create<pgx::mlir::db::NullOp>(aggregationOp.getLoc(), resultingType);
               } else {
                  initVal = builder.create<pgx::mlir::db::ConstantOp>(aggregationOp.getLoc(), getBaseType(resultingType), builder.getI64IntegerAttr(0));
               }
               defaultValues.push_back(initVal);
               aggregationFunctions.push_back([loc, currDestIdx = currDestIdx, attrIsNullable, resultingType = resultingType, currValIdx = currValIdx](::mlir::ValueRange aggr, ::mlir::ValueRange val, ::mlir::OpBuilder& builder) {
                  std::vector<::mlir::Value> res;
                  ::mlir::Value currVal = aggr[currDestIdx];
                  ::mlir::Value newVal = val[currValIdx];
                  ::mlir::Value currGtNew = builder.create<pgx::mlir::db::CmpOp>(loc, pgx::mlir::db::DBCmpPredicate::gt, currVal, newVal);
                  ::mlir::Value casted = newVal;
                  if (newVal.getType() != currVal.getType()) {
                     casted = builder.create<pgx::mlir::db::AsNullableOp>(loc, currVal.getType(), newVal);
                  }
                  ::mlir::Value currGTNewT = builder.create<pgx::mlir::db::DeriveTruth>(loc, currGtNew);
                  ::mlir::Value added = builder.create<mlir::arith::SelectOp>(loc, currGTNewT, currVal, casted);
                  ::mlir::Value updatedVal = added;
                  if (attrIsNullable) {
                     ::mlir::Value isNull1 = builder.create<pgx::mlir::db::IsNullOp>(loc, builder.getI1Type(), newVal);
                     updatedVal = builder.create<mlir::arith::SelectOp>(loc, isNull1, aggr[currDestIdx], added);
                  }
                  if (resultingType.isa<pgx::mlir::db::NullableType>()) {
                     ::mlir::Value isNull = builder.create<pgx::mlir::db::IsNullOp>(loc, builder.getI1Type(), aggr[currDestIdx]);
                     res.push_back(builder.create<mlir::arith::SelectOp>(loc, isNull, casted, updatedVal));
                  } else {
                     res.push_back(updatedVal);
                  }
                  return res;
               });
            } else if (aggrFn.fn() == pgx::mlir::relalg::AggrFunc::avg) {
               aggrTypes.push_back(resultingType);
               aggrTypes.push_back(counterType);
               ::mlir::Value initVal = builder.create<pgx::mlir::db::ConstantOp>(aggregationOp.getLoc(), getBaseType(resultingType), builder.getI64IntegerAttr(0));
               ::mlir::Value initCounterVal = builder.create<pgx::mlir::db::ConstantOp>(aggregationOp.getLoc(), counterType, builder.getI64IntegerAttr(0));
               ::mlir::Value defaultVal = resultingType.isa<pgx::mlir::db::NullableType>() ? builder.create<pgx::mlir::db::AsNullableOp>(aggregationOp.getLoc(), resultingType, initVal) : initVal;
               defaultValues.push_back(defaultVal);
               defaultValues.push_back(initCounterVal);
               finalizeFunctions.push_back([loc, currDestIdx = currDestIdx, destAttr = destAttr, resultingType = resultingType](::mlir::ValueRange range, ::mlir::OpBuilder builder) {
                  ::mlir::Value casted=builder.create<pgx::mlir::db::CastOp>(loc, getBaseType(resultingType), range[currDestIdx+1]);
                  if(resultingType.isa<pgx::mlir::db::NullableType>()&&casted.getType()!=resultingType){
                     casted=builder.create<pgx::mlir::db::AsNullableOp>(loc, resultingType, casted);
                  }
                  ::mlir::Value average=builder.create<pgx::mlir::db::DivOp>(loc, resultingType, range[currDestIdx], casted);
                  return std::make_pair(destAttr, average); });
               aggregationFunctions.push_back([loc, currDestIdx = currDestIdx, currValIdx = currValIdx, attrIsNullable, resultingType = resultingType, counterType = counterType](::mlir::ValueRange aggr, ::mlir::ValueRange val, ::mlir::OpBuilder& builder) {
                  std::vector<::mlir::Value> res;
                  auto one = builder.create<pgx::mlir::db::ConstantOp>(loc, counterType, builder.getI64IntegerAttr(1));
                  ::mlir::Value added1 = builder.create<pgx::mlir::db::AddOp>(loc, resultingType, aggr[currDestIdx], val[currValIdx]);
                  ::mlir::Value added2 = builder.create<pgx::mlir::db::AddOp>(loc, counterType, aggr[currDestIdx + 1], one);
                  if (attrIsNullable) {
                     ::mlir::Value isNull1 = builder.create<pgx::mlir::db::IsNullOp>(loc, builder.getI1Type(), val[currValIdx]);
                     res.push_back(builder.create<mlir::arith::SelectOp>(loc, isNull1, aggr[currDestIdx], added1));
                     res.push_back(builder.create<mlir::arith::SelectOp>(loc, isNull1, aggr[currDestIdx + 1], added2));
                  } else {
                     res.push_back(added1);
                     res.push_back(added2);
                  }

                  return res;
               });
            } else if (aggrFn.fn() == pgx::mlir::relalg::AggrFunc::count) {
               size_t currDestIdx = aggrTypes.size();
               auto initCounterVal = builder.create<pgx::mlir::db::ConstantOp>(aggregationOp.getLoc(), counterType, builder.getI64IntegerAttr(0));
               defaultValues.push_back(initCounterVal);
               aggrTypes.push_back(resultingType);
               finalizeFunctions.push_back([currDestIdx = currDestIdx, destAttr = destAttr](::mlir::ValueRange range, ::mlir::OpBuilder& builder) { return std::make_pair(destAttr, range[currDestIdx]); });

               aggregationFunctions.push_back([loc, currDestIdx = currDestIdx, attrIsNullable, currValIdx = currValIdx, counterType = counterType, resultingType = resultingType](::mlir::ValueRange aggr, ::mlir::ValueRange val, ::mlir::OpBuilder& builder) {
                  std::vector<::mlir::Value> res;
                  auto one = builder.create<pgx::mlir::db::ConstantOp>(loc, counterType, builder.getI64IntegerAttr(1));
                  ::mlir::Value value = builder.create<pgx::mlir::db::AddOp>(loc, resultingType, aggr[currDestIdx], one);
                  if (attrIsNullable) {
                     ::mlir::Value isNull2 = builder.create<pgx::mlir::db::IsNullOp>(loc, builder.getI1Type(), val[currValIdx]);
                     ::mlir::Value tmp = builder.create<mlir::arith::SelectOp>(loc, isNull2, aggr[currDestIdx], value);
                     value = tmp;
                  }

                  res.push_back(value);
                  return res;
               });
            } else if (aggrFn.fn() == pgx::mlir::relalg::AggrFunc::any) {
               size_t currDestIdx = aggrTypes.size();
               auto initVal = builder.create<pgx::mlir::util::UndefOp>(aggregationOp.getLoc(), resultingType);
               defaultValues.push_back(initVal);
               aggrTypes.push_back(resultingType);
               finalizeFunctions.push_back([currDestIdx = currDestIdx, destAttr = destAttr](::mlir::ValueRange range, ::mlir::OpBuilder& builder) { return std::make_pair(destAttr, range[currDestIdx]); });

               aggregationFunctions.push_back([currValIdx = currValIdx](::mlir::ValueRange aggr, ::mlir::ValueRange val, ::mlir::OpBuilder& builder) {
                  std::vector<::mlir::Value> res;
                  res.push_back(val[currValIdx]);
                  return res;
               });
            }
         }
         if (auto countOp = mlir::dyn_cast_or_null<pgx::mlir::relalg::CountRowsOp>(computedVal.getDefiningOp())) {
            auto loc = countOp->getLoc();

            size_t currDestIdx = aggrTypes.size();
            aggrTypes.push_back(counterType);
            auto initCounterVal = builder.create<pgx::mlir::db::ConstantOp>(aggregationOp.getLoc(), counterType, builder.getI64IntegerAttr(0));
            defaultValues.push_back(initCounterVal);
            finalizeFunctions.push_back([currDestIdx = currDestIdx, destAttr = destAttr](::mlir::ValueRange range, ::mlir::OpBuilder& builder) { return std::make_pair(destAttr, range[currDestIdx]); });

            aggregationFunctions.push_back([loc, currDestIdx = currDestIdx, counterType = counterType](::mlir::ValueRange aggr, ::mlir::ValueRange val, ::mlir::OpBuilder& builder) {
               std::vector<::mlir::Value> res;
               auto one = builder.create<pgx::mlir::db::ConstantOp>(loc, counterType, builder.getI64IntegerAttr(1));
               ::mlir::Value added2 = builder.create<pgx::mlir::db::AddOp>(loc, counterType, aggr[currDestIdx], one);
               res.push_back(added2);
               return res;
            });
         }
      };
   }
   virtual void produce(pgx::mlir::relalg::TranslatorContext& context, ::mlir::OpBuilder& builder) override {
      auto scope = context.createScope();
         analyze(builder);
         auto aggrTupleType = mlir::TupleType::get(builder.getContext(), aggrTypes);
         auto initTuple = builder.create<pgx::mlir::util::PackOp>(aggregationOp->getLoc(), aggrTupleType, defaultValues);
         keyTupleType = key.getTupleType(builder.getContext());
         valTupleType = val.getTupleType(builder.getContext());
         aggrHt = builder.create<pgx::mlir::dsa::CreateDS>(aggregationOp.getLoc(), pgx::mlir::dsa::AggregationHashtableType::get(builder.getContext(), keyTupleType, aggrTupleType), initTuple);

      auto iterEntryType = mlir::TupleType::get(builder.getContext(), {keyTupleType, aggrTupleType});
      children[0]->produce(context, builder);

      {
         auto forOp2 = builder.create<pgx::mlir::dsa::ForOp>(aggregationOp->getLoc(), ::mlir::TypeRange{}, aggrHt, ::mlir::Value(), ::mlir::ValueRange{});
         ::mlir::Block* block2 = new ::mlir::Block;
         block2->addArgument(iterEntryType, aggregationOp->getLoc());
         forOp2.getBodyRegion().push_back(block2);
         ::mlir::OpBuilder builder2(forOp2.getBodyRegion());
         auto unpacked = builder2.create<pgx::mlir::util::UnPackOp>(aggregationOp->getLoc(), forOp2.getInductionVar()).getResults();
         ::mlir::ValueRange unpackedKey;
         if (!keyTupleType.getTypes().empty()) {
            unpackedKey = builder2.create<pgx::mlir::util::UnPackOp>(aggregationOp->getLoc(), unpacked[0]).getResults();
         }
         auto unpackedAggr = builder2.create<pgx::mlir::util::UnPackOp>(aggregationOp->getLoc(), unpacked[1]).getResults();

         for (auto fn : finalizeFunctions) {
            auto [attr, val] = fn(unpackedAggr, builder2);
            context.setValueForAttribute(scope, attr, val);
         }
         key.setValuesForColumns(context, scope, unpackedKey);
         consumer->consume(this, builder2, context);
         builder2.create<pgx::mlir::dsa::YieldOp>(aggregationOp->getLoc(), ::mlir::ValueRange{});
      }
      builder.create<pgx::mlir::dsa::FreeOp>(aggregationOp->getLoc(), aggrHt);
   }
   virtual void done() override {
   }
   virtual ~AggregationTranslator() {}
};

std::unique_ptr<pgx::mlir::relalg::Translator> pgx::mlir::relalg::Translator::createAggregationTranslator(pgx::mlir::relalg::AggregationOp sortOp) {
   return std::make_unique<AggregationTranslator>(sortOp);
}