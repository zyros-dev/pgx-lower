#include "lingodb/mlir/Conversion/RelAlgToDB/Translator.h"
#include "lingodb/mlir/Dialect/DB/IR/DBOps.h"
#include "lingodb/mlir/Dialect/DB/IR/DBTypes.h"
#include "lingodb/mlir/Dialect/DSA/IR/DSAOps.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/mlir/Dialect/util/UtilOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace {

struct RowScanField {
    const mlir::relalg::Column* column{};
    mlir::db::PgRowFieldAttr field{};
};

static bool moduleRequestsRowLowerPath(mlir::Operation* op) {
    auto module = op->getParentOfType<mlir::ModuleOp>();
    if (!module) {
        return false;
    }
    auto attr = module->getAttrOfType<mlir::StringAttr>("pgx_lower.lower_path");
    return attr && attr.getValue() == "row";
}

static uint32_t parseRelid(llvm::StringRef tableIdentifier) {
    const auto marker = tableIdentifier.find("|oid:");
    if (marker == llvm::StringRef::npos) {
        return 0;
    }
    uint32_t relid = 0;
    (void)tableIdentifier.drop_front(marker + 5).getAsInteger(10, relid);
    return relid;
}

static mlir::relalg::ColumnDefAttr lookupColumnDef(mlir::relalg::BaseTableOp baseTableOp, llvm::StringRef name) {
    for (auto namedAttr : baseTableOp.getColumnsAttr().getValue()) {
        if (namedAttr.getName() == name) {
            return namedAttr.getValue().dyn_cast_or_null<mlir::relalg::ColumnDefAttr>();
        }
    }
    return {};
}

static std::vector<RowScanField> buildRowScanFields(mlir::relalg::BaseTableOp baseTableOp,
                                                    mlir::relalg::ColumnSet requiredAttributes, mlir::OpBuilder& builder) {
    std::vector<RowScanField> fields;
    auto tableIdentifier = baseTableOp->getAttrOfType<mlir::StringAttr>("table_identifier");
    auto columnOrder = baseTableOp->getAttrOfType<mlir::ArrayAttr>("column_order");
    if (!tableIdentifier || !columnOrder) {
        return fields;
    }

    const uint32_t relid = parseRelid(tableIdentifier.getValue());
    uint32_t rowIndex = 0;
    uint32_t physicalIndex = 0;
    for (auto attr : columnOrder) {
        auto nameAttr = attr.dyn_cast_or_null<mlir::StringAttr>();
        if (!nameAttr) {
            physicalIndex++;
            continue;
        }
        auto columnDef = lookupColumnDef(baseTableOp, nameAttr.getValue());
        if (!columnDef || !requiredAttributes.contains(&columnDef.getColumn())) {
            physicalIndex++;
            continue;
        }

        mlir::Type type = columnDef.getColumn().type;
        const auto oid = mlir::db::getPgTypeOid(type);
        const auto typmod = mlir::db::getPgTypmod(type);
        const auto collation = mlir::db::getPgCollation(type);
        const auto nullability = mlir::db::getPgNullability(type);
        auto field = mlir::db::PgRowFieldAttr::get(builder.getContext(), rowIndex, relid, 1,
                                                   static_cast<int16_t>(physicalIndex + 1),
                                                   builder.getStringAttr(nameAttr.getValue()), type, oid, typmod,
                                                   collation, nullability, false, mlir::db::PgRowFieldOrigin::base);
        fields.push_back({&columnDef.getColumn(), field});
        rowIndex++;
        physicalIndex++;
    }
    return fields;
}

} // namespace

class BaseTableTranslator : public mlir::relalg::Translator {
    static bool registered;
    mlir::relalg::BaseTableOp baseTableOp;

   public:
    BaseTableTranslator(mlir::relalg::BaseTableOp baseTableOp)
    : mlir::relalg::Translator(baseTableOp)
    , baseTableOp(baseTableOp) {}
    virtual void consume(mlir::relalg::Translator* child, mlir::OpBuilder& builder,
                         mlir::relalg::TranslatorContext& context) override {
        assert(false && "should not happen");
    }
    void produceRowPath(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) {
        auto scope = context.createScope();
        auto fields = buildRowScanFields(baseTableOp, requiredAttributes, builder);
        std::vector<mlir::db::PgRowFieldAttr> fieldAttrs;
        fieldAttrs.reserve(fields.size());
        for (const auto& field : fields) {
            fieldAttrs.push_back(field.field);
        }

        auto schema = mlir::db::PgRowSchemaAttr::get(builder.getContext(), fieldAttrs);
        auto rowType = mlir::db::PgRowType::get(builder.getContext(), schema);
        const uint32_t relid = fieldAttrs.empty()
                                   ? parseRelid(
                                         baseTableOp->getAttrOfType<mlir::StringAttr>("table_identifier").getValue())
                                   : fieldAttrs.front().getRelid();
        auto relidValue = builder.create<mlir::arith::ConstantIntOp>(baseTableOp->getLoc(), relid, 32);
        auto scan = builder.create<mlir::db::RuntimeCall>(
            baseTableOp->getLoc(), rowType, builder.getStringAttr("PgRowScanStart"), mlir::ValueRange{relidValue});

        auto whileOp = builder.create<mlir::scf::WhileOp>(baseTableOp->getLoc(), mlir::TypeRange{rowType},
                                                          mlir::ValueRange{scan.getRes()});
        auto* before = new mlir::Block();
        before->addArgument(rowType, baseTableOp->getLoc());
        whileOp.getBefore().push_back(before);
        mlir::OpBuilder beforeBuilder = mlir::OpBuilder::atBlockEnd(before);
        auto hasNext = beforeBuilder.create<mlir::db::RuntimeCall>(baseTableOp->getLoc(), beforeBuilder.getI1Type(),
                                                                   beforeBuilder.getStringAttr("PgRowScanNext"),
                                                                   mlir::ValueRange{before->getArgument(0)});
        beforeBuilder.create<mlir::scf::ConditionOp>(baseTableOp->getLoc(), hasNext.getRes(),
                                                     mlir::ValueRange{before->getArgument(0)});

        auto* after = new mlir::Block();
        after->addArgument(rowType, baseTableOp->getLoc());
        whileOp.getAfter().push_back(after);
        mlir::OpBuilder afterBuilder = mlir::OpBuilder::atBlockEnd(after);
        mlir::Value row = after->getArgument(0);
        for (const auto& field : fields) {
            auto value = afterBuilder.create<mlir::db::PgRowGetOp>(baseTableOp->getLoc(), row, field.field.getIndex());
            context.setValueForAttribute(scope, field.column, value.getResult());
        }
        consumer->consume(this, afterBuilder, context);
        afterBuilder.create<mlir::scf::YieldOp>(baseTableOp->getLoc(), mlir::ValueRange{row});

        builder.setInsertionPointAfter(whileOp);
        builder.create<mlir::db::RuntimeCall>(baseTableOp->getLoc(), mlir::TypeRange{},
                                              builder.getStringAttr("PgRowScanEnd"),
                                              mlir::ValueRange{whileOp.getResults()[0]});
    }
    virtual void produce(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
        if (moduleRequestsRowLowerPath(baseTableOp)) {
            produceRowPath(context, builder);
            return;
        }
        auto scope = context.createScope();
        using namespace mlir;
        std::vector<mlir::Type> types;
        std::vector<const mlir::relalg::Column*> cols;
        std::vector<mlir::Attribute> columnNames;
        std::string tableName = baseTableOp->getAttr("table_identifier").cast<mlir::StringAttr>().str();
        std::string scanDescription = R"({ "table": ")" + tableName + R"(", "columns": [ )";
        bool first = true;
        for (auto namedAttr : baseTableOp.getColumnsAttr().getValue()) {
            auto identifier = namedAttr.getName();
            auto attr = namedAttr.getValue();
            auto attrDef = attr.dyn_cast_or_null<mlir::relalg::ColumnDefAttr>();
            if (requiredAttributes.contains(&attrDef.getColumn())) {
                if (!first) {
                    scanDescription += ",";
                } else {
                    first = false;
                }
                scanDescription += "\"" + identifier.str() + "\"";
                columnNames.push_back(builder.getStringAttr(identifier.strref()));
                types.push_back(getBaseType(attrDef.getColumn().type));
                cols.push_back(&attrDef.getColumn());
            }
        }
        scanDescription += "] }";

        auto tupleType = mlir::TupleType::get(builder.getContext(), types);
        auto recordBatch = mlir::dsa::RecordBatchType::get(builder.getContext(), tupleType);
        mlir::Type chunkIterable = mlir::dsa::GenericIterableType::get(builder.getContext(), recordBatch,
                                                                       "table_chunk_iterator");

        auto chunkIterator = builder.create<mlir::dsa::ScanSource>(baseTableOp->getLoc(), chunkIterable,
                                                                   builder.getStringAttr(scanDescription));

        auto forOp = builder.create<mlir::dsa::ForOp>(baseTableOp->getLoc(), mlir::TypeRange{}, chunkIterator,
                                                      mlir::Value(), mlir::ValueRange{});
        mlir::Block* block = new mlir::Block;
        block->addArgument(recordBatch, baseTableOp->getLoc());
        forOp.getBodyRegion().push_back(block);
        mlir::OpBuilder builder1(forOp.getBodyRegion());
        auto forOp2 = builder1.create<mlir::dsa::ForOp>(baseTableOp->getLoc(), mlir::TypeRange{},
                                                        forOp.getInductionVar(), mlir::Value(), mlir::ValueRange{});
        mlir::Block* block2 = new mlir::Block;
        block2->addArgument(recordBatch.getElementType(), baseTableOp->getLoc());
        forOp2.getBodyRegion().push_back(block2);
        mlir::OpBuilder builder2(forOp2.getBodyRegion());
        size_t i = 0;
        for (const auto* attr : cols) {
            std::vector<mlir::Type> types;
            types.push_back(getBaseType(attr->type));
            if (attr->type.isa<mlir::db::NullableType>()) {
                types.push_back(builder.getI1Type());
            }
            auto atOp = builder2.create<mlir::dsa::At>(baseTableOp->getLoc(), types, forOp2.getInductionVar(), i++);
            if (attr->type.isa<mlir::db::NullableType>()) {
                mlir::Value isNull = builder2.create<mlir::db::NotOp>(baseTableOp->getLoc(), atOp.getValid());
                mlir::Value val = builder2.create<mlir::db::AsNullableOp>(baseTableOp->getLoc(), attr->type,
                                                                          atOp.getVal(), isNull);
                context.setValueForAttribute(scope, attr, val);
            } else {
                context.setValueForAttribute(scope, attr, atOp.getVal());
            }
        }
        consumer->consume(this, builder2, context);
        builder2.create<mlir::dsa::YieldOp>(baseTableOp->getLoc());
        builder1.create<mlir::dsa::YieldOp>(baseTableOp->getLoc());
    }
    virtual ~BaseTableTranslator() {}
};

std::unique_ptr<mlir::relalg::Translator> mlir::relalg::Translator::createBaseTableTranslator(mlir::relalg::BaseTableOp baseTableOp) {
   return std::make_unique<BaseTableTranslator>(baseTableOp);
}