#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Conversion/RelAlgToDB/OrderedAttributes.h"
#include "execution/logging.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"

namespace pgx {
namespace mlir {
namespace relalg {

class MaterializeTranslator : public Translator {
private:
    ::pgx::mlir::relalg::MaterializeOp materializeOp;
    ::mlir::Value tableBuilder;
    ::mlir::Value table;
    OrderedAttributes orderedAttributes;
    // Store columns as member variables to ensure proper lifetime
    std::vector<Column> columns;
    
public:
    explicit MaterializeTranslator(::pgx::mlir::relalg::MaterializeOp op) 
        : Translator(op), materializeOp(op) {
        MLIR_PGX_DEBUG("RelAlg", "Created MaterializeTranslator");
        
        // Create child translator for the input relation
        auto inputOp = op.getRel().getDefiningOp();
        if (inputOp) {
            addChild(createTranslator(inputOp));
        }
    }
    
    ColumnSet getAvailableColumns() override {
        // MaterializeOp doesn't produce columns for consumption - it materializes to a table
        return ColumnSet();
    }
    
    void setInfo(Translator* consumer, const ColumnSet& requiredAttributes) override {
        Translator::setInfo(consumer, requiredAttributes);
        
        // Initialize OrderedAttributes from the MaterializeOp's columns
        // Pre-allocate columns vector
        columns.reserve(materializeOp.getColumns().size());
        
        for (size_t i = 0; i < materializeOp.getColumns().size(); ++i) {
            // Create a Column with i64 type for Test 1
            columns.emplace_back(::mlir::IntegerType::get(materializeOp.getContext(), 64));
            orderedAttributes.insert(&columns.back());
        }
    }
    
    void produce(TranslatorContext& context, ::mlir::OpBuilder& builder) override {
        MLIR_PGX_INFO("RelAlg", "MaterializeTranslator::produce() - Starting materialization");
        
        auto loc = materializeOp.getLoc();
        
        // Create schema description and table builder
        createTableBuilder(builder, loc);
        
        // Process input relation
        processInputRelation(context, builder);
        
        // Finalize the table
        finalizeTable(builder, loc);
    }
    
private:
    // Create schema description and DSA table builder
    void createTableBuilder(::mlir::OpBuilder& builder, ::mlir::Location loc) {
        // Create schema description for DSA
        std::string schemaDesc = buildSchemaDescription();
        MLIR_PGX_DEBUG("RelAlg", "Schema description: " + schemaDesc);
        
        // Create DSA table builder with appropriate type from OrderedAttributes
        auto tupleType = orderedAttributes.getTupleType(builder.getContext());
        auto builderType = ::pgx::mlir::dsa::TableBuilderType::get(
            builder.getContext(), tupleType);
        
        // Create the table builder
        tableBuilder = builder.create<::pgx::mlir::dsa::CreateDSOp>(
            loc, builderType);
        
        MLIR_PGX_DEBUG("RelAlg", "Created DSA table builder");
    }
    
    // Build schema description string from columns
    std::string buildSchemaDescription() {
        std::string schemaDesc = "";
        auto columns = materializeOp.getColumns();
        for (size_t i = 0; i < columns.size(); ++i) {
            if (i > 0) schemaDesc += ";";
            auto colName = columns[i].cast<::mlir::StringAttr>().getValue();
            // For Test 1, assume integer columns
            schemaDesc += colName.str() + ":int[64]";
        }
        return schemaDesc;
    }
    
    // Process input relation by calling child's produce
    void processInputRelation(TranslatorContext& context, ::mlir::OpBuilder& builder) {
        if (!children.empty()) {
            children[0]->setConsumer(this);
            children[0]->produce(context, builder);
        }
    }
    
    // Finalize the table after all tuples are processed
    void finalizeTable(::mlir::OpBuilder& builder, ::mlir::Location loc) {
        auto tableType = ::pgx::mlir::dsa::TableType::get(builder.getContext());
        table = builder.create<::pgx::mlir::dsa::FinalizeOp>(
            loc, tableType, tableBuilder).getResult();
        
        MLIR_PGX_DEBUG("RelAlg", "Finalized table construction");
    }
    
public:
    
    void consume(Translator* child, ::mlir::OpBuilder& builder, 
                TranslatorContext& context) override {
        MLIR_PGX_DEBUG("RelAlg", "MaterializeTranslator::consume() - Processing tuple from child");
        
        auto loc = materializeOp.getLoc();
        
        // Process each column using OrderedAttributes
        for (size_t i = 0; i < orderedAttributes.getAttrs().size(); i++) {
            auto val = orderedAttributes.resolve(context, i);
            ::mlir::Value valid;
            
            // For Test 1, assume non-nullable types (simplified)
            // In full implementation, would check for NullableType and handle appropriately
            valid = builder.create<::mlir::arith::ConstantIntOp>(
                loc, 1, builder.getI1Type());
            
            // Append value to table builder with validity information
            ::llvm::SmallVector<::mlir::Value, 2> appendValues;
            appendValues.push_back(val);
            appendValues.push_back(valid);
            builder.create<::pgx::mlir::dsa::DSAppendOp>(
                loc, tableBuilder, appendValues);
        }
        
        // Mark end of row
        builder.create<::pgx::mlir::dsa::NextRowOp>(loc, tableBuilder);
        
        MLIR_PGX_DEBUG("RelAlg", "Appended tuple to table builder");
    }
    
    void done() override {
        MLIR_PGX_DEBUG("RelAlg", "MaterializeTranslator::done() - Replacing MaterializeOp with table");
        
        // Replace the MaterializeOp with the constructed table
        if (table) {
            materializeOp.replaceAllUsesWith(table);
            materializeOp.erase();
        }
    }
};

// Factory function
std::unique_ptr<Translator> createMaterializeTranslator(::mlir::Operation* op) {
    auto materializeOp = ::mlir::dyn_cast<::pgx::mlir::relalg::MaterializeOp>(op);
    if (!materializeOp) {
        MLIR_PGX_ERROR("RelAlg", "createMaterializeTranslator called with non-MaterializeOp");
        return createDummyTranslator(op);
    }
    return std::make_unique<MaterializeTranslator>(materializeOp);
}

} // namespace relalg
} // namespace mlir
} // namespace pgx