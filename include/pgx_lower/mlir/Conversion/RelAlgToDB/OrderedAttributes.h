#ifndef PGX_LOWER_MLIR_CONVERSION_RELALGTODB_ORDEREDATTRIBUTES_H
#define PGX_LOWER_MLIR_CONVERSION_RELALGTODB_ORDEREDATTRIBUTES_H

#include "mlir/Conversion/RelAlgToDB/TranslatorContext.h"
#include "mlir/Dialect/RelAlg/ColumnSet.h"
#include "mlir/Dialect/RelAlg/IR/Column.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include <vector>

namespace pgx {
namespace mlir {
namespace relalg {

// OrderedAttributes maintains an ordered list of columns with their types
// Used to manage column resolution and tuple type construction
class OrderedAttributes {
private:
    std::vector<::mlir::Type> types;
    std::vector<const Column*> attrs;

public:
    // Constructors
    OrderedAttributes() = default;
    
    // Create from a ColumnSet (converts to ordered)
    static OrderedAttributes fromColumns(const ColumnSet& columnSet) {
        OrderedAttributes res;
        for (auto* attr : columnSet) {
            res.insert(attr);
        }
        return res;
    }
    
    // Create from a vector of columns
    static OrderedAttributes fromVec(const std::vector<const Column*>& vec) {
        OrderedAttributes res;
        for (auto* attr : vec) {
            res.insert(attr);
        }
        return res;
    }
    
    // Resolve a column at a specific position to its corresponding value
    ::mlir::Value resolve(TranslatorContext& context, size_t pos) {
        assert(pos < attrs.size() && "Position out of bounds");
        return context.getValueForAttribute(attrs[pos]);
    }
    
    // Insert a new column with its type
    size_t insert(const Column* attr, ::mlir::Type alternativeType = {}) {
        attrs.push_back(attr);
        if (attr && attr->type) {
            types.push_back(attr->type);
        } else {
            assert(alternativeType && "Must provide type when column has no type");
            types.push_back(alternativeType);
        }
        return attrs.size() - 1;
    }
    
    // Get tuple type representing all columns
    ::mlir::TupleType getTupleType(::mlir::MLIRContext* ctxt, 
                                   const std::vector<::mlir::Type>& additional = {}) {
        std::vector<::mlir::Type> allTypes(additional);
        allTypes.insert(allTypes.end(), types.begin(), types.end());
        return ::mlir::TupleType::get(ctxt, allTypes);
    }
    
    // Set values for all columns in the context
    void setValuesForColumns(TranslatorContext& context, 
                           TranslatorContext::AttributeResolverScope& scope, 
                           ::mlir::ValueRange values) {
        assert(values.size() == attrs.size() && "Value count mismatch");
        for (size_t i = 0; i < attrs.size(); i++) {
            if (attrs[i]) {
                context.setValueForAttribute(scope, attrs[i], values[i]);
            }
        }
    }
    
    // Getters
    const std::vector<const Column*>& getAttrs() const {
        return attrs;
    }
    
    const std::vector<::mlir::Type>& getTypes() const {
        return types;
    }
    
    size_t size() const {
        return attrs.size();
    }
    
    // Get position of a specific column
    size_t getPos(const Column* attr) const {
        auto it = std::find(attrs.begin(), attrs.end(), attr);
        return it != attrs.end() ? std::distance(attrs.begin(), it) : size();
    }
    
    // Check if contains a column
    bool contains(const Column* attr) const {
        return std::find(attrs.begin(), attrs.end(), attr) != attrs.end();
    }
};

} // namespace relalg
} // namespace mlir
} // namespace pgx

#endif // PGX_LOWER_MLIR_CONVERSION_RELALGTODB_ORDEREDATTRIBUTES_H