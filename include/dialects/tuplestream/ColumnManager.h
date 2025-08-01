#ifndef PGX_LOWER_COLUMNMANAGER_H
#define PGX_LOWER_COLUMNMANAGER_H

#include "Column.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include <memory>
#include <unordered_map>

// Include attribute definitions before namespace to get proper definitions
#ifndef TUPLESTREAM_ATTRS_INCLUDED
#define TUPLESTREAM_ATTRS_INCLUDED
#define GET_ATTRDEF_CLASSES
#include "TupleStreamAttrs.h.inc"
#endif

namespace pgx_lower::compiler::dialect::tuples {

// No need for forward declarations now

class ColumnManager {
public:
    ColumnManager() = default;
    ColumnManager(const ColumnManager&) = delete;
    ColumnManager& operator=(const ColumnManager&) = delete;

    void setContext(mlir::MLIRContext* context) {
        this->context = context;
    }

    std::shared_ptr<Column> get(llvm::StringRef scope, llvm::StringRef attribute) {
        auto pair = std::make_pair(std::string(scope), std::string(attribute));
        if (!attributes.count(pair)) {
            auto attr = std::make_shared<Column>();
            attributes[pair] = attr;
            attributesRev[attr.get()] = pair;
        }
        return attributes[pair];
    }

    ColumnDefAttr createDef(mlir::SymbolRefAttr name, mlir::Attribute fromExisting = mlir::Attribute());
    ColumnDefAttr createDef(llvm::StringRef scope, llvm::StringRef name, mlir::Attribute fromExisting = mlir::Attribute());
    ColumnDefAttr createDef(const Column* attr, mlir::Attribute fromExisting = mlir::Attribute());
    ColumnRefAttr createRef(mlir::SymbolRefAttr name);
    ColumnRefAttr createRef(const Column* attr);
    ColumnRefAttr createRef(llvm::StringRef scope, llvm::StringRef name);

    std::pair<std::string, std::string> getName(const Column* attr) {
        return attributesRev.at(attr);
    }

    std::string getUniqueScope(llvm::StringRef base) {
        if (scopeUnifier.count(std::string(base))) {
            scopeUnifier[std::string(base)] += 1;
            return std::string(base) + std::string("_u_") + std::to_string(scopeUnifier[std::string(base)]);
        } else {
            scopeUnifier[std::string(base)] = 0;
            return std::string(base);
        }
    }

private:
    mlir::MLIRContext* context;
    
    struct HashPair {
        template <class T1, class T2>
        size_t operator()(const std::pair<T1, T2>& p) const {
            auto hash1 = std::hash<T1>{}(p.first);
            auto hash2 = std::hash<T2>{}(p.second);
            return hash1 ^ hash2;
        }
    };
    
    std::unordered_map<std::pair<std::string, std::string>, std::shared_ptr<Column>, HashPair> attributes;
    std::unordered_map<const Column*, std::pair<std::string, std::string>> attributesRev;
    std::unordered_map<std::string, size_t> scopeUnifier;
};

} // namespace pgx_lower::compiler::dialect::tuples

#endif // PGX_LOWER_COLUMNMANAGER_H