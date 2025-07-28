
#ifndef PGX_LOWER_RUNTIME_ARROWCOLUMN_H
#define PGX_LOWER_RUNTIME_ARROWCOLUMN_H
#include "helpers.h"

#include <arrow/type_fwd.h>
namespace pgx_lower::compiler::runtime {
class ArrowColumn {
   std::shared_ptr<arrow::ChunkedArray> column;

   public:
   ArrowColumn(std::shared_ptr<arrow::ChunkedArray> column) : column(column) {}
   std::shared_ptr<arrow::ChunkedArray> getColumn() const { return column; }
};
class ArrowColumnBuilder {
   size_t numValues = 0;
   std::unique_ptr<arrow::ArrayBuilder> builderUnique;
   arrow::ArrayBuilder* builder;
   ArrowColumnBuilder* childBuilder;
   std::shared_ptr<arrow::DataType> type;
   std::vector<std::shared_ptr<arrow::Array>> additionalArrays;
   ArrowColumnBuilder(std::shared_ptr<arrow::DataType> type);
   ArrowColumnBuilder(arrow::ArrayBuilder* childBuilder);
   inline void next();

   public:
   static ArrowColumnBuilder* create(VarLen32 type);
   ArrowColumnBuilder* getChildBuilder();
   void addBool(bool isValid, bool value);
   void addFixedSized(bool isValid, uint8_t* value);

   void addList(bool isValid);
   void addBinary(bool isValid, runtime::VarLen32);
   void merge(ArrowColumnBuilder* other);
   ArrowColumn* finish();
   ~ArrowColumnBuilder();
};

} // namespace pgx_lower::compiler::runtime
#endif //PGX_LOWER_RUNTIME_ARROWCOLUMN_H
