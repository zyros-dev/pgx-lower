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

// EntryStorageHelper is defined in SubOpToControlFlowUtilities.cpp

//===----------------------------------------------------------------------===//
// Atomic Reduction Implementation Helper
//===----------------------------------------------------------------------===//

/**
 * @brief Implements atomic reduction operations for performance-critical aggregations
 * 
 * This function analyzes the reduction operation and attempts to convert it to
 * hardware-accelerated atomic operations when possible. It supports:
 * - Floating-point additions (AtomicRMWKind::addf)
 * - Integer additions (AtomicRMWKind::addi) 
 * - Bitwise OR operations (AtomicRMWKind::ori)
 * 
 * For operations that cannot be directly mapped to atomic primitives,
 * it falls back to generic atomic read-modify-write operations.
 * 
 * @param reduceOp The reduction operation to implement atomically
 * @param rewriter The MLIR rewriter for creating new operations
 * @param valueRef Reference to the memory location being reduced
 * @param mapping Column mapping for resolving operands
 */
static void implementAtomicReduce(subop::ReduceOp reduceOp, SubOpRewriter& rewriter, mlir::Value valueRef, ColumnMapping& mapping) {
   auto loc = reduceOp->getLoc();
   auto elementType = mlir::cast<util::RefType>(valueRef.getType()).getElementType();
   auto origElementType = mlir::cast<util::RefType>(valueRef.getType()).getElementType();
   
   if (elementType.isInteger(1)) {
      elementType = rewriter.getI8Type();
      valueRef = rewriter.create<util::GenericMemrefCastOp>(loc, util::RefType::get(rewriter.getContext(), elementType), valueRef);
   }
   
   auto memRefType = mlir::MemRefType::get({}, elementType);
   auto memRef = rewriter.create<util::ToMemrefOp>(reduceOp->getLoc(), memRefType, valueRef);
   auto returnOp = mlir::cast<tuples::ReturnOp>(reduceOp.getRegion().front().getTerminator());
   
   ::mlir::arith::AtomicRMWKind atomicKind = mlir::arith::AtomicRMWKind::maximumf; //maxf is invalid value;
   mlir::Value memberValue = reduceOp.getRegion().front().getArguments().back();
   mlir::Value atomicOperand;

   if (auto* defOp = returnOp.getResults()[0].getDefiningOp()) {
      // Pattern match for floating-point addition
      if (auto addFOp = mlir::dyn_cast_or_null<mlir::arith::AddFOp>(defOp)) {
         if (addFOp.getLhs() == memberValue) {
            atomicKind = mlir::arith::AtomicRMWKind::addf;
            atomicOperand = addFOp.getRhs();
         } else if (addFOp.getRhs() == memberValue) {
            atomicKind = mlir::arith::AtomicRMWKind::addf;
            atomicOperand = addFOp.getLhs();
         }
      }

      // Pattern match for integer addition
      if (auto addIOp = mlir::dyn_cast_or_null<mlir::arith::AddIOp>(defOp)) {
         if (addIOp.getLhs() == memberValue) {
            atomicKind = mlir::arith::AtomicRMWKind::addi;
            atomicOperand = addIOp.getRhs();
         } else if (addIOp.getRhs() == memberValue) {
            atomicKind = mlir::arith::AtomicRMWKind::addi;
            atomicOperand = addIOp.getLhs();
         }
      }
      
      // Pattern match for bitwise OR
      if (auto orIOp = mlir::dyn_cast_or_null<mlir::arith::OrIOp>(defOp)) {
         if (orIOp.getLhs() == memberValue) {
            atomicKind = mlir::arith::AtomicRMWKind::ori;
            atomicOperand = orIOp.getRhs();
         } else if (orIOp.getRhs() == memberValue) {
            atomicKind = mlir::arith::AtomicRMWKind::ori;
            atomicOperand = orIOp.getLhs();
         }
      }
      //TODO: Add support for additional atomic operations (max, min, xor, and, etc.)
   }

   if (atomicOperand) {
      // Direct atomic operation path - faster for simple reductions
      std::vector<mlir::Value> arguments;
      for (auto attr : reduceOp.getColumns()) {
         mlir::Value arg = mapping.resolve(reduceOp, mlir::cast<tuples::ColumnRefAttr>(attr));
         if (arg.getType() != mlir::cast<tuples::ColumnRefAttr>(attr).getColumn().type) {
            arg = rewriter.create<mlir::UnrealizedConversionCastOp>(reduceOp->getLoc(), mlir::cast<tuples::ColumnRefAttr>(attr).getColumn().type, arg).getResult(0);
         }
         arguments.push_back(arg);
      }

      arguments.push_back(rewriter.create<util::UndefOp>(loc, origElementType));
      mlir::IRMapping mapper;
      mlir::Block* b = &reduceOp.getRegion().front();
      assert(b->getNumArguments() == arguments.size());
      for (auto i = 0ull; i < b->getNumArguments(); i++) {
         mapper.map(b->getArgument(i), arguments[i]);
      }
      for (auto& x : b->getOperations()) {
         if (&x != returnOp.getOperation()) {
            rewriter.insert(rewriter.clone(&x, mapper));
         }
      }
      atomicOperand = mapper.lookup(atomicOperand);
      
      // Promote boolean operands to i8 for atomic operations
      if (atomicOperand.getType().isInteger(1)) {
         atomicOperand = rewriter.create<arith::ExtUIOp>(loc, rewriter.getI8Type(), atomicOperand);
      }
      
      rewriter.create<memref::AtomicRMWOp>(loc, atomicOperand.getType(), atomicKind, atomicOperand, memRef, ValueRange{});
      rewriter.eraseOp(reduceOp);
   } else {
      // Generic atomic operation path - handles complex reduction logic
      auto genericOp = rewriter.create<memref::GenericAtomicRMWOp>(loc, memRef, mlir::ValueRange{});
      std::vector<mlir::Value> arguments;
      for (auto attr : reduceOp.getColumns()) {
         mlir::Value arg = mapping.resolve(reduceOp, mlir::cast<tuples::ColumnRefAttr>(attr));
         if (arg.getType() != mlir::cast<tuples::ColumnRefAttr>(attr).getColumn().type) {
            arg = rewriter.create<mlir::UnrealizedConversionCastOp>(reduceOp->getLoc(), mlir::cast<tuples::ColumnRefAttr>(attr).getColumn().type, arg).getResult(0);
         }
         arguments.push_back(arg);
      }

      arguments.push_back(genericOp.getCurrentValue());

      rewriter.atStartOf(genericOp.getBody(), [&](SubOpRewriter& rewriter) {
         rewriter.inlineBlock<tuples::ReturnOpAdaptor>(&reduceOp.getRegion().front(), arguments, [&](tuples::ReturnOpAdaptor adaptor) {
            mlir::Value atomicResult = adaptor.getResults()[0];
            
            // Promote boolean results to i8 for atomic operations
            if (atomicResult.getType().isInteger(1)) {
               atomicResult = rewriter.create<arith::ExtUIOp>(loc, rewriter.getI8Type(), atomicResult);
            }
            
            rewriter.create<memref::AtomicYieldOp>(loc, atomicResult);
            rewriter.eraseOp(reduceOp);
         });
      });
   }
}

//===----------------------------------------------------------------------===//
// ReduceContinuousRefLowering - Non-Atomic Continuous Reference Reduction
//===----------------------------------------------------------------------===//

/**
 * @brief Lowers reduce operations on continuous reference types to control flow
 * 
 * This pattern handles non-atomic reduction operations on continuous reference types.
 * Continuous references provide direct access to array elements in memory, enabling
 * efficient bulk processing operations.
 * 
 * The lowering process:
 * 1. Unpacks the continuous reference to get base pointer and index
 * 2. Creates direct memory access to the target element
 * 3. Loads current aggregate values from memory
 * 4. Executes the reduction function inline
 * 5. Stores updated aggregate values back to memory
 */
class ReduceContinuousRefLowering : public SubOpTupleStreamConsumerConversionPattern<subop::ReduceOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::ReduceOp>::SubOpTupleStreamConsumerConversionPattern;

   LogicalResult matchAndRewrite(subop::ReduceOp reduceOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      auto continuousRefEntryType = mlir::dyn_cast_or_null<subop::ContinuousEntryRefType>(reduceOp.getRef().getColumn().type);
      if (!continuousRefEntryType) { 
         PGX_DEBUG("ReduceContinuousRefLowering: Not a continuous reference type");
         return failure(); 
      }
      
      if (reduceOp->hasAttr("atomic")) {
         PGX_DEBUG("ReduceContinuousRefLowering: Atomic operation, delegating to atomic lowering");
         return mlir::failure();
      }
      
      PGX_DEBUG("ReduceContinuousRefLowering: Processing non-atomic continuous reference reduction");
      
      // Unpack the continuous reference to get [index, buffer]
      llvm::SmallVector<mlir::Value> unpackedReference;
      rewriter.createOrFold<util::UnPackOp>(unpackedReference, reduceOp->getLoc(), mapping.resolve(reduceOp, reduceOp.getRef()));
      
      // Set up storage helper for managing aggregate state
      EntryStorageHelper storageHelper(reduceOp, continuousRefEntryType.getMembers(), continuousRefEntryType.hasLock(), typeConverter);
      auto ptrType = storageHelper.getRefType();
      
      // Calculate memory address: base + (index * element_size) 
      auto baseRef = rewriter.create<util::BufferGetRef>(reduceOp->getLoc(), ptrType, unpackedReference[1]);
      auto elementRef = rewriter.create<util::ArrayElementPtrOp>(reduceOp->getLoc(), ptrType, baseRef, unpackedReference[0]);
      
      // Load current aggregate values from memory
      auto values = storageHelper.getValueMap(elementRef, rewriter, reduceOp->getLoc());
      
      // Prepare arguments for the reduction function
      std::vector<mlir::Value> arguments;
      
      // Add input column values
      for (auto attr : reduceOp.getColumns()) {
         mlir::Value arg = mapping.resolve(reduceOp, mlir::cast<tuples::ColumnRefAttr>(attr));
         if (arg.getType() != mlir::cast<tuples::ColumnRefAttr>(attr).getColumn().type) {
            arg = rewriter.create<mlir::UnrealizedConversionCastOp>(reduceOp->getLoc(), mlir::cast<tuples::ColumnRefAttr>(attr).getColumn().type, arg).getResult(0);
         }
         arguments.push_back(arg);
      }
      
      // Add current aggregate member values
      for (auto member : reduceOp.getMembers()) {
         mlir::Value arg = values.get(mlir::cast<mlir::StringAttr>(member).str());
         if (arg.getType() != reduceOp.getRegion().getArgument(arguments.size()).getType()) {
            arg = rewriter.create<mlir::UnrealizedConversionCastOp>(reduceOp->getLoc(), reduceOp.getRegion().getArgument(arguments.size()).getType(), arg).getResult(0);
         }
         arguments.push_back(arg);
      }

      // Inline the reduction function and handle results
      rewriter.inlineBlock<tuples::ReturnOpAdaptor>(&reduceOp.getRegion().front(), arguments, [&](tuples::ReturnOpAdaptor adaptor) {
         // Update aggregate values with function results
         for (size_t i = 0; i < reduceOp.getMembers().size(); i++) {
            const std::string memberName = mlir::cast<mlir::StringAttr>(reduceOp.getMembers()[i]).str();
            auto& memberVal = values.get(memberName);
            auto updatedVal = adaptor.getResults()[i];
            if (updatedVal.getType() != memberVal.getType()) {
               updatedVal = rewriter.create<mlir::UnrealizedConversionCastOp>(reduceOp->getLoc(), memberVal.getType(), updatedVal).getResult(0);
            }
            memberVal = updatedVal;
         }
         
         // Store updated values back to memory
         values.store();
         rewriter.eraseOp(reduceOp);
      });

      return success();
   }
};

//===----------------------------------------------------------------------===//
// ReduceContinuousRefAtomicLowering - Atomic Continuous Reference Reduction  
//===----------------------------------------------------------------------===//

/**
 * @brief Lowers atomic reduce operations on continuous reference types
 * 
 * This pattern handles atomic reduction operations on continuous reference types,
 * providing thread-safe aggregation for parallel processing scenarios.
 * 
 * Atomic reductions are critical for:
 * - Parallel aggregation in multi-threaded query execution
 * - Lock-free data structures for high-performance analytics
 * - Avoiding race conditions in concurrent access patterns
 * 
 * The atomic lowering leverages hardware-level atomic instructions when possible,
 * falling back to generic atomic read-modify-write operations for complex logic.
 */
class ReduceContinuousRefAtomicLowering : public SubOpTupleStreamConsumerConversionPattern<subop::ReduceOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::ReduceOp>::SubOpTupleStreamConsumerConversionPattern;

   LogicalResult matchAndRewrite(subop::ReduceOp reduceOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      auto continuousRefEntryType = mlir::dyn_cast_or_null<subop::ContinuousEntryRefType>(reduceOp.getRef().getColumn().type);
      if (!continuousRefEntryType) { 
         PGX_DEBUG("ReduceContinuousRefAtomicLowering: Not a continuous reference type");
         return failure(); 
      }
      
      if (!reduceOp->hasAttr("atomic")) {
         PGX_DEBUG("ReduceContinuousRefAtomicLowering: Not an atomic operation, delegating to non-atomic lowering");
         return mlir::failure();
      }
      
      PGX_DEBUG("ReduceContinuousRefAtomicLowering: Processing atomic continuous reference reduction");
      
      auto loc = reduceOp->getLoc();
      
      // Unpack the continuous reference to get [index, buffer]
      llvm::SmallVector<mlir::Value> unpackedReference;
      rewriter.createOrFold<util::UnPackOp>(unpackedReference, reduceOp->getLoc(), mapping.resolve(reduceOp, reduceOp.getRef()));
      
      // Set up storage helper for managing aggregate state
      EntryStorageHelper storageHelper(reduceOp, continuousRefEntryType.getMembers(), continuousRefEntryType.hasLock(), typeConverter);
      auto ptrType = storageHelper.getRefType();
      
      // Calculate memory address: base + (index * element_size)
      auto baseRef = rewriter.create<util::BufferGetRef>(reduceOp->getLoc(), ptrType, unpackedReference[1]);
      auto elementRef = rewriter.create<util::ArrayElementPtrOp>(reduceOp->getLoc(), ptrType, baseRef, unpackedReference[0]);
      
      // Get direct pointer to the specific member being reduced atomically
      // Note: Atomic operations currently support single-member reductions
      auto valueRef = storageHelper.getPointer(elementRef, mlir::cast<mlir::StringAttr>(reduceOp.getMembers()[0]).str(), rewriter, loc);
      
      // Delegate to the atomic implementation helper
      implementAtomicReduce(reduceOp, rewriter, valueRef, mapping);

      return success();
   }
};

//===----------------------------------------------------------------------===//
// ReduceOpLowering - Generic State Reference Reduction
//===----------------------------------------------------------------------===//

/**
 * @brief Lowers reduce operations on generic state references to control flow
 * 
 * This is the most general reduction lowering pattern, handling both atomic and
 * non-atomic operations on state entry references. It serves as the fallback
 * for reduction operations that don't match more specific patterns.
 * 
 * Key capabilities:
 * - Supports both atomic and non-atomic reduction modes
 * - Handles complex multi-member aggregate states  
 * - Provides type conversion and alignment for heterogeneous data
 * - Manages lock-based synchronization when required
 * 
 * The pattern automatically selects between atomic and non-atomic implementations
 * based on the operation attributes, ensuring optimal performance for each use case.
 */
class ReduceOpLowering : public SubOpTupleStreamConsumerConversionPattern<subop::ReduceOp> {
   public:
   using SubOpTupleStreamConsumerConversionPattern<subop::ReduceOp>::SubOpTupleStreamConsumerConversionPattern;

   LogicalResult matchAndRewrite(subop::ReduceOp reduceOp, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const override {
      auto referenceType = mlir::cast<subop::StateEntryReference>(reduceOp.getRef().getColumn().type);
      auto members = referenceType.getMembers();
      auto ref = mapping.resolve(reduceOp, reduceOp.getRef());
      
      PGX_DEBUG("ReduceOpLowering: Processing generic state reference reduction");
      
      // Set up storage helper for managing aggregate state layout
      EntryStorageHelper storageHelper(reduceOp, members, referenceType.hasLock(), typeConverter);
      
      if (reduceOp->hasAttr("atomic")) {
         PGX_DEBUG("ReduceOpLowering: Using atomic reduction path");
         
         // Atomic path: Get direct pointer to the member and implement atomic reduction
         // Currently supports single-member atomic reductions
         auto valueRef = storageHelper.getPointer(ref, mlir::cast<mlir::StringAttr>(reduceOp.getMembers()[0]).str(), rewriter, reduceOp->getLoc());
         implementAtomicReduce(reduceOp, rewriter, valueRef, mapping);
      } else {
         PGX_DEBUG("ReduceOpLowering: Using non-atomic reduction path");
         
         // Non-atomic path: Load values, execute function, store results
         auto values = storageHelper.getValueMap(ref, rewriter, reduceOp->getLoc());
         std::vector<mlir::Value> arguments;
         
         // Prepare input column arguments
         for (auto attr : reduceOp.getColumns()) {
            mlir::Value arg = mapping.resolve(reduceOp, mlir::cast<tuples::ColumnRefAttr>(attr));
            if (arg.getType() != mlir::cast<tuples::ColumnRefAttr>(attr).getColumn().type) {
               arg = rewriter.create<mlir::UnrealizedConversionCastOp>(reduceOp->getLoc(), mlir::cast<tuples::ColumnRefAttr>(attr).getColumn().type, arg).getResult(0);
            }
            arguments.push_back(arg);
         }
         
         // Prepare current aggregate member arguments
         for (auto member : reduceOp.getMembers()) {
            mlir::Value arg = values.get(mlir::cast<mlir::StringAttr>(member).str());
            if (arg.getType() != reduceOp.getRegion().getArgument(arguments.size()).getType()) {
               arg = rewriter.create<mlir::UnrealizedConversionCastOp>(reduceOp->getLoc(), reduceOp.getRegion().getArgument(arguments.size()).getType(), arg).getResult(0);
            }
            arguments.push_back(arg);
         }

         // Inline the reduction function and handle results
         rewriter.inlineBlock<tuples::ReturnOpAdaptor>(&reduceOp.getRegion().front(), arguments, [&](tuples::ReturnOpAdaptor adaptor) {
            // Update each aggregate member with the computed result
            for (size_t i = 0; i < reduceOp.getMembers().size(); i++) {
               const std::string memberName = mlir::cast<mlir::StringAttr>(reduceOp.getMembers()[i]).str();
               auto& memberVal = values.get(memberName);
               auto updatedVal = adaptor.getResults()[i];
               if (updatedVal.getType() != memberVal.getType()) {
                  updatedVal = rewriter.create<mlir::UnrealizedConversionCastOp>(reduceOp->getLoc(), memberVal.getType(), updatedVal).getResult(0);
               }
               memberVal = updatedVal;
            }
            
            // Persist the updated aggregate state
            values.store();
            rewriter.eraseOp(reduceOp);
         });
      }

      return success();
   }
};

} // namespace subop_to_cf
} // namespace dialect
} // namespace compiler
} // namespace pgx_lower