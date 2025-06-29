#pragma once

// MLIR 20.x properties compatibility
namespace mlir { namespace pg {

// Empty properties struct for operations that don't need properties
struct EmptyProperties {
    bool operator==(const EmptyProperties &) const { return true; }
    bool operator!=(const EmptyProperties &) const { return false; }
};

// Mixin to add getProperties() method to generated operations
template<typename Derived>
class OpWithPropertiesMixin {
   public:
    EmptyProperties getProperties() const { return EmptyProperties{}; }
};

}} // namespace mlir::pg