include "mlir/Dialect/Arith/IR/ArithOps.td"
//include "lingodb/mlir/Dialect/DB/IR/DBOps.td"
include "mlir/IR/PatternBase.td"

def ConstAddCmpFold :
    Pat<(Arith_CmpFOp $pred,
          (Arith_AddFOp $a, (Arith_ConstantOp $c2), $addfastmath),
          (Arith_ConstantOp $c1),
          $fastmath),
        (Arith_CmpFOp $pred, $a, (Arith_SubFOp (Arith_ConstantOp $c1), (Arith_ConstantOp $c2), $addfastmath), $fastmath)>;
def ConstMulCmpFold :
    Pat<(Arith_CmpFOp $pred,
          (Arith_MulFOp $a, (Arith_ConstantOp $c2), $mulfastmath),
          (Arith_ConstantOp $c1),
          $fastmath),
        (Arith_CmpFOp $pred, $a, (Arith_DivFOp (Arith_ConstantOp $c1), (Arith_ConstantOp $c2), $mulfastmath), $fastmath)>;