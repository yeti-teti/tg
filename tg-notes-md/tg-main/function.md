### function.py

***** Imports ***** 
        - tinygrad.helpers
            - argsort
        - tinygrad.dtype
            - dtypes
            - DType
            - sum_acc_dtype
        - tinygrad.ops
            - Ops
            - resolve
            - sint
            - UOp
        - tinygrad.tensor
            - Function

***** Classes ***** 
        - Contiguous(Function)
        - ContiguousBackward(Function)
        - Cast(Function)
        - Reciprocal(Function)
       ----------------------- Unary Ops -----------------------
        - Sin(Function)
        - Relu(Function)
        - Log(Function)
        - Exp(Function)
        - Sqrt(Function)
        - Sign(Function)
       ----------------------- Binary Ops -----------------------
        - Less(Function)
        - Neq(Function)
        - Xor(Function)
        - BitwiseAnd(Function)
        - Threefry(Function)
        - Add(Function)
        - Mul(Function)
        - IDiv(Function)
        ----------------------- Ternary Ops -----------------------
        - Where(Function)
        ----------------------- Reduce Ops -----------------------
        - Sum(Function)
        - Prod(Function)
        - Max(Function)
        - Expand(Function)
        - Reshape(Function)
        - Permute(Function)
        - Pad(Function)
        - Shrink(Function)
        - Flip(Function)