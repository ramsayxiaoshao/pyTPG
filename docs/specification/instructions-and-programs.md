# Instructions, registers, and programs

## Instruction representation

An instruction contains:

- an `OperatorName`, resolved by an explicitly owned operator registry;
- one destination `RegisterIndex`;
- an ordered tuple of operands.

An operand is exactly one of:

- `InputOperand(InputIndex)`;
- `RegisterOperand(RegisterIndex)`;
- `ConstantOperand(float)`.

This tagged representation avoids an encoded addressing-mode bit and lets an
operator declare its accepted arity and operand kinds. Adding an operator MUST
NOT require modifying the executor's control flow.

Operator lookup, arity checking, and bounds checking occur when a program is
validated against a runtime configuration. Unknown operators and out-of-range
indices MUST be reported as validation errors; they MUST NOT be silently wrapped.

## Program execution contract

A program is a non-empty ordered tuple of instructions. The reference executor
MUST:

1. allocate a fresh, independent, all-zero register file for every execution;
2. execute instructions in tuple order;
3. resolve every operand through its explicit tag;
4. write only the instruction's destination register;
5. apply the numerical-safety policy below; and
6. return the final finite value in register 0 as the raw bid.

Stateful composition does not alter this lifecycle. Persistent episode memory
is appended to the observation as explicit input channels; it does not seed or
reuse program registers. See [Stateful memory](stateful-memory.md).

The observation MUST be a one-dimensional sequence whose length exactly equals
`RuntimeConfig.input_size`. Each value MUST be a finite real number. The executor
MUST reject booleans, non-numeric values, NaN, infinity, and shape mismatches; it
MUST NOT clip, pad, flatten, or otherwise repair an observation.

Register and input indexing is strict and zero-based. Out-of-range indices MUST
fail validation and MUST NOT wrap. `RuntimeConfig.register_count` MUST be at
least one. A zero-length observation is valid for constant-only programs.

The executor MUST NOT use global randomness. Pure deterministic operators SHOULD
be the default instruction set. Stochastic custom operators, if later supported,
must receive an explicit random generator.

## Built-in operators

The default registry contains these exact fixed-arity names:

| Name | Arity | Result |
| --- | ---: | --- |
| `identity` | 1 | `x` |
| `add` | 2 | `x + y` |
| `subtract` | 2 | `x - y` |
| `multiply` | 2 | `x * y` |
| `divide` | 2 | `0.0` if `y == 0.0`, otherwise `x / y` |
| `sin` | 1 | `sin(x)` |
| `cos` | 1 | `cos(x)` |
| `log` | 1 | `0.0` if `x <= 0.0`, otherwise natural `log(x)` |
| `min` | 2 | `min(x, y)` |
| `max` | 2 | `max(x, y)` |

Every call to `default_operator_registry()` MUST return a fresh registry so one
experiment cannot mutate another experiment's instruction set. Custom operators
are registered as `(name, arity, callable)` values without executor changes.

## Numerical safety

Protected division and logarithm use the exact behavior in the table. Any
`ArithmeticError` or `ValueError` raised by an operator produces `0.0`. Any NaN
or infinity returned by an operator is also normalized to `0.0` before the
destination write. A custom operator returning a non-real value violates the
operator contract and raises `OperatorExecutionError` rather than being silently
repaired.

This policy is applied after every instruction, so registers and bids are always
finite. Changing any protected value is an observable semantic change requiring
updated specifications and regression tests.
