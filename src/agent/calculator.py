"""A safe arithmetic calculator tool.

Expressions are parsed with ``ast`` and evaluated by walking the tree; ``eval``
is never used. Only numeric literals, ``+ - * / // % **`` and unary ``+ -`` are
accepted. Names, calls, attributes, strings, comparisons, and every other
syntax node are rejected. Size limits stop denial-of-service inputs such as
``9**9**9`` or very long literals.
"""

from __future__ import annotations

import ast
import operator

MAX_EXPRESSION_LENGTH = 200
MAX_ABS_VALUE = 10**100
MAX_EXPONENT = 100
MAX_DEPTH = 50

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


class CalculatorError(ValueError):
    """Raised for any expression the calculator refuses or cannot evaluate."""


def _check(value: int | float | complex) -> int | float:
    if isinstance(value, complex):
        raise CalculatorError("result is not a real number")
    if value != value or abs(value) > MAX_ABS_VALUE:
        raise CalculatorError("result is too large or undefined")
    return value


def _evaluate(node: ast.AST, depth: int = 0) -> int | float:
    if depth > MAX_DEPTH:
        raise CalculatorError("expression is nested too deeply")
    if isinstance(node, ast.Expression):
        return _evaluate(node.body, depth + 1)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise CalculatorError("only plain numbers are allowed")
        return _check(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _check(_UNARY[type(node.op)](_evaluate(node.operand, depth + 1)))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left = _evaluate(node.left, depth + 1)
        right = _evaluate(node.right, depth + 1)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise CalculatorError(f"exponent magnitude is limited to {MAX_EXPONENT}")
        try:
            return _check(_BINARY[type(node.op)](left, right))
        except ZeroDivisionError:
            raise CalculatorError("division by zero") from None
        except OverflowError:
            raise CalculatorError("result is too large or undefined") from None
    raise CalculatorError(f"unsupported syntax: {type(node).__name__}")


def evaluate(expression: str) -> int | float:
    """Evaluate a pure arithmetic expression or raise :class:`CalculatorError`."""
    if not isinstance(expression, str):
        raise CalculatorError("expression must be a string")
    expression = expression.strip()
    if not expression:
        raise CalculatorError("expression is empty")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise CalculatorError(f"expression is longer than {MAX_EXPRESSION_LENGTH} characters")
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        raise CalculatorError("expression is not valid arithmetic") from None
    try:
        return _evaluate(tree)
    except RecursionError:
        raise CalculatorError("expression is nested too deeply") from None


def format_number(value: int | float) -> str:
    """Render integers exactly and floats with up to ten significant digits."""
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        return str(int(value))
    return str(value) if isinstance(value, int) else format(value, ".10g")


def find_tool_call(text: str) -> tuple[str, int, int] | None:
    """Locate the first ``CALC(<expression>)`` call with balanced parentheses.

    Returns ``(expression, start, end)`` where ``text[start:end]`` is the whole
    call, or ``None`` if there is no complete call.
    """
    start = text.find("CALC(")
    if start == -1:
        return None
    depth = 0
    for index in range(start + len("CALC"), len(text)):
        character = text[index]
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                return text[start + len("CALC(") : index], start, index + 1
    return None
