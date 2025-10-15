"""
Utility functions for quantum optimization projects.

This module contains shared utility functions used across different modules
in the quantum optimization projects. It provides functions for parsing
complex number expressions and mathematical evaluation.
"""

import re
import ast
import math
from typing import Tuple


def _eval_expression(expr: str) -> float:
    """Safely evaluate mathematical expressions."""
    expr = expr.replace("^", "**")
    allowed_names = {"pi": math.pi, "e": math.e}
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Num,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Call,
        ast.Attribute,
    )
    local_funcs = {"cos": math.cos, "sin": math.sin, "tan": math.tan}
    node = ast.parse(expr, mode="eval")
    for sub in ast.walk(node):
        if not isinstance(sub, allowed_nodes):
            raise ValueError(f"Disallowed expression: {expr}")
        if isinstance(sub, ast.Call):
            if isinstance(sub.func, ast.Name):
                if sub.func.id not in local_funcs:
                    raise ValueError(f"Function '{sub.func.id}' not allowed in '{expr}'")
            elif isinstance(sub.func, ast.Attribute):
                if not (
                    isinstance(sub.func.value, ast.Name)
                    and sub.func.value.id == "math"
                    and sub.func.attr in ("cos", "sin", "tan")
                ):
                    raise ValueError(f"Function call not allowed in '{expr}'")
        if (
            isinstance(sub, ast.Name)
            and sub.id not in allowed_names
            and sub.id not in local_funcs
            and sub.id != "math"
        ):
            raise ValueError(f"Name '{sub.id}' not allowed in '{expr}'")

    return float(
        eval(
            compile(node, filename="<ast>", mode="eval"),
            {"__builtins__": None, **local_funcs, "math": math},
            allowed_names,
        )
    )


def _parse_complex_token(token: str) -> complex:
    """Parse a complex number from a string token."""
    s = token.strip()
    if not s:
        raise ValueError("Empty complex token")
    s_nospace = s.replace(" ", "")
    polar_match = re.fullmatch(
        r"(?:(?P<r>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\*)?e\^\(i(?P<theta>[^\)]+)\)",
        s_nospace,
    )
    if polar_match:
        r_group = polar_match.group("r")
        theta_group = polar_match.group("theta")
        r_val = float(r_group) if r_group is not None else 1.0
        theta_val = _eval_expression(theta_group)
        return complex(r_val * math.cos(theta_val), r_val * math.sin(theta_val))

    s_norm = s.replace("I", "i")
    s_norm = s_norm.replace("i", "j")
    s_norm = re.sub(r"(^|[^\w\.])\+j", r"\g<1>+1j", s_norm)
    s_norm = re.sub(r"(^|[^\w\.])\-j", r"\g<1>-1j", s_norm)
    s_norm = re.sub(r"(^|[^\w\.])j", r"\g<1>1j", s_norm)
    s_norm = s_norm.replace(" ", "")

    try:
        return complex(s_norm)
    except Exception as e:
        raise ValueError(f"Cannot parse complex number from '{token}': {e}")


def parse_target_superposition(target_superposition: list) -> Tuple:
    """
    Parse target superposition tokens into complex coefficients.
    
    Parameters
    ----------
    target_superposition : list
        List of string tokens representing complex numbers
        
    Returns
    -------
    Tuple
        Tuple of parsed complex coefficients
    """
    raw_tokens = []
    for t in target_superposition:
        parts = [p.strip() for p in t.split(",")]
        raw_tokens.extend([p for p in parts if p != ""])

    parsed_coeffs = tuple(_parse_complex_token(tok) for tok in raw_tokens)

    if len(parsed_coeffs) >= 2:
        parsed_coeffs = (float(parsed_coeffs[0].real),) + parsed_coeffs[1:]
    
    return parsed_coeffs
