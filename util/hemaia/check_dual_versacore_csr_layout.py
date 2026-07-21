#!/usr/bin/env python3
# Copyright 2025 KU Leuven
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Check the hand-maintained Dual VersaCore CSR layout against upstream.

Both headers express addresses relative to STREAMER_WRITER1_BUSY_CSR. This
checker evaluates the selected object-like macros with that external base set
to zero, then compares their relative values. Expression spelling and integer
suffix differences therefore do not count as drift.

Usage:
  check_dual_versacore_csr_layout.py <local_h> <upstream_h>

Exit 0 on match, 1 on layout drift, and 2 on usage, IO, or parse errors.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path


LAYOUT_MACROS = (
    "DUAL_VC_CSR_ADDR_BASE",
    "DUAL_VC_OVERWRITE_ACCUM",
    "DUAL_VC_ACCUM_BOUND",
    "DUAL_VC_OUTPUT_BOUND",
    "DUAL_VC_SUBTRACTIONS",
    "DUAL_VC_ARRAY_SHAPE_CFG",
    "DUAL_VC_DATA_TYPE_CFG",
    "DUAL_VC_MODE",
    "DUAL_VC_RESCALE0_INPUT_ZP",
    "DUAL_VC_RESCALE0_MULTIPLIER",
    "DUAL_VC_RESCALE0_OUTPUT_ZP",
    "DUAL_VC_RESCALE0_SHIFT",
    "DUAL_VC_RESCALE1_INPUT_ZP",
    "DUAL_VC_RESCALE1_MULTIPLIER",
    "DUAL_VC_RESCALE1_OUTPUT_ZP",
    "DUAL_VC_RESCALE1_SHIFT",
    "DUAL_VC_RESCALE_MUL_INPUT_ZP",
    "DUAL_VC_RESCALE_MUL_MULTIPLIER",
    "DUAL_VC_RESCALE_MUL_OUTPUT_ZP",
    "DUAL_VC_RESCALE_MUL_SHIFT",
    "DUAL_VC_NUM_RW_CSR",
    "DUAL_VC_START",
    "DUAL_VC_BUSY",
    "DUAL_VC_PERFORMANCE_COUNTER",
)

EXTERNAL_VALUES = {"STREAMER_WRITER1_BUSY_CSR": 0}
_DEFINE_RE = re.compile(r"^\s*#\s*define\s+([A-Za-z_]\w*)(.*)$")
_C_INTEGER_RE = re.compile(
    r"\b(0[xX][0-9a-fA-F]+|0[bB][01]+|0[0-7]+|[1-9][0-9]*|0)(?:[uUlL]+)\b"
)


class LayoutError(RuntimeError):
    pass


def strip_comments(source: str) -> str:
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"//.*?$", "", source, flags=re.MULTILINE)


def parse_object_defines(source: str) -> dict[str, str]:
    source = strip_comments(source.replace("\\\n", ""))
    defines: dict[str, str] = {}
    for line in source.splitlines():
        match = _DEFINE_RE.match(line)
        if not match:
            continue
        name, tail = match.groups()
        if tail.startswith("("):
            continue  # Function-like macro: NAME(...), with no intervening space.
        expression = tail.strip()
        if expression:
            defines[name] = expression
    return defines


def evaluate_macro(
    name: str,
    defines: dict[str, str],
    cache: dict[str, int],
    stack: tuple[str, ...] = (),
) -> int:
    if name in cache:
        return cache[name]
    if name in stack:
        chain = " -> ".join((*stack, name))
        raise LayoutError(f"cyclic macro definition: {chain}")
    if name not in defines:
        raise LayoutError(f"missing macro used by layout expression: {name}")

    expression = _C_INTEGER_RE.sub(r"\1", defines[name])
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise LayoutError(f"cannot parse {name}={defines[name]!r}: {error.msg}") from error

    def evaluate(node: ast.AST) -> int:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Name):
            return evaluate_macro(node.id, defines, cache, (*stack, name))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return evaluate(node.left) + evaluate(node.right)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
            return evaluate(node.left) - evaluate(node.right)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
            return evaluate(node.operand)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -evaluate(node.operand)
        raise LayoutError(
            f"unsupported expression in {name}={defines[name]!r}: "
            f"{ast.dump(node, include_attributes=False)}"
        )

    value = evaluate(tree)
    cache[name] = value
    return value


def evaluate_layout(path: Path) -> dict[str, int]:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as error:
        raise LayoutError(f"cannot read {path}: {error}") from error

    defines = parse_object_defines(source)
    missing = [name for name in LAYOUT_MACROS if name not in defines]
    if missing:
        raise LayoutError(f"{path} is missing: {', '.join(missing)}")

    cache = dict(EXTERNAL_VALUES)
    return {name: evaluate_macro(name, defines, cache) for name in LAYOUT_MACROS}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("local_h", type=Path)
    parser.add_argument("upstream_h", type=Path)
    args = parser.parse_args()

    try:
        local = evaluate_layout(args.local_h)
        upstream = evaluate_layout(args.upstream_h)
    except LayoutError as error:
        print(f"[dual-versacore-csr-sync] ERROR: {error}", file=sys.stderr)
        return 2

    drift = [name for name in LAYOUT_MACROS if local[name] != upstream[name]]
    if drift:
        print(
            "[dual-versacore-csr-sync] ERROR: CSR layout drift relative to "
            "STREAMER_WRITER1_BUSY_CSR:",
            file=sys.stderr,
        )
        for name in drift:
            print(
                f"  {name}: local={local[name]}, upstream={upstream[name]}",
                file=sys.stderr,
            )
        return 1

    print(
        f"[dual-versacore-csr-sync] OK: {len(LAYOUT_MACROS)} macros match "
        f"{args.upstream_h}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
