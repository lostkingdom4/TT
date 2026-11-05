#!/usr/bin/env python3
import argparse
import ast
import fnmatch
import io
import os
import sys
from typing import List, Optional, Tuple

# Python 3.9+ for ast.unparse
if not hasattr(ast, "unparse"):
    print("Python 3.9+ is required.", file=sys.stderr)
    sys.exit(1)


def is_four_const(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, int) and node.value == 4


class Inside:
    """Track whether we are inside get_inputs or get_init_inputs."""
    def __init__(self):
        self.in_get_inputs = False
        self.in_get_init = False


class FourBumper(ast.NodeTransformer):
    """
    Rules (only inside get_inputs/get_init_inputs):
      - Lists/Tuples of length 4 with all elements==4 -> replace with [B,C,H,W] / (B,C,H,W)
      - Calls with 4 positional args all == 4 (e.g., torch.rand(4,4,4,4)) -> map to (B,C,H,W)
      - Dict in get_init_inputs: 'inplane':4 -> C; 'plane':4 -> PLANE
      - Any other literal 4 -> scalar replacement (default = C)
    """
    def __init__(self, B: int, C: int, H: int, W: int, plane: Optional[int], scalar4: Optional[int]):
        super().__init__()
        self.B, self.C, self.H, self.W = B, C, H, W
        self.P = C if plane is None else plane
        self.scalar4 = C if scalar4 is None else scalar4
        self.ctx = Inside()
        self.changed = False

    def _BCHW_elts(self, node_type):
        return node_type(elts=[ast.Constant(self.B), ast.Constant(self.C),
                               ast.Constant(self.H), ast.Constant(self.W)],
                         ctx=ast.Load())

    def _map_args_BCHW(self, args: List[ast.AST]) -> Optional[List[ast.AST]]:
        if len(args) == 4 and all(is_four_const(a) for a in args):
            self.changed = True
            return [ast.Constant(self.B), ast.Constant(self.C),
                    ast.Constant(self.H), ast.Constant(self.W)]
        return None

    def _maybe_shape_list_tuple(self, node: ast.AST) -> ast.AST:
        if isinstance(node, (ast.List, ast.Tuple)) and len(node.elts) == 4 and all(is_four_const(e) for e in node.elts):
            self.changed = True
            return self._BCHW_elts(type(node))
        return node

    # ---- Function boundaries ----
    def visit_FunctionDef(self, node: ast.FunctionDef):
        # enter
        was_inputs = self.ctx.in_get_inputs
        was_init = self.ctx.in_get_init
        if node.name == "get_inputs":
            self.ctx.in_get_inputs = True
        elif node.name == "get_init_inputs":
            self.ctx.in_get_init = True

        node = self.generic_visit(node)

        # exit
        self.ctx.in_get_inputs = was_inputs
        self.ctx.in_get_init = was_init
        return node

    # ---- Calls (torch.rand(...), etc.) ----
    def visit_Call(self, node: ast.Call):
        node = self.generic_visit(node)
        if not (self.ctx.in_get_inputs or self.ctx.in_get_init):
            return node

        # If single list/tuple arg that is [4,4,4,4] -> map
        if node.args:
            first = node.args[0]
            replaced = self._maybe_shape_list_tuple(first)
            if replaced is not first:
                node.args[0] = replaced
                return node

            # Or 4 positional args all 4 -> map
            new_args = self._map_args_BCHW(node.args)
            if new_args is not None:
                node.args = new_args
                return node

        return node

    # ---- Dict (handle inplane/plane in get_init_inputs) ----
    def visit_Dict(self, node: ast.Dict):
        node = self.generic_visit(node)
        if not self.ctx.in_get_init:
            return node
        for i, k in enumerate(node.keys):
            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                if k.value == "inplane" and is_four_const(node.values[i]):
                    node.values[i] = ast.Constant(self.C)
                    self.changed = True
                elif k.value == "plane" and is_four_const(node.values[i]):
                    node.values[i] = ast.Constant(self.P)
                    self.changed = True
        return node

    # ---- Lists/Tuples (free-standing shapes like [4,4,4,4]) ----
    def visit_List(self, node: ast.List):
        node = self.generic_visit(node)
        if not (self.ctx.in_get_inputs or self.ctx.in_get_init):
            return node
        return self._maybe_shape_list_tuple(node)

    def visit_Tuple(self, node: ast.Tuple):
        node = self.generic_visit(node)
        if not (self.ctx.in_get_inputs or self.ctx.in_get_init):
            return node
        return self._maybe_shape_list_tuple(node)

    # ---- Literal 4 anywhere else inside the two functions -> scalar replacement ----
    def visit_Constant(self, node: ast.Constant):
        if not (self.ctx.in_get_inputs or self.ctx.in_get_init):
            return node
        if is_four_const(node):
            # Let List/Tuple/Call handlers take priority for shape-aware mapping.
            # If we reached here, treat as scalar replacement.
            self.changed = True
            return ast.Constant(self.scalar4)
        return node


def process_file(path: str, B: int, C: int, H: int, W: int, plane: Optional[int],
                 scalar4: Optional[int], dry: bool) -> Tuple[bool, str]:
    try:
        src = io.open(path, "r", encoding="utf-8").read()
    except Exception as e:
        return False, f"[ERR]  {path}: {e}"

    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return False, f"[SKIP] {path}: syntax error ({e})"

    rewriter = FourBumper(B, C, H, W, plane, scalar4)
    new_tree = rewriter.visit(tree)
    ast.fix_missing_locations(new_tree)

    if not rewriter.changed:
        return False, f"[SKIP] {path}: no changes inside target functions"

    new_src = ast.unparse(new_tree)
    if dry:
        return True, f"[DRY]  {path}: would update"

    try:
        io.open(path + ".bak", "w", encoding="utf-8").write(src)
        io.open(path, "w", encoding="utf-8").write(new_src)
        return True, f"[OK]   {path}: updated (backup at {path}.bak)"
    except Exception as e:
        return False, f"[ERR]  {path}: {e}"


def should_process(filename: str, include: List[str], exclude: List[str]) -> bool:
    base = os.path.basename(filename)
    if include and not any(fnmatch.fnmatch(base, pat) for pat in include):
        return False
    if exclude and any(fnmatch.fnmatch(base, pat) for pat in exclude):
        return False
    return True


def main():
    ap = argparse.ArgumentParser(
        description="Replace literal 4s INSIDE get_inputs/get_init_inputs with larger values. "
                    "Shape-aware for 4D tensors; scalar replacement elsewhere."
    )
    ap.add_argument("root", help="File or directory")
    ap.add_argument("-B", "--batch", type=int, default=8, help="B in BCHW (default 8)")
    ap.add_argument("-C", "--channels", type=int, default=64, help="C in BCHW and default scalar replacement (default 64)")
    ap.add_argument("-H", "--height", type=int, default=56, help="H in BCHW (default 56)")
    ap.add_argument("-W", "--width", type=int, default=56, help="W in BCHW (default 56)")
    ap.add_argument("--plane", type=int, default=None, help="plane in get_init_inputs (default: channels)")
    ap.add_argument("--scalar4", type=int, default=None,
                    help="Scalar fallback for lone literal 4s (default: channels)")
    ap.add_argument("--include", nargs="*", default=["*.py"], help="Only process files matching these globs")
    ap.add_argument("--exclude", nargs="*", default=["*test*.py", "*__init__*.py"], help="Skip files matching these globs")
    ap.add_argument("--dry-run", action="store_true", help="Show which files would change, don’t write")
    args = ap.parse_args()

    # Single file
    if os.path.isfile(args.root):
        ok, msg = process_file(args.root, args.batch, args.channels, args.height, args.width,
                               args.plane, args.scalar4, args.dry_run)
        print(msg)
        sys.exit(0 if ok else 2)

    # Directory walk
    if not os.path.isdir(args.root):
        print(f"Not found: {args.root}", file=sys.stderr)
        sys.exit(2)

    any_ok = False
    for dirpath, _, files in os.walk(args.root):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            if not should_process(path, args.include, args.exclude):
                continue
            ok, msg = process_file(path, args.batch, args.channels, args.height, args.width,
                                   args.plane, args.scalar4, args.dry_run)
            any_ok = any_ok or ok
            print(msg)
    sys.exit(0 if any_ok else 2)


if __name__ == "__main__":
    main()