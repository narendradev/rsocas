"""Meta-Harness Bridge — bridge between Meta-Harness and Lambda-RLM combinators.

Meta-Harness proposes candidate combinators as Python code.  This bridge
validates them against Lambda-RLM's type system and promotes valid
candidates into the combinator registry.
"""

from __future__ import annotations

import ast
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from rsocas.contracts.combinators import ValidationSnapshot


@dataclass(frozen=True)
class CombinatorCandidate:
    """A candidate combinator discovered by Meta-Harness.

    Wraps a Python callable with metadata for validation.
    """

    name: str
    code: str
    type_signature: str = ""
    hypothesis: str = ""

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return {
            "name": self.name,
            "code": self.code,
            "type_signature": self.type_signature,
            "hypothesis": self.hypothesis,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CombinatorCandidate:
        """Deserialize from a plain dictionary."""
        return cls(
            name=data["name"],
            code=data["code"],
            type_signature=data.get("type_signature", ""),
            hypothesis=data.get("hypothesis", ""),
        )


class MetaHarnessBridge:
    """Bridge between Meta-Harness and RSOCAS combinator system.

    Meta-Harness proposes candidate combinators as Python code.
    This bridge validates them against Lambda-RLM's type system
    and promotes valid candidates into the combinator registry.
    """

    def __init__(
        self,
        candidates_dir: str = "./candidates",
        archive_dir: str = "./archive",
    ) -> None:
        self._candidates_dir = Path(candidates_dir)
        self._archive_dir = Path(archive_dir)
        self._candidates_dir.mkdir(parents=True, exist_ok=True)
        self._archive_dir.mkdir(parents=True, exist_ok=True)

    def validate_candidate(
        self, candidate: CombinatorCandidate
    ) -> tuple[bool, str]:
        """Validate a candidate combinator.

        Checks:
        1. Syntax: code compiles without error.
        2. Import: code can be exec'd without side effects.
        3. Callable: the resulting object is callable.
        4. Bounded: no unbounded loops (simple AST check).

        Returns:
            (valid, message) tuple.
        """
        # 1. Syntax check
        try:
            tree = ast.parse(candidate.code)
        except SyntaxError as exc:
            return False, f"Syntax error: {exc}"

        # 4. Bounded check (before exec to avoid running bad code)
        unbounded = _check_unbounded(tree)
        if unbounded:
            return False, f"Unbounded construct detected: {unbounded}"

        # 2. Import/exec check
        namespace: dict = {}
        try:
            compiled = compile(tree, f"<candidate:{candidate.name}>", "exec")
            exec(compiled, namespace)  # noqa: S102
        except Exception as exc:
            return False, f"Execution error: {exc}"

        # 3. Callable check — look for a function with the candidate's name
        #    or any callable in the namespace
        candidate_fn = namespace.get(candidate.name)
        if candidate_fn is not None:
            if not callable(candidate_fn):
                return False, f"'{candidate.name}' exists but is not callable."
            return True, "ok"

        # Check if any callable was defined
        user_callables = [
            name
            for name, obj in namespace.items()
            if callable(obj) and not name.startswith("_")
        ]
        if user_callables:
            return True, "ok"

        return False, "No callable function found in candidate code."

    def write_candidate(self, candidate: CombinatorCandidate) -> Path:
        """Write a candidate to the candidates directory as a .py file.

        Returns:
            The path to the written file.
        """
        safe_name = candidate.name.replace(" ", "_").replace("/", "_")
        file_path = self._candidates_dir / f"{safe_name}.py"
        header = (
            f'"""Candidate combinator: {candidate.name}\n'
            f"\n"
            f"Type signature: {candidate.type_signature}\n"
            f"Hypothesis: {candidate.hypothesis}\n"
            f'"""\n\n'
        )
        file_path.write_text(header + candidate.code, encoding="utf-8")
        return file_path

    def load_candidates(self) -> list[CombinatorCandidate]:
        """Load all candidates from the candidates directory.

        Reads .py files and extracts metadata from their docstrings.
        """
        candidates: list[CombinatorCandidate] = []
        for py_file in sorted(self._candidates_dir.glob("*.py")):
            content = py_file.read_text(encoding="utf-8")
            name = py_file.stem

            # Try to extract metadata from docstring
            type_sig = ""
            hypothesis = ""
            try:
                tree = ast.parse(content)
                docstring = ast.get_docstring(tree)
                if docstring:
                    for line in docstring.splitlines():
                        stripped = line.strip()
                        if stripped.startswith("Type signature:"):
                            type_sig = stripped[len("Type signature:"):].strip()
                        elif stripped.startswith("Hypothesis:"):
                            hypothesis = stripped[len("Hypothesis:"):].strip()
            except SyntaxError:
                pass

            # Strip the header docstring to get just the code
            code = _strip_module_docstring(content)

            candidates.append(
                CombinatorCandidate(
                    name=name,
                    code=code,
                    type_signature=type_sig,
                    hypothesis=hypothesis,
                )
            )
        return candidates

    def archive_candidate(
        self,
        candidate: CombinatorCandidate,
        validation: ValidationSnapshot,
        accepted: bool,
    ) -> None:
        """Archive a candidate with its validation results.

        Writes to archive_dir/YYYY-MM-DD-name.json.
        """
        now = datetime.now(tz=timezone.utc)
        date_str = now.strftime("%Y-%m-%d")
        safe_name = candidate.name.replace(" ", "_").replace("/", "_")
        archive_path = self._archive_dir / f"{date_str}-{safe_name}.json"

        record = {
            "candidate": candidate.to_dict(),
            "validation": {
                "task_types": list(validation.task_types),
                "input_size_range": list(validation.input_size_range),
                "n_samples": validation.n_samples,
                "mean_score": validation.mean_score,
                "score_std": validation.score_std,
                "timestamp": validation.timestamp,
            },
            "accepted": accepted,
            "archived_at": now.isoformat(),
        }
        archive_path.write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )

    def generate_skill_md(self, task_description: str) -> str:
        """Generate a SKILL.md file content for Meta-Harness's proposer agent.

        This instructs the proposer to discover new combinators that:
        - Are pure functions (no side effects)
        - Have bounded execution (no infinite loops)
        - Accept standard signatures (chunk -> result, chunks -> result)
        - Include type annotations
        """
        return (
            f"# Combinator Discovery Skill\n"
            f"\n"
            f"## Task\n"
            f"\n"
            f"{task_description}\n"
            f"\n"
            f"## Requirements\n"
            f"\n"
            f"Propose Python combinators that satisfy ALL of the following:\n"
            f"\n"
            f"1. **Pure function**: No side effects, no mutation of inputs, "
            f"no I/O operations.\n"
            f"2. **Bounded execution**: No infinite loops (`while True`), "
            f"no unbounded recursion.\n"
            f"3. **Standard signature**: Accept one of:\n"
            f"   - `(chunk: str) -> str` for single-chunk processing\n"
            f"   - `(chunks: list[str]) -> str` for multi-chunk reduction\n"
            f"   - `(chunks: list[str], k: int) -> list[str]` for filtering\n"
            f"4. **Type annotations**: All parameters and return values "
            f"must have type annotations.\n"
            f"5. **Docstring**: Include a docstring explaining the "
            f"combinator's hypothesis.\n"
            f"\n"
            f"## Output Format\n"
            f"\n"
            f"```python\n"
            f"def combinator_name(chunks: list[str]) -> str:\n"
            f'    """One-line description of what this combinator does.\n'
            f"\n"
            f"    Hypothesis: Why this combination strategy might work "
            f"better.\n"
            f'    """\n'
            f"    # implementation\n"
            f"    ...\n"
            f"```\n"
            f"\n"
            f"## Anti-patterns (REJECT)\n"
            f"\n"
            f"- `while True` or `while condition` without guaranteed termination\n"
            f"- `import os`, `import subprocess`, or any system calls\n"
            f"- Global state mutation\n"
            f"- Network requests\n"
            f"- File I/O\n"
        )


def _check_unbounded(tree: ast.Module) -> str:
    """Check AST for unbounded loops or recursion patterns.

    Returns a description string if found, empty string otherwise.
    """
    for node in ast.walk(tree):
        # Check for while True
        if isinstance(node, ast.While):
            if isinstance(node.test, ast.Constant) and node.test.value is True:
                return "while True loop"
            if isinstance(node.test, ast.NameConstant) and getattr(
                node.test, "value", None
            ) is True:
                return "while True loop"

        # Check for bare recursive calls (function calling itself)
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    if isinstance(child.func, ast.Name) and child.func.id == func_name:
                        return f"recursive call to '{func_name}'"

    return ""


def _strip_module_docstring(source: str) -> str:
    """Strip the module-level docstring from Python source, returning the rest."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    if not tree.body:
        return source

    first = tree.body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
        # Get the end line of the docstring
        end_line = first.end_lineno or 0
        lines = source.splitlines(keepends=True)
        remaining = lines[end_line:]
        result = "".join(remaining).strip()
        return result if result else source

    return source
