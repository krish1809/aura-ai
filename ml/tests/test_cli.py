import ast
import importlib.util
from pathlib import Path

from aura_ml.cli import COMMANDS


def test_every_cli_command_has_a_main():
    """Parse each module (no import: training modules need torch) and check it defines main()."""
    for cmd, module in COMMANDS.items():
        spec = importlib.util.find_spec(module)
        assert spec and spec.origin, f"{cmd}: module {module} not found"
        tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
        names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        assert "main" in names, f"`aura-ml {cmd}` -> {module} has no main()"
