"""idun's runtime dependencies are NumPy and numba.

pandas and Arrow are accepted as inputs and SciPy and scikit-learn as companions, but importing every module of the
package loads none of them beyond what NumPy and numba load themselves (numba imports SciPy when it is installed).
"""

import subprocess
import sys

OPTIONAL = ("pandas", "pyarrow", "scipy", "sklearn")


def test_importing_every_idun_module_loads_no_optional_package() -> None:
    code = (
        "import importlib, pkgutil, sys, numba, numpy\n"
        "before = {name.split('.')[0] for name in sys.modules}\n"
        "import idun\n"
        "for module in pkgutil.walk_packages(idun.__path__, 'idun.'):\n"
        "    importlib.import_module(module.name)\n"
        f"print(sorted(({{name.split('.')[0] for name in sys.modules}} - before) & {set(OPTIONAL)!r}))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
