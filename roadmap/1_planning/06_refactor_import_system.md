# Feature: Refactor Import System and Remove PYTHONPATH Reliance

**Status:** Planning

**Goal:** Modernize the project's import system to work seamlessly with Poetry without requiring `PYTHONPATH` manipulation, improving maintainability and deployment.

**Background:**
The project currently uses `sys.path.insert(0, dirname(current_dir))` in scripts like `train/run_me.py` to allow imports from the project root directory (e.g., `import utils`, `import pipeline`). This is a common workaround in older projects but is not ideal with package managers like Poetry.

**Tasks:**

1.  **Analyze Current Import Patterns:**
    *   Identify all instances of `sys.path` manipulation.
    *   Map out current inter-module dependencies and import statements.
2.  **Design New Package Structure (if needed):**
    *   Consider adopting a `src/` layout where all main project code (`pipeline`, `model`, `utils`, `params`, etc.) resides within a `src/pnet_prostate_paper/` directory (or similar package name).
    *   Alternatively, ensure the current top-level directories (`pipeline`, `model`, etc.) are treated as part of a single implicit namespace package if a `src/` layout is too disruptive.
3.  **Update Import Statements:**
    *   Change all import statements to be relative (e.g., `from . import utils`) or absolute from the defined package name (e.g., `from pnet_prostate_paper.utils import logs`).
4.  **Configure `pyproject.toml` (if using `src/` layout):**
    *   Ensure `pyproject.toml` correctly points to the package directory if a `src/` layout is adopted.
    ```toml
    # Example for src layout
    [tool.poetry]
    name = "pnet_prostate_paper"
    # ...
    packages = [{include = "pnet_prostate_paper", from = "src"}]
    ```
5.  **Remove `sys.path` Manipulations:**
    *   Delete all lines that modify `sys.path` (e.g., `sys.path.insert(...)`).
6.  **Test Imports:**
    *   Run scripts and attempt to import modules to ensure the new import system works correctly.
    *   Utilize `poetry install` and run commands like `poetry run python -m train.run_me` (adjusting for actual script paths/module execution) to test.

**Expected Benefits:**
-   Cleaner, more maintainable import structure.
-   Project can be installed as a package by Poetry (`pip install .`).
-   Eliminates the need for users to manually set `PYTHONPATH`.
-   Better compatibility with IDEs and static analysis tools.

**Potential Challenges:**
-   Can be a time-consuming refactor for large codebases with many interdependencies.
-   Risk of breaking imports if not done carefully.
