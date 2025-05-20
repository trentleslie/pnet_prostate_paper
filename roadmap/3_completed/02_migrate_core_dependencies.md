# Feature: Migrate Core Dependencies to Poetry

**Status:** Completed (with one deferral)

**Goal:** Transition project dependencies from Conda (`environment.yml`) to Poetry (`pyproject.toml`), ensuring Python 3.11 compatibility.

**Tasks Completed:**

1.  **Analyzed `environment.yml`:** Identified all original Python 2.7 dependencies.
2.  **Identified Compatible Versions:** Researched and found Python 3.11 compatible versions for most packages.
3.  **Added to `pyproject.toml`:** Populated `[tool.poetry.dependencies]` section with the identified packages and versions.
    *   Example packages added: `h5py`, `imageio`, `tensorflow`, `pandas`, `plotly`, `scikit-learn`, `matplotlib`, etc.
4.  **Resolved Dependencies:** Ran `poetry lock` to generate `poetry.lock` and `poetry install` to install dependencies.

**Deferred Items:**
-   **`kaleido` Installation:** Encountered issues installing `kaleido` via Poetry due to lack of a compatible pre-compiled wheel for the environment. Installation of `kaleido` is deferred. (See `0_backlog/08_resolve_kaleido_dependency.md`)

**Key Outcomes:**
- Most project dependencies are now managed by Poetry and installed successfully for Python 3.11.
- `poetry.lock` file generated, ensuring reproducible builds.
- Core functionality related to these dependencies can be tested once code refactoring progresses.
