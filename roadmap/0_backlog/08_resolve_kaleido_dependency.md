# Feature: Resolve `kaleido` Dependency

**Status:** Backlog (Deferred)

**Goal:** Enable static image export for Plotly figures by successfully installing and integrating the `kaleido` package or an alternative.

**Background:**
The `kaleido` package is used by Plotly to export figures to static formats (PNG, JPG, PDF, SVG). During the initial Poetry setup for Python 3.11, installing `kaleido==0.2.1.post1` failed due to the absence of a pre-compiled wheel for the user's Linux environment and Python 3.11 combination. The error message typically indicates "Failed to build kaleido" or "No matching distribution found for kaleido".

**Tasks:**

1.  **Re-attempt `kaleido` Installation (Post Core Refactoring):**
    *   Once the main Python 3 and TensorFlow refactoring is stable, try `poetry add kaleido` again. Sometimes, other environment changes or package updates can resolve such issues.
    *   Try `poetry add kaleido --version="<specific_older_compatible_version>"` if a known working version for Python 3.11 on Linux can be identified.
2.  **Investigate Pre-compiled Wheel Availability:**
    *   Check the `kaleido` PyPI page for newer versions or different wheels that might support the target environment.
3.  **Explore System-Level Dependencies:**
    *   Some Python packages with compiled components require system libraries. Research if `kaleido` needs specific system packages installed (e.g., via `apt-get`) that might be missing.
4.  **Attempt Installation via `pip` (within Poetry environment):**
    *   If Poetry struggles, try `poetry run pip install kaleido`. This sometimes offers more flexibility or different build options. If successful, the dependency might need to be managed carefully or added to `pyproject.toml` with specific markers.
5.  **Consider Building from Source:**
    *   If pre-compiled wheels are unavailable, investigate if `kaleido` can be built from source in the user's environment. This might involve installing build tools (compilers, etc.).
6.  **Explore Alternative Libraries:**
    *   If `kaleido` remains problematic, research alternative methods for Plotly static image export:
        *   **Orca (deprecated but might still work for some use cases):** This was the predecessor to `kaleido`.
        *   **Plotly Online Export:** If an internet connection is acceptable during figure generation, Plotly's online services can be used (requires API key for higher volume).
        *   Other third-party libraries that interface with Plotly for export.
7.  **Update Code:**
    *   Once a solution is found, update any Python code that uses Plotly's `fig.write_image(...)` to ensure it works with the chosen method.

**Priority:** Medium-Low. Core model functionality and Python 3 migration take precedence. Static image export is important for publication-ready figures but not for initial model training and validation.

**Blockers:**
-   Availability of compatible `kaleido` wheels or ease of building from source.
-   Complexity of integrating alternative export solutions.
