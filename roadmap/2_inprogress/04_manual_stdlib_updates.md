# Feature: Manual Python 2 to 3 Standard Library Updates

**Status:** In Progress

**Goal:** Manually refactor Python code to replace or update standard library modules and idioms that have changed between Python 2 and Python 3 and were not fully handled by `2to3`.

**Tasks:**

1.  **Replace `imp` module with `importlib`:** (Partially Done)
    *   Identified usage in `train/run_me.py` for dynamic loading of parameter files.
    *   Refactored `train/run_me.py` to use `importlib.util.spec_from_file_location` and `importlib.util.module_from_spec`.
    *   *Next Steps:* Identify and refactor any other uses of `imp`.
2.  **Convert `xrange` to `range`:**
    *   *Next Steps:* Search codebase for any remaining `xrange` instances and replace with `range`. (`2to3` should have caught most).
3.  **Update dictionary `iteritems()`/`iterkeys()`/`itervalues()` to `items()`/`keys()`/`values()`:**
    *   *Next Steps:* Search codebase and update as needed. (`2to3` should have caught most).
4.  **Ensure correct string/bytes handling:**
    *   *Next Steps:* Be mindful of areas involving file I/O or network communication where string/byte distinctions are critical. This might surface during testing.
5.  **Update division operators (`/` vs `//`):**
    *   *Next Steps:* Review calculations, especially those involving integers, to ensure division behaves as expected (float vs. floor division). (`from __future__ import division` handled by `2to3` might cover many cases).
6.  **Final `print` statement review:**
    *   *Next Steps:* Ensure all `print` statements are `print()` function calls with parentheses. (`2to3` should have caught all).

**Files Currently Being Worked On/Reviewed:**
-   `train/run_me.py` (for `imp` module)

**Blockers/Challenges:**
-   Identifying all instances of deprecated patterns might require thorough code review or runtime testing.
