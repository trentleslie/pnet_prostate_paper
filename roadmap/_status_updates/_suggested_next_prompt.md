## Suggested Next Prompt

**1. Context Brief:**
We have successfully completed a comprehensive debugging of the P-NET model builder test suite, with all 13 tests in `/procedure/pnet_prostate_paper/model/testing/test_model_builders.py` now passing. This resolves critical issues related to mock data, parameter handling, and custom layer functionality, particularly the `SparseTFSimple` attention mechanism. The model building components are now considered stable.

**2. Initial Steps:**
1.  **Review Project Context:** Familiarize yourself with the overall project goals and architecture by reviewing `/procedure/pnet_prostate_paper/CLAUDE.md`.
2.  **Review Latest Status Update:** Read the detailed status update at `/procedure/pnet_prostate_paper/roadmap/_status_updates/2025-05-28-pnet-test-suite-debugged.md` for a full understanding of recent accomplishments and the current project state.

**3. Work Priorities:**
The highest priority is to **initiate planning for P-NET Full Training Pipeline Integration Testing.**
   - **Action:**
     1. Create a new backlog item (e.g., a simple `.md` file) in `/procedure/pnet_prostate_paper/roadmap/0_backlog/` describing the goal: "Validate P-NET models (built using debugged functions) in an end-to-end training scenario, including data loading, model forward/backward pass, and metrics."
     2. Instruct your AI assistant (Cascade) to execute `/procedure/pnet_prostate_paper/roadmap/1_planning/STAGE_GATE_PROMPT_PLAN.md` for this new backlog item. This will generate initial planning documents (`README.md`, `spec.md`, `design.md`) in a new subfolder within `/procedure/pnet_prostate_paper/roadmap/1_planning/`.

**4. Key References:**
*   Status Update: `/procedure/pnet_prostate_paper/roadmap/_status_updates/2025-05-28-pnet-test-suite-debugged.md`
*   Completed Debugging Summary: `/procedure/pnet_prostate_paper/roadmap/3_completed/pnet_model_test_debugging/summary.md`
*   Main Test File: `/procedure/pnet_prostate_paper/model/testing/test_model_builders.py`
*   Key Model Building Utility: `/procedure/pnet_prostate_paper/model/builders/builders_utils.py` (contains `get_pnet`)
*   Custom Layers: `/procedure/pnet_prostate_paper/model/layers_custom_tf2.py`
*   Roadmap Planning Prompt: `/procedure/pnet_prostate_paper/roadmap/1_planning/STAGE_GATE_PROMPT_PLAN.md`

**5. Workflow Integration:**
*   **Phase 1 (Planning - Current Priority):** Use Cascade to process the new "Integration Testing" backlog item through the planning stage gate as described above. Review and refine the AI-generated planning documents.
*   **Phase 2 (Implementation - Future):** Once planning is satisfactory, this "Integration Testing" feature can move to `/procedure/pnet_prostate_paper/roadmap/2_inprogress/`. For developing the actual integration test scripts and mock data:
    *   Consider creating a detailed prompt for a Claude code instance to help design and implement the test harness, mock data generation for training, and the training script itself. This prompt should reference the planning documents created in Phase 1.