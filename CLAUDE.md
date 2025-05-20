# Biomapper Project - Instructions for AI Assistants

## Project Structure and Workflow

### Root Directory Structure

The Biomapper project has the following key directories at the root level:

- `/analysis/`: Code for data analysis, visualizations, and figure generation
- `/data/`: Data access functions, pathway information, and dataset preparation
- `/deepexplain/`: Deep learning explanation tools
- `/docs/`: Project documentation for reference
- `/model/`: Core model implementations, layers, and utilities
- `/pipeline/`: Training and evaluation pipelines
- `/preprocessing/`: Data preprocessing functionality
- `/roadmap/`: Staged development workflow tracking
- `/review/`: Analysis and review work done during paper revision
- `/train/`: Training scripts and parameters
- `/utils/`: Utility functions and helper code

### Code Organization

The model-related code follows these organizational principles:

- `/model/`: Contains all model implementation code
  - `/model/builders/`: Functions that build specific model architectures
  - `/model/refactored_code/`: Reference implementations for TensorFlow 2.x migration
  - `/model/testing/`: Test scripts for validating model functionality
  - Core files like `nn.py`, `model_factory.py`, `coef_weights_utils.py` at the model root

### Documentation Organization

Project documentation follows these conventions:

- `/docs/`: Main documentation directory
  - `/docs/tensorflow_migration/`: Documentation related to TensorFlow 1.x to 2.x migration
  - Each documentation subdirectory contains topic-specific markdown files
  - `README.md` files provide overview and navigation guidance
  
### Testing Organization

Test code follows these conventions:

- `/model/testing/`: Contains test scripts for model components
  - Test files are named according to the component they test (e.g., `test_gradient_layer.py`)
  - Each test directory contains a `README.md` with instructions for running tests

### Roadmap Structure

The Biomapper project uses a staged development workflow to track features from conception to completion. Details on how to use this system are in [`/procedure/pnet_prostate_paper/roadmap/HOW_TO_UPDATE_ROADMAP_STAGES.md`](./roadmap/HOW_TO_UPDATE_ROADMAP_STAGES.md).

Key directories to understand:

- `/roadmap/0_backlog/`: Raw feature ideas and requests not yet planned
- `/roadmap/1_planning/`: Features actively being planned with specs and designs
- `/roadmap/2_inprogress/`: Features under active implementation
- `/roadmap/3_completed/`: Implemented and verified features
- `/roadmap/4_archived/`: Obsolete or deferred features
- `/roadmap/_reference/`: Foundational documents and architectural notes
- `/roadmap/_templates/`: Templates for various feature documents
- `/roadmap/_status_updates/`: Chronological project status updates
- `/roadmap/technical_notes/`: In-depth technical explorations

### Determining Project Status

Follow these steps to understand the current project status and priorities:

1. **Check Both Status Files**:
   - Read `/roadmap/_status_updates/_status_onboarding.md` to understand the format and context of status updates
   - Find and read the most recent file in `/roadmap/_status_updates/` by sorting files by date
   - Cross-reference these to get both the structure of status reporting and the latest content

2. **Review Stage Directories**: Examine the contents of stage directories to understand:
   - What's in the backlog → `/roadmap/0_backlog/`
   - What's being planned → `/roadmap/1_planning/`
   - What's under development → `/roadmap/2_inprogress/`
   - What was recently completed → `/roadmap/3_completed/`

3. **Consult Key Documents**: The following documents are particularly important:
   - `/roadmap/technical_notes/core_mapping_logic/iterative_mapping_strategy.md`: The central guide for mapping processes
   - `/roadmap/HOW_TO_UPDATE_ROADMAP_STAGES.md`: Instructions for maintaining the roadmap
   - Various README files within feature directories

## Working with the Roadmap

### Using Stage Gate Prompts

When instructed to process a feature through a stage gate, follow these steps:

1. Locate the appropriate stage gate prompt file. For example: `/roadmap/1_planning/STAGE_GATE_PROMPT_PLAN.md`
2. Read both the stage gate prompt and the source feature file
3. Execute the instructions in the prompt, creating appropriate folders and files
4. Ensure all required documentation is generated according to templates

### Creating or Updating Backlog Items

When creating new backlog items or updating existing ones:

1. Use the format established in existing backlog items
2. Ensure each item has clear sections for:
   - Overview
   - Problem Statement
   - Key Requirements
   - Success Criteria
   - Any other relevant sections

## Self-Correcting Mechanism

This mechanism applies to any AI assistant working on the project. When a user indicates deviations from the intended development process or expectations, follow this process:

1. **Recognize the deviation**: Acknowledge the gap between expectations and reality

2. **Document the deviation**: Create or update a CLAUDE.md file at the appropriate level:
   - Project level for project-wide deviations
   - Roadmap level for roadmap workflow deviations
   - Stage level for stage-specific process deviations
   - Feature level for feature-specific deviations

3. **Specify the correction**: Include clear instructions on what the correct process should be

4. **Suggest recovery steps**: Outline steps to get back on track

5. **Adapt to the new context**: Adjust future interactions to prevent similar deviations

### Deviation Detection Triggers

Look for these trigger phrases that indicate expectations are not being met:

- "That's not how we do it"
- "That's not the right process"
- "We don't use that approach"
- "You're not following our workflow"
- "That's not our convention"
- "That's not what I expected"
- "This isn't working how I wanted"
- "We need to change how this works"

### Example CLAUDE.md Update Format

When a deviation is detected for a feature in `roadmap/1_planning/feature_x/`, create or update `roadmap/1_planning/feature_x/CLAUDE.md` with content like:

```markdown
# Feature X - CLAUDE.md

## Workflow Deviation Notes

On [DATE], the following deviation was noted:
- [Description of the deviation]
- [Correct approach]

### Recovery Steps
1. [Step 1]
2. [Step 2]
3. [Step 3]

### Future Reference
When working on this feature, always:
- [Guideline 1]
- [Guideline 2]
- [Guideline 3]
```

## Key Technical Documents and Migrations

### TensorFlow Migration

The project is currently migrating from TensorFlow 1.x to TensorFlow 2.x. Key resources for this migration:

- **Documentation**: `/docs/tensorflow_migration/` contains overview, patterns, approach, and component refactoring plans
- **Reference Implementations**: `/model/refactored_code/` contains refactored code examples for TF2.x
- **Tests**: `/model/testing/` contains test scripts for validating refactored implementations
- **Migration Roadmap**: `/roadmap/2_inprogress/05_tensorflow_migration.md` outlines the migration plan and status

When working on TensorFlow-related code:
1. Reference the migration documentation for conversion patterns
2. Follow the established TF2.x patterns in new code
3. Use the reference implementations as guides for refactoring existing code
4. Ensure all changes are covered by tests

### Other Key Technical Documents

- `/docs/tensorflow_migration/migration_overview.md`: High-level overview of the TensorFlow migration
- `/docs/tensorflow_migration/tf2_patterns.md`: Specific patterns for TF1.x to TF2.x conversion
- `/model/testing/README.md`: Information about testing strategies and existing tests

## Project Priorities - How to Determine

To determine current project priorities:

1. **Find the most recent status update** in `/roadmap/_status_updates/` by sorting files by date
2. **Review the 'Next Steps' and 'Priorities' sections** in that document
3. **Cross-reference with stage folders**:
   - High-priority items may already have entries in `/roadmap/1_planning/` or `/roadmap/2_inprogress/`
   - New priorities might still be in `/roadmap/0_backlog/` awaiting planning

This dynamic approach ensures you're always working with the most current priorities rather than relying on static lists that may become outdated.

## Working with Refactored Code

When working with refactored code in the migration process:

1. **Reference Implementation**: The `/model/refactored_code/` directory contains reference implementations that demonstrate the proper patterns for TF2.x compatibility.

2. **Integration Process**: 
   - First validate the refactored code with tests
   - Then integrate the code into the main codebase in the appropriate locations
   - Keep documentation updated to reflect changes

3. **Testing Requirements**:
   - All refactored code must have associated tests
   - Tests should verify functionality before and after refactoring
   - Pay special attention to numerical equivalence when refactoring mathematical operations