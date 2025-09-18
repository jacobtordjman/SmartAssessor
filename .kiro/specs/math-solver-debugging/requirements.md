# Requirements Document

## Introduction

The current SmartAssessor math solver is producing incorrect answers due to issues with the two-LLM architecture (WizardMath solver + tiny parser). The system needs to improve accuracy and reliability for determinant calculations.

## Requirements

### Requirement 1

**User Story:** As a developer, I want to debug why the math solver is giving wrong answers, so that I can identify and fix the root causes.

#### Acceptance Criteria

1. WHEN analyzing solver output THEN the system SHALL show the raw WizardMath model output without parser interference
2. WHEN comparing answers THEN the system SHALL use proper floating-point comparison with appropriate tolerances
3. WHEN a solver gives a correct answer THEN the system SHALL properly recognize it as correct


### Requirement 2

**User Story:** As a developer, I want to simplify the parsing logic, so that parsing errors don't interfere with correct solver outputs.

#### Acceptance Criteria

1. WHEN parsing solver output THEN the system SHALL use only regex-based parsing without LLM interference
2. WHEN the parser fails to extract an answer THEN the system SHALL show the full solver output for manual inspection
3. WHEN extracting numeric answers THEN the system SHALL handle various number formats (integers, decimals, scientific notation)
4. WHEN parsing short solutions THEN the system SHALL extract them directly from solver output without modification

### Requirement 3

**User Story:** As a developer, I want to improve the solver prompting strategy, so that WizardMath produces more accurate determinant calculations.

#### Acceptance Criteria

1. WHEN prompting the solver THEN the system SHALL follow HuggingFace WizardMath documentation for optimal prompt structure
2. WHEN providing instructions THEN the system SHALL use general determinant calculation guidance without matrix-size-specific formulas
3. WHEN generating responses THEN the system SHALL enforce consistent output formatting
4. WHEN prompting THEN the system SHALL avoid confusing examples that might mislead the model's reasoning

### Requirement 4

