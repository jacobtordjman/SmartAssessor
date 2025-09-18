# Design Document

## Overview

This design addresses the issues in the SmartAssessor math solver by simplifying the architecture, improving prompting strategies, and adding comprehensive debugging capabilities. The solution removes the problematic parser LLM and focuses on making the WizardMath solver more reliable.

## Architecture

### Current Architecture Issues
- Two-LLM system (WizardMath + tiny parser) creates confusion
- Parser LLM generates incorrect "short solutions" 
- Complex parsing logic introduces failure points
- Limited debugging visibility

### Proposed Architecture
- Single-LLM system using only WizardMath
- Regex-only parsing for reliability
- Enhanced debugging and logging
- Simplified prompt structure following HuggingFace guidelines

## Components and Interfaces

### 1. Simplified Solver Pipeline
```python
class SimplifiedMathSolver:
    def __init__(self, model_path: str)
    def solve_problem(self, question: str) -> SolverResult
    def _create_prompt(self, question: str) -> str
    def _parse_response(self, raw_output: str) -> ParsedResult
```

### 2. Enhanced Parser
```python
class RegexParser:
    def extract_final_answer(self, text: str) -> Optional[float]
    def extract_solution_steps(self, text: str) -> str
    def _apply_fallback_patterns(self, text: str) -> Dict[str, Any]
```

### 3. Debugging System
```python
class DebugLogger:
    def log_prompt(self, prompt: str)
    def log_raw_output(self, output: str)
    def log_parsing_result(self, result: ParsedResult)
    def log_comparison(self, expected: float, actual: float, correct: bool)
```

### 4. Assessment Engine
```python
class MathAssessment:
    def assess_problems(self, problems: List[Problem]) -> AssessmentResult
    def _compare_answers(self, expected: float, actual: float) -> bool
    def _generate_report(self, results: List[ProblemResult]) -> str
```

## Data Models

### Problem
```python
@dataclass
class Problem:
    question: str
    expected_answer: float
    problem_id: Optional[str] = None
```

### SolverResult
```python
@dataclass
class SolverResult:
    raw_output: str
    prompt_used: str
    generation_time: float
```

### ParsedResult
```python
@dataclass
class ParsedResult:
    final_answer: Optional[float]
    solution_steps: str
    parsing_success: bool
    parsing_method: str  # "primary_regex", "fallback_regex", "failed"
```

### ProblemResult
```python
@dataclass
class ProblemResult:
    problem: Problem
    solver_result: SolverResult
    parsed_result: ParsedResult
    is_correct: bool
    error_message: Optional[str] = None
```

## Error Handling

### Solver Errors
- Model loading failures → Clear error messages with troubleshooting steps
- Generation timeouts → Configurable timeout with retry logic
- CUDA memory issues → Automatic fallback to CPU with warnings

### Parsing Errors
- No numeric answer found → Show full output for manual inspection
- Multiple answers detected → Use first occurrence with warning
- Invalid number format → Log parsing attempt and continue

### Comparison Errors
- Floating point precision issues → Use configurable tolerance (default: 1e-6)
- Scientific notation handling → Normalize before comparison
- Edge cases (infinity, NaN) → Explicit handling with clear messages

## Testing Strategy

### Unit Tests
- Test regex patterns with various output formats
- Test prompt generation with different question types
- Test answer comparison with edge cases
- Test error handling scenarios

### Integration Tests
- End-to-end problem solving with known correct answers
- Performance testing with batch problem sets
- Memory usage monitoring during extended runs
- Cross-validation with manual calculations

### Debugging Tests
- Verify all debug output is captured correctly
- Test logging at different verbosity levels
- Validate debug output format consistency
- Test debug output with various failure modes

## Implementation Notes

### WizardMath Prompting Best Practices
Based on HuggingFace documentation:
- Use clear, direct mathematical language
- Avoid overly complex examples in prompts
- Structure prompts with consistent formatting
- Let the model use its training for step-by-step reasoning

### Regex Patterns
- Primary pattern: `Final Answer:\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?)`
- Fallback patterns for common variations
- Handle both "Final Answer:" and "Answer:" formats
- Support scientific notation and negative numbers

### Performance Considerations
- Single model loading reduces memory usage
- Regex parsing is faster than LLM parsing
- Batch processing for multiple problems
- Optional GPU memory optimization