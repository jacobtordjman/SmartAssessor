import os
from pathlib import Path
from backend.app.wizardmath_llm import WizardMathLLM

# Keep the LocalLLM class for backward compatibility
class LocalLLM:
    """
    Wrapper class for backward compatibility with the existing codebase.
    Uses the new WizardMathLLM implementation under the hood.
    """
    def __init__(self):
        print("Initializing LocalLLM with WizardMath backend...")
        # Initialize the WizardMath model in force AI mode for full capabilities
        # Allow remote downloads on fresh environments (e.g., Colab)
        self.wizardmath = WizardMathLLM(force_ai_mode=True, use_local_cache=False)
        print("✅ LocalLLM (WizardMath) loaded successfully.")

    def chat(self, student_blob: str) -> str:
        """
        Process student submission for grading.
        
        Args:
            student_blob: Raw text containing student's work
            
        Returns:
            Formatted grading response
        """
        # Use the WizardMath chat method
        return self.wizardmath.chat(student_blob)

# Singleton instance
llm = LocalLLM()
