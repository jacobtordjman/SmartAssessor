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
        print("[LocalLLM] Initializing with WizardMath backend...")
        # Initialize the WizardMath model preferring local/Drive paths (no downloads)
        # If BASE_MODEL_PATH and ADAPTER_PATH point to Google Drive, the loader will run fully offline.
        self.wizardmath = WizardMathLLM(force_ai_mode=False, use_local_cache=True)
        print(f"[LocalLLM] Ready. model_loaded={self.wizardmath.model_loaded}, lightweight_mode={self.wizardmath.lightweight_mode}")

    def chat(self, student_blob: str) -> str:
        """
        Process student submission for grading.
        
        Args:
            student_blob: Raw text containing student's work
            
        Returns:
            Formatted grading response
        """
        print(f"[LocalLLM.chat] called. text_len={len(student_blob)} lightweight={self.wizardmath.lightweight_mode} model_loaded={self.wizardmath.model_loaded}")
        # Use the WizardMath chat method
        return self.wizardmath.chat(student_blob)

# Singleton instance
llm = LocalLLM()
