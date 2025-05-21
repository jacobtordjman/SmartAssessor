import os
from pathlib import Path
from gpt4all import GPT4All

class LocalLLM:
    def __init__(self):
        base_dir = Path(__file__).parent
        models_dir = base_dir / "models"
        model_file = "llama-2-7b-chat.gguf"
        model_path = models_dir / model_file

        print(f"Loading GPT4All model from {model_path} ...")
        self.llm = GPT4All(
            model_name=model_file,
            model_path=str(models_dir),
            allow_download=False,
            verbose=False
        )
        print("✅ GPT4All loaded successfully.")

    def chat(self, student_blob: str) -> str:
        system = (
            "Evaluate each submitted equation or linear-algebra solution strictly for correctness. "
            "Do not add any greetings or commentary.  \n"
            "For each item, output exactly:\n"
            "Q1: Correct.\n"
            "Q2: Incorrect. The correct answer is <answer>.  \n"
            "(If incorrect, briefly explain why.)\n"
            "Maintain the original numbering."
        )

        # Wrap in Llama-2 Chat INST tags:
        prompt = (
            "[INST] <<SYS>>\n"
            f"{system}\n"
            "<</SYS>>\n\n"
            f"{student_blob}"
            " [/INST]"
        )

        # DEBUG: show exactly what goes to the model
        print("┌──[LocalLLM] PROMPT TO MODEL────────────────────────────────────────────")
        print(prompt)
        print("└─────────────────────────────────────────────────────────────────────────")

        full_output = self.llm.generate(prompt)

        print(f"[LocalLLM] raw model output (first 200 chars):\n{full_output[:200]!r}")

        # No need to strip echo if using INST tags, but keep in case:
        if full_output.startswith(prompt):
            response = full_output[len(prompt):].strip()
        else:
            response = full_output.strip()

        print(f"[LocalLLM] cleaned response (first 200 chars):\n{response[:200]!r}")
        return response

# Singleton instance
llm = LocalLLM()
