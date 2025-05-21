# agents.py

from autogen.agentchat.conversable_agent import ConversableAgent
from .llm_client import llm

def chunk_text(text: str, max_chars: int = 4000) -> list[str]:
    pieces, cur = [], ""
    for block in text.split("\n\n"):
        if not cur:
            cur = block
        elif len(cur) + len(block) + 2 <= max_chars:
            cur += "\n\n" + block
        else:
            pieces.append(cur.strip())
            cur = block
    if cur:
        pieces.append(cur.strip())
    return pieces

class GradingAgent(ConversableAgent):
    """
    A simple grading agent that uses a local GPT4All model
    to mark each question as correct or incorrect.
    """
    def __init__(self):
        super().__init__(
            name="grading_agent",
            system_message=(
                "You are an expert grader. The user will send you the full text "
                "of a student's submission (questions + answers). For each question, "
                "reply with one of:\n"
                "- \"Q#: Correct.\"\n"
                "- \"Q#: Incorrect. The correct answer is ... because ...\"\n"
                "Keep each response concise and numbered."
            ),
        )

    async def a_receive(self, message, sender, request_reply=None):
        """
        Chunk the incoming student submission if too long,
        grade each piece, and send back the combined feedback.
        """
        pieces = chunk_text(message)
        feedbacks = [llm.chat(piece) for piece in pieces]
        combined = "\n\n".join(feedbacks)
        await self.a_send(combined, sender)

# Singleton instance
grading_agent = GradingAgent()
