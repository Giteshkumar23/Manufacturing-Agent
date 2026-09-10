"""
Copilot Agent — thin wrapper providing the chat interface through the orchestrator.
"""
from agents.orchestrator import chat_copilot

class CopilotAgent:
    name = "CopilotAgent"

    def chat(self, message: str, factory_context: dict) -> str:
        return chat_copilot(message, factory_context)
