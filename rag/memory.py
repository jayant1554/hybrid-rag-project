from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, AIMessage

class ChatHistory:
    def __init__(self, max_turns: int = 10):
        self.history: list = []
        self.max_turns = max_turns

    def add(self, question: str, answer: str):
        self.history.append({"question": question, "answer": answer})
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]

    def format_history(self) -> str:
        if not self.history:
            return ""
        turns = [f"Human: {h['question']}\nAssistant: {h['answer']}" for h in self.history[:-1]]
        return "\n\n".join(turns)

    def clear(self):
        self.history = []
