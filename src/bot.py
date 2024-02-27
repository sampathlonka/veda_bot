from llama_index.chat_engine.condense_plus_context import CondensePlusContextChatEngine
from llama_index.memory import ChatMemoryBuffer

memory = ChatMemoryBuffer.from_defaults(token_limit=3900)

context = """
  You are an expert on Vedas and related scriptures.\
  Your role is to respond to questions about vedic scriptures and associated information based on available sources.\
  For every query, you must use either any one of the tool or use available history/context.
  Please provide well-informed answers. Don't use prior knowledge.
"""

react_agent = ReActAgent.from_tools(tools, llm=llm_AI4, context=context, verbose=True)

sample_prompts = ["What is the padapatha of a vedamantra from RigVeda, mandala 1, shukta 84, mantra 3?", "What is the meaning of the vedamantra from RigVeda, first mandala, first shukta, and first mantra?",
                  "Describe the first kandah,second shukta from Atharvaveda?","How many mantras are there in RigVeda?"]
class Veda_Bot:
    def __init__(self, memory, retriever=None):
        self.memory = memory
    def respond(message, chat_history):
        bot_message = react_agent.chat(message).response
        chat_history.append((message, bot_message))
        return " ", chat_history
