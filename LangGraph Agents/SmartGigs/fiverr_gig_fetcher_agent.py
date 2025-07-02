from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

llm = init_chat_model("google_genai:gemini-2.0-flash")
