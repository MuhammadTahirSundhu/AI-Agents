from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

import os

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

class State(TypedDict):
    """
    State for the specific Fiverr Gigs Fetcher Agent.
    """
    messages: list[HumanMessage | AIMessage]
    context: str | None

def chatbot(state: State) -> State:
    """
    Chatbot function to handle the conversation with the user.
    """
    input = state["context"] + state["messages"][-1].content
    response = llm.invoke(input)
    state["messages"].append(AIMessage(content=response.content))
    return state

def build_chatbot_graph():
    builder = StateGraph(State)
    builder.add_node("chatbot_node", chatbot)
    builder.add_edge(START, "chatbot_node")
    builder.add_edge("chatbot_node", END)
    graph = builder.compile()
    return graph

def run_chatbot():
    """
    Run the chatbot with an initial state.
    """
    graph = build_chatbot_graph()
    state: State = {
        "messages": [],
        "context":""
    }

    while True:
        user_input = input("Enter your query: ")
        if user_input.lower() == "exit":
            print("Exiting the chatbot.")
            return state
        state["messages"].append(HumanMessage(content=user_input))
        state = graph.invoke(state)
        print("Response:", state["messages"][-1].content)
    

if __name__ == "__main__":
    history = run_chatbot()
    print(f"Chat history:\n\n")
    for message in history["messages"]:
        if isinstance(message, HumanMessage):
            print(f"You: {message.content}")
        elif isinstance(message, AIMessage):
            print(f"AI: {message.content}\n")
    