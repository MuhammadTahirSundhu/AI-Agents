import json
import os
from typing import Dict, List, Optional, Annotated
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_together import ChatTogether

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Fiverr Gigs Chatbot API", version="1.0.0")

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# llm = ChatOpenAI(
#     model="gpt-4o", 
#     temperature=0.2, 
#     openai_api_key=os.getenv("OPENAI_API_KEY")
# )

# llm = ChatTogether(
#     model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
#     temperature=0.2,
#     together_api_key=os.getenv("TOGETHER_API_KEY")
# )

# Pydantic models for request/response
class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str

class ChatRequest(BaseModel):
    input: str
    history: List[ChatMessage] = []

class ChatResponse(BaseModel):
    history: List[ChatMessage]
    response: str

# Tools
@tool
def get_gigs(query: str) -> List[Dict[str, str]]:
    """Fetches a list of Fiverr gigs from a JSON file based on the user's query."""
    gigs_file = "gigs.json"
    # print(f"Fetching gigs from file : {gigs_file}")
    # Check if file exists
    if not os.path.exists(gigs_file):
        raise FileNotFoundError(f"The file {gigs_file} does not exist.")
    
    # Read the JSON file
    with open(gigs_file, "r") as f:
        gigs_data = json.load(f)
    
    print(f"Fetching gigs for query: {query}")
    
    # Extract gigs from the JSON data
    gigs = gigs_data.get("gig_list", {}).get("gigs", [])
    
    # Filter gigs where query matches title or sub_category_name (case-insensitive)
    filtered_gigs = [
        gig for gig in gigs 
        if query.lower() in gig["title"].lower() 
        or query.lower() in gig["sub_category_name"].lower()
    ]
    
    # Return filtered gigs, or up to 10 gigs if no matches found
    return filtered_gigs or gigs[:10]

@tool
def chatbot_keyword_search(project_details: str) -> str:
    """
    Chatbot function to handle keyword search for Fiverr gigs.
    """
    # Invoke the LLM with the project details
    response = llm.invoke(f"project_details: {project_details} \n\nPlease provide the most relevant and most suitable Fiverr keyword based on the above project details. in your response there should be format like \"find the <keyword> gigs\" and nothing else.")
    print(f"Keyword search response: {response.content}")
    return response.content

# Initialize tools
tools = [get_gigs, chatbot_keyword_search]
llm_with_tools = llm.bind_tools(tools=tools)

# State definition for LangGraph
class State(TypedDict):
    """
    State for the specific Fiverr Gigs Fetcher Agent.
    """
    messages: Annotated[List[BaseMessage], add_messages]

def chatbot(state: State):
    """
    Chatbot function to handle the conversation with the user.
    """
    # Invoke the LLM with all messages
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def build_chatbot_graph():
    """Build and compile the chatbot graph."""
    builder = StateGraph(State)
    builder.add_node("chatbot", chatbot)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "chatbot")
    builder.add_conditional_edges(
        "chatbot",
        tools_condition,
        {
            "tools": "tools",
            "__end__": END,
        },
    )
    builder.add_edge("tools", "chatbot")
    
    graph = builder.compile()
    return graph

def convert_history_to_messages(history: List[ChatMessage]) -> List[BaseMessage]:
    """Convert API history format to LangChain messages."""
    messages = []
    for msg in history:
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            messages.append(AIMessage(content=msg.content))
    return messages

def convert_messages_to_history(messages: List[BaseMessage]) -> List[ChatMessage]:
    """Convert LangChain messages to API history format."""
    history = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            history.append(ChatMessage(role="user", content=msg.content))
        elif isinstance(msg, AIMessage):
            history.append(ChatMessage(role="assistant", content=msg.content))
        elif isinstance(msg, SystemMessage):
            history.append(ChatMessage(role="system", content=msg.content))
        # Skip SystemMessage from history output
    return history

# System message for the chatbot
SYSTEM_MESSAGE = SystemMessage(content="""
# Fiverr Gig  Assistant System Prompt

## Primary Role
You are a helpful assistant specialized in fetching and recommending Fiverr gigs based on user queries. if user asks for capabilities, provide a brief overview of this Fiverr Gig fetcher Assistant  and how you can assist them.

## Instructions
  - Dont give dummy gigs responses, always use the tools to fetch the gigs.
  - Use the `chatbot_keyword_search` tool to process keywords from user queries.
                               
## Core Functionality

### General Queries
- For general questions unrelated to Fiverr projects, respond gracefully with helpful information
- Maintain a conversational and supportive tone

### Fiverr-Related Queries
When users provide:
- Specific project details
- Keywords related to Fiverr services
- Requests in the format "find the [keyword] gigs"
- Detailed project descriptions (paragraphs explaining what they want done)

**Then you should:**
1. Use the `chatbot_keyword_search` tool for keyword processing
2. Use the `get_gigs` tool to fetch relevant gigs
3. Pass the user's project details or keywords to the appropriate tool

## Gig Display Process

### Step 1: Display Fetched Gigs
- Present all fetched gigs in an **attractive, well-formatted display**
- Include **all available details** for each gig
- Use clear headings, bullet points, and organized layout

### Step 2: Offer Personalized Recommendations
After displaying the gigs, ask:
> "Would you like me to analyze these gigs and recommend the 3 most suitable ones for your specific project?"

### Step 3: Handle User Response
**If user says YES:**
- Compare all fetched gigs against the user's project requirements
- Analyze factors like: relevance, seller ratings, pricing, delivery time, reviews, etc.
- Display the **3 most suitable gigs** with detailed explanations of why they're recommended in a very clear and organized manner

**If user says NO:**
- End the conversation gracefully
- Offer assistance with other queries if needed

## Additional Support

### Gig Details
- If user requests more information about a specific gig, provide comprehensive details
- Include seller information, package options, reviews, delivery times, etc.

### New Searches
- If user asks for different keywords or new project types, restart the process
- Use the same systematic approach for each new query

## Response Style
- Be professional yet friendly
- Use clear, organized formatting
- Focus on being helpful and efficient
- Provide actionable information
""")

# Initialize the graph globally
graph = build_chatbot_graph()

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint that processes user input and returns AI response.
    """
    try:
        state_messages = []
        # Add conversation history
        print("request history:", request.history)
        history_messages = convert_history_to_messages(request.history)
        if not history_messages:
            # Initialize state with system message
            print("Initializing state with system message only")
            state_messages = [SYSTEM_MESSAGE]

        state_messages.extend(history_messages)
        
        # Add current user input
        state_messages.append(HumanMessage(content=request.input))
        # Create state
        state: State = {
            "messages": state_messages
        }
        
        # Run the graph
        result = graph.invoke(state)
        
        # Get the updated messages
        updated_messages = result["messages"]
        
        # Convert to API format (excluding system message)
        api_history = convert_messages_to_history(updated_messages)  # Skip system message
        # Get the latest AI response
        latest_ai_message = None
        for msg in reversed(updated_messages):
            if isinstance(msg, AIMessage):
                latest_ai_message = msg.content
                break
        
        if latest_ai_message is None:
            latest_ai_message = "No response generated"
        
        return ChatResponse(
            history=api_history,
            response=latest_ai_message
        )
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Fiverr Gigs Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "chat": "POST /chat - Main chat endpoint",
            "health": "GET /health - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Fiverr Gigs Chatbot API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)