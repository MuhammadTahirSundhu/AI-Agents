from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import asyncio
import re
import os
import json
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import PromptTemplate
from youtube_apis import read_and_parse_json , fetch_channel_with_their_avg_comments , parse_json_to_context_string
from notion_database import create_chat_history, get_chat_history_for_user
import requests
# Load environment variables
load_dotenv()

app = FastAPI(title="YouTube Influencer Finder API", version="1.0.0")

# Pydantic models for request/response
class ChatMessage(BaseModel):
    role: str
    content: str

class InfluencerFinderRequest(BaseModel):
    input_query: str
    ChatHistory: List[ChatMessage] = []
    user_id: str

class InfluencerFinderResponse(BaseModel):
    chat_history: List[ChatMessage]
    response: str

class HealthResponse(BaseModel):
    status: str
    service: str

def convert_dict_to_langchain_messages(chat_history_dict):
    """Convert dictionary format chat history to LangChain message objects."""
    messages = []
    for msg in chat_history_dict:
        if msg["role"] in ["user", "human"]:
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] in ["assistant", "ai"]:
            messages.append(AIMessage(content=msg["content"]))
        elif msg["role"] == "system":
            messages.append(SystemMessage(content=msg["content"]))
    return messages

def convert_langchain_to_dict(chat_history):
    """Convert LangChain message objects to dictionary format."""
    return [{"role": msg.type, "content": msg.content} for msg in chat_history]

# def call_llm(prompt, chat_history):
#     """Call the external LLM API."""
#     url = "https://206c-20-106-58-127.ngrok-free.app/chat"
#     prompt_text = prompt.text if hasattr(prompt, 'text') else str(prompt)
    
#     # Convert chat history to string format for the system message
#     chat_history_str = ""
#     for msg in chat_history:
#         chat_history_str += f"{msg.type}: {msg.content}\n"
    
#     payload = {
#         "messages": [
#             {
#                 "role": "system",
#                 "content": (
#                     "You are an ethical AI Influencer Sourcing Agent designed to assist users in "
#                     "finding YouTube influencers for marketing campaigns or answering general queries. "
#                     "Your responses must be honest, transparent, and respect privacy. "
#                     "You have access to a function for fetching influencer data and must store results before responding. "
#                     "Interpret user intent, suggest functions when unclear, and prompt for missing parameters."
#                     f"\n**Chat History**\n{chat_history_str}"
#                 )
#             },
#             {
#                 "role": "user",
#                 "content": prompt_text
#             }
#         ],
#         "temperature": 0.5,
#         "model": "gpt-4o"
#     }
    
#     try:
#         response = requests.post(url, json=payload)
#         response.raise_for_status()
#         api_response = response.json()
#         if api_response.get("status") == "success":
#             return api_response.get("response")
#         else:
#             return f"Error: API request failed - {api_response.get('message', 'No message provided')}"
#     except requests.RequestException as e:
#         return f"Error: Failed to connect to the API - {str(e)}"

def call_llm(prompt, chat_history):
    """Call the OpenAI API with GPT-4o model."""
    from openai import OpenAI
    
    prompt_text = prompt.text if hasattr(prompt, 'text') else str(prompt)
    
    role_mapping = {
        "human": "user",
        "ai": "assistant",
        "system": "system",
        "assistant": "assistant",
        "user": "user"
    }
    
    messages = [
        {"role": "system", "content": """You are an ethical AI Influencer Sourcing Agent designed to assist users in finding YouTube influencers for marketing campaigns or answering general queries. 
        Your responses must be honest, transparent, and respect privacy. You have access to a function for fetching influencer data and must store results before responding. 
        Interpret user intent, suggest functions when unclear, and prompt for missing parameters."""},
        *[{"role": role_mapping.get(msg.type, "user"), "content": msg.content} for msg in chat_history],
        {"role": "user", "content": prompt_text}
    ]

    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI API failed: {str(e)}")
        return f"Error: Failed to connect to OpenAI API - {str(e)}"
    
def extract_json_response_to_list(input_text):
    """Extract JSON data from input text and store it in a list."""
    try:
        json_pattern = r'```json\n(.*?)\n```'
        match = re.search(json_pattern, input_text, re.DOTALL)
        
        if not match:
            return []
        
        json_string = match.group(1)
        parsed_data = json.loads(json_string)
        
        if not isinstance(parsed_data, list):
            raise ValueError("JSON data must be a list of objects")
        
        return parsed_data
    
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error processing JSON: {str(e)}")
        return []

def create_prompt_template():
    """Create and return the prompt template for the chatbot."""
    return PromptTemplate(
        template="""
        You are an ethical YouTube Influencer Sourcing Agent. Your task is to interpret user queries, identify the appropriate function to call, and handle general questions with moral, responsible answers. You have access to one function for fetching influencer data. Follow these guidelines strictly:

        **CRITICAL INSTRUCTIONS**:
        ⚠️ **NEVER PROVIDE FAKE, DUMMY, OR FABRICATED DATA** ⚠️
        - Only work with REAL data provided in the Chat History section
        - If no chat history data is available, do NOT create fictional influencer information
        - If chat history is empty or contains no influencer data, only suggest fetching new data or ask for clarification
        - NEVER generate sample data, placeholder information, or hypothetical results
        - If the user query is 'exit', 'quit', or 'end', respond with 'Session ended. Chat history saved.' and do not process further.

        **Available Functions**:
           1. **Name**: Fetch Influencers
              - **Description**: Fetch YouTube influencers based on trending or niche criteria. Results include channel info, popular videos, and comments.
              - **Parameters**:
                 - method (integer): 1 for trending channels, 2 for niche-based channels
                 - max_results (integer): maximum 20 channels to fetch
                 - niche (string): required ONLY for method=2, leave empty for method=1
              - **Examples**:
                 - For trending: "Fetch Influencer: method=1, max_results=5, niche="
                 - For niche: "Fetch Influencer: method=2, max_results=3, niche=fitness"

        **Data Processing Rules**:
           1. **Check Chat History First**: 
              - ALWAYS examine the Chat History section for existing influencer data
              - If Chat History contains influencer data, proceed to ranking immediately
              - If Chat History is empty or contains no influencer data, suggest fetching new data
              
           2. **For trending channels (method=1)**:
              - Only need method and max_results parameters
              - Always set niche as empty string
              - Example response: "Fetch Influencer: method=1, max_results=3, niche= "
              
           3. **For niche-based channels (method=2)**:
              - Need all three parameters: method, max_results, and niche
              - Extract niche from user query (fitness, gaming, tech, etc.)
              - Example response: "Fetch Influencer: method=2, max_results=5, niche=fitness"
              
           4. **Parameter Extraction Rules**:
              - If user mentions "trending", use method=1
              - If user specifies a topic/niche (fitness, gaming, etc.) or does not mention "trending", use method=2
              - Extract number from query for max_results
              - If parameters missing, ask user to clarify
              
           5. **General Queries**:
              - For non-influencer queries, provide concise, ethical answers
              - If you ask you about what you can do or how you work, explain your role
              - Don't give "Fetch Influencer: method=1, max_results=3, niche= " and "Fetch Influencer: method=2, max_results=5, niche=fitness" for general queries
              
           6. **Ranking Available Influencer Data**:
              - **ONLY rank if Chat History contains actual influencer data**
              - If Chat History has influencer data, immediately rank based on:
                - Relevance to Niche (40%): Match keywords, titles, comments with user's requested niche
                - Audience Engagement (30%): Comment sentiment, engagement quality, and relevance  
                - Popularity (30%): Subscriber count and video views
                - Score (0-100): (0.4 * niche_relevance + 0.3 * engagement + 0.3 * popularity)
                - Rank in descending order by score, show all available details
                - End response with "Finished"
           
           7. **Ranked Result Format**:
                When ranking channels from Chat History data, extract and show these details in JSON format:
                Store_in_Notion_database  
                ```json
                [
                  {{
                    "Channel Name": "actual_channel_name_from_chat_history",
                    "Handle": "actual_handle_from_chat_history",
                    "Description": "actual_description_from_chat_history",
                    "Subscribers": actual_subscriber_count_number,
                    "Total Views": actual_total_views_number,
                    "Videos Count": actual_video_count_number,
                    "Country": "actual_country_from_chat_history",
                    "Ranking Score": "actual_ranking_score_number"
                  }}
                ]
                ```

        **Response Logic**:
        1. If the user query is 'exit', 'quit', or 'end', respond with 'Session ended. Chat history saved.' and do not process further
        2. Check if Chat History contains influencer data
        3. If YES: Immediately rank the available data and provide JSON output
        4. If NO: Determine if user wants to fetch new data or answer general questions
        5. Never mix real data with fake data
        6. Always be transparent about data availability

        **Chat History Data**: {ChatHistory}
        **User Query**: {Query}
        
        Remember: 
        - if chat history is empty dont not tell in response that chat history is empty
        - Use ONLY real data from Chat History section
        - For method=1 (trending), always use empty niche
        - For method=2, extract niche from user query
        - If Chat History has data, rank it immediately - don't ask to fetch more unless specifically requested
        """,
        input_variables=['ChatHistory', 'Query']
    )

def parse_fetch_command(response):
    """Parse the Fetch Influencer command from LLM response."""
    pattern = r"Fetch Influencer:\s*method=(\d+),\s*max_results=(\d+),\s*niche=([^,\n]*)"
    match = re.search(pattern, response)
    
    if match:
        method = int(match.group(1))
        max_results = int(match.group(2))
        niche_raw = match.group(3).strip()
        
        niche = None if method == 1 else (niche_raw if niche_raw else None)
        return method, max_results, niche
    return None, None, None

async def process_influencer_query(input_query, chat_history_dict, user_id):
    """Process a single influencer query with improved logging."""
    # Check for exit command
    exit_commands = ["exit", "quit", "end"]
    if input_query.lower().strip() in exit_commands:
        # Store chat history in Notion
        chat_history = convert_dict_to_langchain_messages(chat_history_dict)
        try:
            chat_history_id = await create_chat_history(
                input_query=chat_history_dict[0]["content"] if chat_history_dict else "Help me to find influencers",
                user_id=user_id,
                chat_history=convert_langchain_to_dict(chat_history),
                response="Session ended. Chat history saved."
            )
            database_stored = bool(chat_history_id)
            print("Database Storage Result:", database_stored)
        except Exception as e:
            print(f"Error storing chat history in database: {str(e)}")
            database_stored = False

        return {
            "updated_chat_history": convert_langchain_to_dict(chat_history),
            "response": "Session ended. Chat history saved.",
            "database_stored": database_stored,
            "fetch_attempted": False,
            "ranked_channels": []
        }

    # Normal query processing
    chat_history = convert_dict_to_langchain_messages(chat_history_dict)
    chat_history.append(HumanMessage(content=input_query))
    prompt_template = create_prompt_template()
    final_prompt = prompt_template.invoke({"ChatHistory": convert_langchain_to_dict(chat_history), "Query": input_query})
    response = call_llm(final_prompt, chat_history)
    
    print("LLM Response:", response)
    
    chat_history.append(AIMessage(content=response))
    ranked_channels = []
    fetch_attempted = False

    if "```json" in response or "Store_in_Notion_database" in response:
        ranked_channels = extract_json_response_to_list(response)
        print("Extracted Ranked Channels:", ranked_channels)

    if "Fetch Influencer" in response:
        method, max_results, niche = parse_fetch_command(response)
        if method is not None and max_results is not None:
            if method == 2 and not niche:
                error_msg = "Error: Niche is required for method=2."
                chat_history.append(AIMessage(content=error_msg))
            else:
                try:
                    #  Uncomment the next line to use actual API call
                    result = fetch_channel_with_their_avg_comments(method, max_results, niche)
                    new_data = parse_json_to_context_string(result)
                    # for getting data from file
                    new_data = read_and_parse_json("./channel_comments.json")
                    chat_history.pop()  # Remove last AIMessage
                    chat_history.append(AIMessage(content=f"{new_data} \n\n DO you want me to rank them on basis of their popularity?"))
                    fetch_attempted = True
                except Exception as e:
                    error_msg = f"Error fetching influencer data: {str(e)}"
                    chat_history.append(AIMessage(content=error_msg))
    
    updated_chat_history = convert_langchain_to_dict(chat_history)
    response = chat_history[-1].content if fetch_attempted else response
    return {
        "updated_chat_history": updated_chat_history,
        "response": response,
        "database_stored": False,  # No storage until exit
        "fetch_attempted": fetch_attempted,
        "ranked_channels": ranked_channels
    }

def format_ranked_channels(ranked_channels):
    """Format ranked channels into a human-readable string."""
    output = ["📊 Ranked Channels:", "-" * 50]
    for i, channel in enumerate(ranked_channels, 1):
        output.append(f"{i}. {channel.get('Channel Name', 'Unknown Channel')}")
        output.append(f"   - Handle: {channel.get('Handle', 'N/A')}")
        output.append(f"   - Subscribers: {channel.get('Subscribers', 0)}")
        output.append(f"   - Total Views: {channel.get('Total Views', 0)}")
        output.append(f"   - Videos Count: {channel.get('Videos Count', 0)}")
        output.append(f"   - Country: {channel.get('Country', 'N/A')}")
        output.append(f"   - Ranking Score: {channel.get('Ranking Score', 0)}")
        output.append("-" * 50)
    return "\n".join(output)

@app.post('/Youtube_Influencer_Finder', response_model=List[InfluencerFinderResponse])
async def youtube_influencer_finder(request: InfluencerFinderRequest):
    """Main endpoint for YouTube Influencer Finder chatbot."""
    try:
        input_query = request.input_query
        chat_history = [msg.dict() for msg in request.ChatHistory]
        user_id = request.user_id

        if not chat_history:
            result = await get_chat_history_for_user(user_id)
            if result != "No chat history found.":
                chat_history = result
        
        print("Chat History:", chat_history)
        
        if not input_query:
            raise HTTPException(status_code=400, detail="input_query is required")
                
        # Process the query
        result = await process_influencer_query(input_query, chat_history, user_id)

        # result["updated_chat_history"].append({
        #     "role": "assistant",
        #     "content": f"Ranked Channels:\n{json.dumps(result['ranked_channels'], indent=2)}"
        # })        

        if result["database_stored"]:
            print("💾 Chat history successfully stored in database")
        
        if result["fetch_attempted"]:
            print("🔍 Data fetch was attempted")

        if result["ranked_channels"]:
            print("📊 Ranked Channels:", json.dumps(result["ranked_channels"], indent=2))
        
        return [{
            "chat_history": [ChatMessage(**msg) for msg in result["updated_chat_history"]],
            "response": format_ranked_channels(result["ranked_channels"]) if result["ranked_channels"] else result["response"]
        }]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/health', response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", service="YouTube Influencer Finder")

if __name__ == '__main__':
    import uvicorn
    print("YouTube Influencer Finder API Started!")
    print("Available endpoints:")
    print("- POST /Youtube_Influencer_Finder")
    print("- GET /health")
    print("-" * 50)
    uvicorn.run("Youtube_influencer_endpoint:app", host="0.0.0.0", port=5000, reload=True)