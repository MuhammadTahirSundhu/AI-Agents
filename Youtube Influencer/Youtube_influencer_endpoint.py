
from flask import Flask, request, jsonify
import asyncio
import re
import requests
import os
import json
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from youtube_apis import fetch_channel_with_their_avg_comments, parse_json_to_context_string, read_and_parse_json
from notion_database import create_influencer, create_query, create_user, notion

# Load environment variables
load_dotenv()

app = Flask(__name__)

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

def call_llm(prompt, chat_history):
    """Call the external LLM API or fallback to Google Gemini if API fails."""
    url = "https://206c-20-106-58-127.ngrok-free.app/chat"
    prompt_text = prompt.text if hasattr(prompt, 'text') else str(prompt)
    
    # Prepare messages for API or Gemini
    messages = [
        SystemMessage(content="""You are an ethical AI Influencer Sourcing Agent designed to assist users in finding YouTube influencers for marketing campaigns or answering general queries. 
        Your responses must be honest, transparent, and respect privacy. You have access to a function for fetching influencer data and must store results before responding. 
        Interpret user intent, suggest functions when unclear, and prompt for missing parameters."""),
        *chat_history,
        HumanMessage(content=prompt_text)
    ]

    # Try external API first
    try:
        payload = {
            "messages": [{"role": msg.type, "content": msg.content} for msg in messages],
            "temperature": 0.5,
            "model": "gpt-4o"
        }
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        api_response = response.json()
        if api_response.get("status") == "success":
            return api_response.get("response")
        else:
            print(f"API request failed: {api_response.get('message', 'No message provided')}")
    except (requests.RequestException, ValueError) as e:
        print(f"External API failed: {str(e)}. Switching to Google Gemini.")

    # Fallback to Google Gemini
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.5
        )
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"Error: Failed to connect to Google Gemini - {str(e)}"

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
                    "Channel ID": "actual_channel_id_from_chat_history",
                    "Handle": "actual_handle_from_chat_history",
                    "Description": "actual_description_from_chat_history",
                    "Subscribers": actual_subscriber_count_number,
                    "Total Views": actual_total_views_number,
                    "Videos Count": actual_video_count_number,
                    "Joined Date": "actual_joined_date_from_chat_history",
                    "Country": "actual_country_from_chat_history",
                    "Top Video Links": "actual_video_links_from_chat_history",
                    "Top Comments": "actual_comments_from_chat_history",
                    "External Links": "actual_external_links_from_chat_history",
                    "Last Updated": "current_date",
                    "Ranking Score": calculated_score_0_to_100
                  }}
                ]
                ```

        **Response Logic**:
        1. First, check if Chat History contains influencer data
        2. If YES: Immediately rank the available data and provide JSON output
        3. If NO: Determine if user wants to fetch new data or answer general questions
        4. Never mix real data with fake data
        5. Always be transparent about data availability

        **Chat History Data**: {ChatHistory}
        **User Query**: {Query}
        
        Remember: 
        - Use ONLY real data from Chat History section
        - For method=1 (trending), always use empty niche
        - For method=2, extract niche from user query
        - If Chat History has data, rank it immediately - don't ask to fetch more unless specificallyColour requested
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
        
        if method == 1:
            niche = None
        else:
            niche = niche_raw if niche_raw else None
            
        return method, max_results, niche
    return None, None, None

async def store_influencers_in_database(influencers, user_id):
    """Store influencers in the database with enhanced error handling."""
    try:
        if not user_id:
            print("Error: user_id is missing")
            return False

        if not influencers:
            print("Error: No influencers provided for storage")
            return False

        for influencer in influencers:
            required_fields = ["Channel Name", "Subscribers", "Total Views"]
            if not all(key in influencer for key in required_fields):
                print(f"Skipping influencer due to missing fields: {influencer}")
                continue

            influencer_id = await create_influencer(
                channel_name=influencer.get("Channel Name", ""),
                channel_url=influencer.get("External Links", ""),
                views=influencer.get("Total Views", 0),
                subscribers=influencer.get("Subscribers", 0),
                video_count=influencer.get("Videos Count", 0),
                handle=influencer.get("Handle", ""),
                description=influencer.get("Description", ""),
                country=influencer.get("Country", ""),
                joined_date=influencer.get("Joined Date", ""),
                top_video_links=influencer.get("Top Video Links", ""),
                top_comments=influencer.get("Top Comments", ""),
                user_id=user_id
            )
            if not influencer_id:
                print(f"Failed to create influencer {influencer.get('Channel Name', 'Unknown')}")
                return False

        print("All influencers stored successfully")
        return True
    except Exception as e:
        print(f"Error storing influencers: {str(e)}")
        return False
    finally:
        await notion.aclose()

def process_influencer_query(input_query, chat_history_dict, user_id):
    """Process a single influencer query with improved logging."""
    chat_history = convert_dict_to_langchain_messages(chat_history_dict)
    chat_history.append(HumanMessage(content=input_query))
    
    prompt_template = create_prompt_template()
    final_prompt = prompt_template.invoke({"ChatHistory": convert_langchain_to_dict(chat_history), "Query": input_query})
    response = call_llm(final_prompt, chat_history)
    
    print("LLM Response:", response)
    
    chat_history.append(AIMessage(content=response))
    database_stored = False
    ranked_channels = []
    if "```json" in response or "Store_in_Notion_database" in response:
        ranked_channels = extract_json_response_to_list(response)
        print("Extracted Ranked Channels:", ranked_channels)
        if ranked_channels:
            try:
                database_stored = asyncio.run(store_influencers_in_database(ranked_channels, user_id))
                print("Database Storage Result:", database_stored)
            except Exception as e:
                print(f"Error storing in database: {str(e)}")
    print("Extracted Ranked Channels:", ranked_channels)


    fetch_attempted = False
    if "Fetch Influencer" in response:
        method, max_results, niche = parse_fetch_command(response)
        if method is not None and max_results is not None:
            if method == 2 and not niche:
                error_msg = "Error: Niche is required for method=2."
                chat_history.append(AIMessage(content=error_msg))
            else:
                try:
                    new_data = read_and_parse_json("./channel_comments.json")  # Replace with actual API call
                    chat_history.append(AIMessage(content=new_data))
                    fetch_attempted = True
                except Exception as e:
                    error_msg = f"Error fetching influencer data: {str(e)}"
                    chat_history.append(AIMessage(content=error_msg))
    
    updated_chat_history = convert_langchain_to_dict(chat_history)
    
    return {
        "updated_chat_history": updated_chat_history,
        "response": response,
        "database_stored": database_stored,
        "fetch_attempted": fetch_attempted,
        "ranked_channels": ranked_channels
    }

@app.route('/Youtube_Influencer_Finder', methods=['POST'])
def youtube_influencer_finder():
    """Main endpoint for YouTube Influencer Finder chatbot."""
    try:
        # Get input data
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        input_query = data.get('input_query', '')
        chat_history = data.get('ChatHistory', [])  # Expect ChatHistory from client
        user_id = data.get('user_id','')

        if not input_query:
            return jsonify({"error": "input_query is required"}), 400
        
        # Process the query
        result = process_influencer_query(input_query, chat_history,user_id)
        
        # Return the result with ranked channels
        return jsonify({
            "status": "success",
            "updated_chat_history": result["updated_chat_history"],
            "response": result["response"],
            "database_stored": result["database_stored"],
            "fetch_attempted": result["fetch_attempted"],
            "ranked_channels": result["ranked_channels"]  # Include ranked channels in response
        })
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "YouTube Influencer Finder"})

if __name__ == '__main__':
    print("YouTube Influencer Finder API Started!")
    print("Available endpoints:")
    print("- POST /Youtube_Influencer_Finder")
    print("- GET /health")
    print("-" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)