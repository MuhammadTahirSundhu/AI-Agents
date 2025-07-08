from fastapi import FastAPI, HTTPException
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import re
import os
import json
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import PromptTemplate
from notion_database import create_chat_history, get_chat_history_for_user
import requests

# Load environment variables
load_dotenv()

app = FastAPI(title="Universal Social Media Influencer Finder API", version="1.0.0")

# Social Media Types Mapping
SOCIAL_MEDIA_TYPES = {
    "instagram": "INST",
    "facebook": "FB", 
    "twitter": "TW",
    "youtube": "YT",
    "tiktok": "TT",
    "telegram": "TG",
    "inst": "INST",
    "fb": "FB",
    "tw": "TW",
    "yt": "YT", 
    "tt": "TT",
    "tg": "TG"
}

# Common locations mapping
LOCATION_MAPPING = {
    "pakistan": "pakistan",
    "india": "india", 
    "usa": "usa",
    "uk": "uk",
    "canada": "canada",
    "australia": "australia",
    "germany": "germany",
    "france": "france",
    "japan": "japan",
    "brazil": "brazil",
    "turkey": "turkey",
    "saudi arabia": "saudi arabia",
    "uae": "uae",
    "united states": "usa",
    "united kingdom": "uk"
}

# Enhanced niche/category keywords mapping
NICHE_KEYWORDS = {
    "fitness": ["fitness", "gym", "workout", "training", "bodybuilding", "crossfit", "yoga", "pilates", "health"],
    "fashion": ["fashion", "style", "clothing", "outfit", "designer", "model", "runway", "trends"],
    "food": ["food", "cooking", "recipe", "chef", "restaurant", "cuisine", "foodie", "culinary", "baking"],
    "travel": ["travel", "tourism", "adventure", "vacation", "destination", "wanderlust", "explore"],
    "tech": ["technology", "tech", "gadgets", "software", "coding", "programming", "ai", "startup", "electronic", "electronics"],
    "beauty": ["beauty", "makeup", "skincare", "cosmetics", "hair", "nails", "spa"],
    "lifestyle": ["lifestyle", "life", "daily", "routine", "motivation", "inspiration"],
    "gaming": ["gaming", "games", "esports", "streamer", "gamer", "console", "pc"],
    "music": ["music", "musician", "singer", "artist", "band", "concert", "album"],
    "sports": ["sports", "athlete", "football", "basketball", "soccer", "tennis", "cricket"],
    "education": ["education", "teacher", "learning", "study", "academic", "school", "university"],
    "business": ["business", "entrepreneur", "startup", "marketing", "finance", "investment"],
    "entertainment": ["entertainment", "comedy", "funny", "humor", "celebrity", "show"],
    "art": ["art", "artist", "painting", "drawing", "creative", "design", "gallery"],
    "pets": ["pets", "dog", "cat", "animal", "puppy", "kitten", "pet care"],
    "parenting": ["parenting", "family", "kids", "children", "mom", "dad", "baby"],
    "automotive": ["car", "auto", "vehicle", "motorcycle", "racing", "automotive"]
}

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

def extract_niche_from_query(user_input):
    """Extract niche/category from user input using enhanced keyword matching."""
    user_input_lower = user_input.lower()
    
    # Direct keyword matching
    for niche, keywords in NICHE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in user_input_lower:
                return niche
    
    # Pattern-based extraction for specific phrases
    niche_patterns = [
        r'(\w+)\s+influencers?',  # "fitness influencers"
        r'(\w+)\s+bloggers?',     # "food bloggers"
        r'(\w+)\s+content creators?',  # "tech content creators"
        r'influencers?\s+in\s+(\w+)',  # "influencers in fitness"
        r'(\w+)\s+creators?',     # "fashion creators"
        r'(\w+)\s+channels?',     # "cooking channels"
        r'(\w+)\s+accounts?',     # "travel accounts"
    ]
    
    for pattern in niche_patterns:
        match = re.search(pattern, user_input_lower)
        if match:
            potential_niche = match.group(1)
            # Check if the extracted word is a known niche
            if potential_niche in NICHE_KEYWORDS:
                return potential_niche
            # Check if it matches any keywords
            for niche, keywords in NICHE_KEYWORDS.items():
                if potential_niche in keywords:
                    return niche
    
    # If no specific niche found, try to extract any meaningful category word
    # This is a fallback for cases where the niche might not be in our predefined list
    words = user_input_lower.split()
    for word in words:
        if len(word) > 3 and word not in ['find', 'search', 'get', 'show', 'influencers', 'bloggers', 'creators']:
            # Return the first meaningful word as potential niche
            return word
    
    return "general"  # Default fallback

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
        SystemMessage(content="""You are an ethical AI Influencer Sourcing Agent designed to assist users in finding social media influencers for marketing campaigns or answering general queries. 
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

def parse_social_media_types(user_input):
    """Parse social media types from user input."""
    user_input_lower = user_input.lower()
    detected_types = []
    
    for platform, code in SOCIAL_MEDIA_TYPES.items():
        if platform in user_input_lower:
            if code not in detected_types:
                detected_types.append(code)
    
    # If no specific platform mentioned, return None (will search all)
    return ",".join(detected_types) if detected_types else None

def parse_location(user_input):
    """Parse location from user input."""
    user_input_lower = user_input.lower()
    
    for location_name, location_code in LOCATION_MAPPING.items():
        if location_name in user_input_lower:
            return location_code
    
    # If no specific location mentioned, return None (will search globally)
    return None

def get_social_media_display_name(social_type):
    """Get display name for social media type."""
    display_names = {
        "INST": "Instagram",
        "FB": "Facebook", 
        "TW": "Twitter",
        "YT": "YouTube",
        "TT": "TikTok",
        "TG": "Telegram"
    }
    return display_names.get(social_type, social_type)

def fetch_social_media_influencers(query: str, page: int = 1, per_page: int = 10, sort: str = "-score", location: str = None, social_types: str = None):
    """Fetch influencers from multiple social media platforms using the API."""
    url = "https://instagram-statistics-api.p.rapidapi.com/search"
    
    querystring = {
        "q": query,
        "page": str(page),
        "perPage": str(per_page),
        "sort": sort,
        "trackTotal": "true"
    }
    
    # Add location only if specified
    if location:
        querystring["locations"] = location
    
    # Add social types only if specified
    if social_types:
        querystring["socialTypes"] = social_types
    
    headers = {
        "x-rapidapi-key": os.getenv("RAPIDAPI_KEY", "160c11660emsh2a96a4835527853p158f25jsnbc0d7223389d"),
        "x-rapidapi-host": "instagram-statistics-api.p.rapidapi.com"
    }
    
    try:
        response = requests.get(url, headers=headers, params=querystring)
        response.raise_for_status()
        api_response = response.json()
        print(f"API Response Status: {response.status_code}")
        print(f"API Response: {api_response}")
        return api_response
    except requests.RequestException as e:
        print(f"API Request Error: {str(e)}")
        raise Exception(f"Failed to fetch social media influencers: {str(e)}")

def parse_social_media_data_to_context_string(api_response):
    """Parse Social Media API response to context string with better error handling."""
    if not api_response:
        return "No social media influencer data available."
    
    # Handle different response formats
    if isinstance(api_response, dict):
        if "data" in api_response:
            data = api_response["data"]
        elif "results" in api_response:
            data = api_response["results"]
        else:
            # If the response itself contains influencer data
            data = [api_response] if not isinstance(api_response, list) else api_response
    else:
        data = api_response if isinstance(api_response, list) else [api_response]
    
    if not data:
        return "No social media influencer data available."
    
    context_parts = ["Social Media Influencers Data:"]
    
    # Group by social media type for better organization
    influencers_by_type = {}
    
    for influencer in data:
        if not isinstance(influencer, dict):
            continue
            
        social_type = influencer.get('socialType', 'Unknown')
        if social_type not in influencers_by_type:
            influencers_by_type[social_type] = []
        influencers_by_type[social_type].append(influencer)
    
    for social_type, influencers in influencers_by_type.items():
        platform_name = get_social_media_display_name(social_type)
        context_parts.append(f"\n=== {platform_name} Influencers ===")
        
        for i, influencer in enumerate(influencers, 1):
            context_parts.append(f"\n{i}. {platform_name} Influencer:")
            context_parts.append(f"   - Name: {influencer.get('name', 'Unknown')}")
            context_parts.append(f"   - Handle: @{influencer.get('screenName', 'unknown')}")
            context_parts.append(f"   - Platform: {platform_name}")
            context_parts.append(f"   - Followers/Subscribers: {influencer.get('usersCount', 0):,}")
            context_parts.append(f"   - Profile URL: {influencer.get('url', 'N/A')}")
            context_parts.append(f"   - Profile Image: {influencer.get('image', 'N/A')}")
            context_parts.append(f"   - Average Engagement Rate: {influencer.get('avgER', 0):.4f}")
            context_parts.append(f"   - Average Interactions: {influencer.get('avgInteractions', 0)}")
            context_parts.append(f"   - Average Views: {influencer.get('avgViews', 'N/A')}")
            context_parts.append(f"   - Quality Score: {influencer.get('qualityScore', 0):.4f}")
            context_parts.append(f"   - Verified: {influencer.get('verified', False)}")
            
            # Add location information
            if 'membersCities' in influencer and influencer['membersCities']:
                cities = [str(city) for city in influencer['membersCities'] if city]
                if cities:
                    context_parts.append(f"   - Top Cities: {', '.join(cities[:3])}")
            
            if 'membersCountries' in influencer and influencer['membersCountries']:
                countries = [str(country) for country in influencer['membersCountries'] if country]
                if countries:
                    context_parts.append(f"   - Top Countries: {', '.join(countries[:3])}")
    
    return "\n".join(context_parts)

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
    """Create and return the prompt template for the universal social media chatbot."""
    return PromptTemplate(
        template="""
        You are an ethical Universal Social Media Influencer Sourcing Agent. Your task is to interpret user queries, identify the appropriate function to call, and handle general questions with moral, responsible answers. You have access to one function for fetching influencer data from ALL social media platforms. Follow these guidelines strictly:

        **CRITICAL INSTRUCTIONS**:
        ⚠️ **NEVER PROVIDE FAKE, DUMMY, OR FABRICATED DATA** ⚠️
        - Only work with REAL data provided in the Chat History section
        - If no chat history data is available, do NOT create fictional influencer information
        - If chat history is empty or contains no influencer data, only suggest fetching new data or ask for clarification
        - NEVER generate sample data, placeholder information, or hypothetical results
        - If the user query is 'exit', 'quit', or 'end', respond with 'Session ended. Chat history saved.' and do not process further.

        **Supported Social Media Platforms**:
        - Instagram (INST)
        - Facebook (FB)
        - Twitter (TW)
        - YouTube (YT)
        - TikTok (TT)
        - Telegram (TG)

        **Available Functions**:
           1. **Name**: Fetch Social Media Influencers
              - **Description**: Fetch influencers from multiple social media platforms based on search query, location, and platform preferences.
              - **Parameters**:
                 - query (string): NICHE/CATEGORY extracted from user request (fitness, fashion, food, tech, etc.)
                 - max_results (integer): maximum 50 influencers to fetch
                 - location (string): OPTIONAL - country/region (pakistan, india, usa, etc.). If not specified, searches globally
                 - social_types (string): OPTIONAL - comma-separated platform codes (INST,YT,TT, etc.). If not specified, searches all platforms
                 - sort (string): sorting criteria (-score, -usersCount, -avgER, etc.)
              - **Query Extraction Examples**:
                 - "Find fitness influencers" → query=fitness (Global search, all platforms)
                 - "Find YouTube fitness influencers in Pakistan" → query=fitness, social_types=YT, location=pakistan
                 - "Instagram and TikTok fashion influencers" → query=fashion, social_types=INST,TT (Global search)
                 - "Food bloggers in USA" → query=food, location=usa (All platforms)
                 - "Tech content creators" → query=tech (Global search, all platforms)
                 - "Beauty influencers on Instagram" → query=beauty, social_types=INST
                 - "Electronic gadgets influencers" → query=tech (tech covers electronics/gadgets)
              - **Command Examples**:
                 - "Fetch Social Media Influencer: query=fitness, max_results=10"
                 - "Fetch Social Media Influencer: query=fashion, max_results=5, social_types=INST,TT"
                 - "Fetch Social Media Influencer: query=food, max_results=8, location=usa"
                 - "Fetch Social Media Influencer: query=tech, max_results=12, location=pakistan, social_types=YT,TW, sort=-usersCount"

        **Enhanced Niche Extraction Rules**:
           - Always extract the main niche/category from the user's request
           - Common niches: fitness, fashion, food, travel, tech, beauty, lifestyle, gaming, music, sports, education, business, entertainment, art, pets, parenting, automotive
           - Look for keywords like: "influencers", "bloggers", "creators", "content creators", "channels", "accounts"
           - Pattern matching: "fitness influencers" → fitness, "food bloggers" → food, "tech creators" → tech
           - Electronics/gadgets/technology all map to "tech" niche
           - If no clear niche is found, use the most relevant category word from the request
           - Never use vague terms like "influencer" or "creator" as the query - always extract the specific niche

        **Data Processing Rules**:
           1. **Check Chat History First**: 
              - ALWAYS examine the Chat History section for existing influencer data
              - If Chat History contains influencer data, proceed to ranking immediately
              - If Chat History is empty or contains no influencer data, suggest fetching new data
              
           2. **Parameter Extraction Rules**:
              - Extract NICHE/CATEGORY from user request (fitness, fashion, food, tech, etc.) - THIS IS THE MOST IMPORTANT
              - Extract number for max_results (default: 10, max: 50)
              - Extract location ONLY if user specifically mentions a country/region
              - Extract social media platforms ONLY if user specifically mentions them
              - Sort options: -score (relevance), -usersCount (followers), -avgER (engagement rate)
              - Default behavior: If location or social_types not specified, search globally across all platforms
              
           3. **General Queries**:
              - For non-influencer queries, provide concise, ethical answers
              - If you ask you about what you can do or how you work, explain your role
              - Don't give fetch commands for general queries
              - Don't tell them you have not data or chat history is empty
              - If user asks about your capabilities, explain you can fetch influencer data and answer general questions
              
           4. **Ranking Available Influencer Data**:
              - **ONLY rank if Chat History contains actual influencer data**
              - If Chat History has influencer data, immediately rank based on:
                - Relevance to Query/Niche (35%): Match with requested category
                - Engagement Quality (30%): Engagement rate and interaction quality
                - Platform Authority (20%): Follower count and platform influence
                - Authenticity & Quality (15%): Verification status and quality score
                - Score (0-100): Combined weighted score
                - Rank in descending order by score, show all available details
                - Group by platform for better organization
                - End response with "Finished"
           
           5. **Ranked Result Format**:
                When ranking influencers from Chat History data, extract and show these details in JSON format:
                Store_in_Notion_database  
                ```json
                [
                  {{
                    "Influencer Name": "actual_name_from_chat_history",
                    "Handle": "actual_handle_from_chat_history",
                    "Platform": "actual_platform_from_chat_history",
                    "Followers": actual_followers_count_number,
                    "Engagement Rate": actual_engagement_rate_number,
                    "Average Interactions": actual_avg_interactions_number,
                    "Quality Score": actual_quality_score_number,
                    "Verified": actual_verification_status,
                    "Location": "actual_location_from_chat_history",
                    "Profile URL": "actual_profile_url_from_chat_history",
                    "Ranking Score": actual_ranking_score_number
                  }}
                ]
                ```

        **Response Logic**:
        1. If the user query is 'exit', 'quit', or 'end', respond with 'Session ended. Chat history saved.' and do not process further
        2. Check if Chat History contains social media influencer data
        3. If YES: Immediately rank the available data and provide JSON output
        4. If NO: Extract niche from user request and determine if user wants to fetch new data or answer general questions
        5. Never mix real data with fake data
        6. Always be transparent about data availability
        7. Always extract the specific niche/category from the user's request for the query parameter
        8. If user doesn't specify location or platform, search globally across all platforms

        **Chat History Data**: {ChatHistory}
        **User Query**: {Query}
        
        Remember: 
        - if chat history is empty dont not tell in response that chat history is empty
        - Use ONLY real data from Chat History section
        - ALWAYS extract the specific niche/category from user request for the query parameter
        - If user doesn't specify location, search globally (don't add location parameter)
        - If user doesn't specify social media platform, search all platforms (don't add social_types parameter)
        - If Chat History has data, rank it immediately - don't ask to fetch more unless specifically requested
        - Focus on cross-platform metrics and organize results by platform
        - The query parameter should always be the extracted niche/category, not generic terms
        - Electronic/gadgets/technology all map to "tech" niche
        """,
        input_variables=['ChatHistory', 'Query']
    )

def parse_fetch_command(response):
    """Parse the Fetch Social Media Influencer command from LLM response."""
    # More flexible pattern to handle optional parameters
    base_pattern = r"Fetch Social Media Influencer:\s*query=([^,]+),\s*max_results=(\d+)"
    match = re.search(base_pattern, response)
    
    if not match:
        return None, None, None, None, None
    
    query = match.group(1).strip()
    max_results = int(match.group(2))
    
    # Extract optional parameters
    location = None
    social_types = None
    sort_type = "-score"  # default
    
    # Look for location parameter
    location_match = re.search(r"location=([^,\s]+)", response)
    if location_match:
        location = location_match.group(1).strip()
    
    # Look for social_types parameter
    social_types_match = re.search(r"social_types=([^,\s]+)", response)
    if social_types_match:
        social_types = social_types_match.group(1).strip()
    
    # Look for sort parameter
    sort_match = re.search(r"sort=([^,\s]+)", response)
    if sort_match:
        sort_type = sort_match.group(1).strip()
    
    return query, max_results, location, social_types, sort_type

async def process_influencer_query(input_query, chat_history_dict, user_id):
    """Process a single social media influencer query with enhanced niche extraction."""
    # Check for exit command
    exit_commands = ["exit", "quit", "end"]
    if input_query.lower().strip() in exit_commands:
        # Store chat history in Notion
        chat_history = convert_dict_to_langchain_messages(chat_history_dict)
        try:
            chat_history_id = await create_chat_history(
                input_query=chat_history_dict[0]["content"] if chat_history_dict else "Help me to find social media influencers",
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
            "ranked_influencers": []
        }

    # Normal query processing
    chat_history = convert_dict_to_langchain_messages(chat_history_dict)
    chat_history.append(HumanMessage(content=input_query))
    prompt_template = create_prompt_template()
    final_prompt = prompt_template.invoke({"ChatHistory": convert_langchain_to_dict(chat_history), "Query": input_query})
    response = call_llm(final_prompt, chat_history)
    
    print("LLM Response:", response)
    
    chat_history.append(AIMessage(content=response))
    ranked_influencers = []
    fetch_attempted = False

    if "```json" in response or "Store_in_Notion_database" in response:
        ranked_influencers = extract_json_response_to_list(response)
        print("Extracted Ranked Influencers:", ranked_influencers)

    if "Fetch Social Media Influencer" in response:
        query, max_results, location, social_types, sort_type = parse_fetch_command(response)
        
        # Enhanced niche extraction - fallback if LLM didn't extract properly
        if not query or query.lower() in ['influencer', 'creator', 'general']:
            extracted_niche = extract_niche_from_query(input_query)
            query = extracted_niche
            print(f"Enhanced niche extraction: {extracted_niche}")
        
        if query and max_results:
            try:
                # Parse additional parameters from user input if not in LLM response
                if not location:
                    location = parse_location(input_query)
                if not social_types:
                    social_types = parse_social_media_types(input_query)
                
                print(f"Fetching influencers - Query: {query}, Max Results: {max_results}, Location: {location}, Social Types: {social_types}, Sort: {sort_type}")
                
                # Fetch social media influencers using the API
                result = fetch_social_media_influencers(
                    query=query,
                    page=1,
                    per_page=min(max_results, 50),  # API limit
                    sort=sort_type or "-score",
                    location=location,
                    social_types=social_types
                )
                
                print(f"API Result: {result}")
                
                new_data = parse_social_media_data_to_context_string(result)
                print(f"Parsed Data: {new_data}")
                
                chat_history.pop()  # Remove last AIMessage
                
                # Create a more informative response
                platforms_searched = "all platforms" if not social_types else f"platforms: {social_types}"
                location_searched = "globally" if not location else f"in {location}"

                success_message = f"{new_data} \n\n📊 Found {query} influencers from {platforms_searched} {location_searched}. Do you want me to rank them based on their engagement, quality score, relevance, and platform authority?"
                chat_history.append(AIMessage(content=success_message))
                fetch_attempted = True
            except Exception as e:
                error_msg = f"Error fetching social media influencer data: {str(e)}"
                chat_history.append(AIMessage(content=error_msg))
        else:
            error_msg = "Error: Missing required parameters for social media influencer search."
            chat_history.append(AIMessage(content=error_msg))
    
    updated_chat_history = convert_langchain_to_dict(chat_history)
    response = chat_history[-1].content if fetch_attempted else response
    return {
        "updated_chat_history": updated_chat_history,
        "response": response,
        "database_stored": False,  # No storage until exit
        "fetch_attempted": fetch_attempted,
        "ranked_influencers": ranked_influencers
    }

def format_ranked_influencers(ranked_influencers):
    """Format ranked social media influencers into a human-readable string."""
    if not ranked_influencers:
        return "No ranked influencers available."
    
    # Group by platform for better organization
    platform_groups = {}
    for influencer in ranked_influencers:
        platform = influencer.get('Platform', 'Unknown')
        if platform not in platform_groups:
            platform_groups[platform] = []
        platform_groups[platform].append(influencer)
    
    output = ["🌟 Ranked Social Media Influencers:", "=" * 80]
    
    for platform, influencers in platform_groups.items():
        output.append(f"\n📱 {platform} Influencers:")
        output.append("-" * 60)
        
        for i, influencer in enumerate(influencers, 1):
            output.append(f"{i}. {influencer.get('Influencer Name', 'Unknown Influencer')}")
            output.append(f"   - Handle: {influencer.get('Handle', 'N/A')}")
            output.append(f"   - Platform: {influencer.get('Platform', 'N/A')}")
            output.append(f"   - Followers: {influencer.get('Followers', 0):,}")
            output.append(f"   - Engagement Rate: {influencer.get('Engagement Rate', 0):.4f}")
            output.append(f"   - Average Interactions: {influencer.get('Average Interactions', 0):,}")
            output.append(f"   - Quality Score: {influencer.get('Quality Score', 0):.4f}")
            output.append(f"   - Verified: {influencer.get('Verified', False)}")
            output.append(f"   - Location: {influencer.get('Location', 'N/A')}")
            output.append(f"   - Profile URL: {influencer.get('Profile URL', 'N/A')}")
            output.append(f"   - Ranking Score: {influencer.get('Ranking Score', 0)}")
            output.append("-" * 60)
    
    return "\n".join(output)

@app.post('/Social_Media_Influencer_Finder', response_model=List[InfluencerFinderResponse])
async def social_media_influencer_finder(request: InfluencerFinderRequest):
    """Main endpoint for Universal Social Media Influencer Finder chatbot."""
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

        if result["database_stored"]:
            print("💾 Chat history successfully stored in database")
        
        if result["fetch_attempted"]:
            print("🔍 Data fetch was attempted")

        if result["ranked_influencers"]:
            print("📊 Ranked Influencers:", json.dumps(result["ranked_influencers"], indent=2))
        
        return [{
            "chat_history": [ChatMessage(**msg) for msg in result["updated_chat_history"]],
            "response": format_ranked_influencers(result["ranked_influencers"]) if result["ranked_influencers"] else result["response"]
        }]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/health', response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", service="Universal Social Media Influencer Finder")

if __name__ == '__main__':
    import uvicorn
    print("Universal Social Media Influencer Finder API Started!")
    print("Available endpoints:")
    print("- POST /Social_Media_Influencer_Finder")
    print("- GET /health")
    print("Supported Platforms: Instagram, Facebook, Twitter, YouTube, TikTok, Telegram")
    print("-" * 80)
    # Use __file__ to get the current module name automatically
    module_name = __file__.split('\\')[-1].split('.')[0] if '\\' in __file__ else __file__.split('/')[-1].split('.')[0]
    uvicorn.run(f"{module_name}:app", host="0.0.0.0", port=5000, reload=True)