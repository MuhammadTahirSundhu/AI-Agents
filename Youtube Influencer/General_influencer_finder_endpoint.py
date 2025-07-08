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
    
    return "general"  # Default fallback

def extract_social_media_platforms(user_input):
    """Extract social media platforms from user input."""
    user_input_lower = user_input.lower()
    platforms = []
    
    for platform, code in SOCIAL_MEDIA_TYPES.items():
        if platform in user_input_lower:
            if code not in platforms:
                platforms.append(code)
    
    return ",".join(platforms) if platforms else None

def extract_location(user_input):
    """Extract location from user input."""
    user_input_lower = user_input.lower()
    
    for location, mapped_location in LOCATION_MAPPING.items():
        if location in user_input_lower:
            return mapped_location
    
    return None

def extract_number_of_results(user_input):
    """Extract the number of results requested from user input."""
    # Look for patterns like "find 10 influencers", "get 5 creators", etc.
    patterns = [
        r'find (\d+)',
        r'get (\d+)',
        r'show (\d+)',
        r'(\d+) influencers?',
        r'(\d+) creators?',
        r'(\d+) bloggers?'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, user_input.lower())
        if match:
            return int(match.group(1))
    
    return 10  # Default

def should_fetch_influencers(user_input, chat_history):
    """Determine if we should fetch influencers based on user input and chat history."""
    # Keywords that indicate user wants to find influencers
    fetch_keywords = [
        "find", "search", "get", "show", "looking for", "need", "want",
        "influencer", "blogger", "creator", "content creator", "youtuber"
    ]
    
    user_input_lower = user_input.lower()
    
    # Check if user input contains fetch keywords
    for keyword in fetch_keywords:
        if keyword in user_input_lower:
            return True
    
    # Check if chat history is empty or doesn't contain influencer data
    if not chat_history:
        return True
    
    # Check if recent chat history contains influencer data
    recent_messages = chat_history[-5:]  # Check last 5 messages
    for msg in recent_messages:
        if "influencer" in msg.get("content", "").lower():
            return False  # Already have influencer data
    
    return True

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
    """Call the Google Gemini API."""
    prompt_text = prompt.text if hasattr(prompt, 'text') else str(prompt)
    
    # Prepare messages for Gemini
    messages = [
        SystemMessage(content="""You are an ethical AI Influencer Sourcing Agent designed to assist users in finding social media influencers for marketing campaigns or answering general queries. 
        Your responses must be honest, transparent, and respect privacy. When you have influencer data available, analyze and rank it based on relevance, engagement, and quality metrics.
        Always be helpful and provide clear, actionable responses."""),
        *chat_history,
        HumanMessage(content=prompt_text)
    ]
    
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

def validate_social_types(social_types: str) -> str:
    """Validate and normalize social media types."""
    if not social_types:
        return "INST,FB,TW,YT,TT,TG"  # Default to all platforms
    valid_types = {"INST", "FB", "TW", "YT", "TT", "TG"}
    types = [t.strip().upper() for t in social_types.split(",") if t.strip()]
    validated_types = [t for t in types if t in valid_types]
    if not validated_types:
        raise ValueError("No valid social media types provided")
    return ",".join(validated_types)

def validate_location(location: Optional[str]) -> Optional[str]:
    """Validate and normalize location."""
    if not location:
        return None
    location = location.lower()
    return LOCATION_MAPPING.get(location, None)

def fetch_social_media_influencers(
    query: str,
    page: int = 1,
    per_page: int = 10,
    sort: str = "-score",
    location: Optional[str] = None,
    social_types: Optional[str] = None
) -> Dict:
    """Fetch influencers from multiple social media platforms using the API."""
    if not query:
        raise ValueError("Query parameter is required")

    url = "https://instagram-statistics-api.p.rapidapi.com/search"
    querystring = {
        "q": query.strip(),
        "page": str(max(1, int(page))),
        "perPage": str(per_page),
        "sort": sort.strip(),
        "trackTotal": "true"
    }

    if location:
        querystring["locations"] = location
    if social_types:
        querystring["socialTypes"] = social_types

    headers = {
        "x-rapidapi-key": os.getenv("INSTAGRAM_API_KEY", "your_default_api_key"),
        "x-rapidapi-host": "instagram-statistics-api.p.rapidapi.com"
    }

    try:
        response = requests.get(url, headers=headers, params=querystring, timeout=15)
        response.raise_for_status()
        api_response = response.json()

        if not isinstance(api_response, dict):
            raise ValueError("Invalid API response format")
        if "data" not in api_response or not isinstance(api_response["data"], list):
            return {"data": [], "error": "No influencer data returned by API"}

        print(f"API Response Status: {response.status_code}")
        print(f"API Response: {api_response}")
        return api_response

    except requests.exceptions.HTTPError as he:
        print(f"HTTP Error: {str(he)}")
        return {"data": [], "error": f"API request failed with HTTP error: {str(he)}"}
    except requests.exceptions.RequestException as re:
        print(f"Request Error: {str(re)}")
        return {"data": [], "error": f"Failed to fetch influencers: {str(re)}"}
    except Exception as e:
        print(f"Unexpected Error: {str(e)}")
        return {"data": [], "error": f"Unexpected error: {str(e)}"}

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

def parse_social_media_data_to_context_string(api_response: Dict) -> str:
    """Parse Social Media API response to a human-readable context string."""
    if not isinstance(api_response, dict) or "data" not in api_response:
        return "Error: Invalid or missing API response data"

    data = api_response.get("data", [])
    if not data:
        error_msg = api_response.get("error", "No social media influencer data available")
        return error_msg

    context_parts = ["Social Media Influencers Data:"]

    # Group by social media type
    influencers_by_type = {}
    for influencer in data:
        if not isinstance(influencer, dict):
            continue
        social_type = influencer.get("socialType", "Unknown").upper()
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

            # Safe handling of numeric fields
            users_count = influencer.get("usersCount", 0)
            context_parts.append(f"   - Followers/Subscribers: {int(users_count):,}" if users_count else "   - Followers/Subscribers: 0")

            context_parts.append(f"   - Profile URL: {influencer.get('url', 'N/A')}")
            context_parts.append(f"   - Profile Image: {influencer.get('image', 'N/A')}")

            avg_er = influencer.get("avgER", 0)
            context_parts.append(f"   - Average Engagement Rate: {float(avg_er):.4f}%" if avg_er else "   - Average Engagement Rate: 0.0000%")

            avg_interactions = influencer.get("avgInteractions", 0)
            context_parts.append(f"   - Average Interactions: {int(avg_interactions):,}" if avg_interactions else "   - Average Interactions: 0")

            avg_views = influencer.get("avgViews", None)
            context_parts.append(f"   - Average Views: {int(avg_views):,}" if isinstance(avg_views, (int, float)) else "   - Average Views: N/A")

            quality_score = influencer.get("qualityScore", 0)
            context_parts.append(f"   - Quality Score: {float(quality_score):.4f}" if quality_score else "   - Quality Score: 0.0000")

            context_parts.append(f"   - Verified: {influencer.get('verified', False)}")

            # Safe handling of location data
            cities = influencer.get("membersCities", [])
            if cities and isinstance(cities, list):
                city_strings = [
                    f"{city.get('name', 'Unknown')} ({float(city.get('value', 0)):.2%})"
                    for city in cities
                    if isinstance(city, dict) and city.get("name")
                ]
                if city_strings:
                    context_parts.append(f"   - Top Cities: {', '.join(city_strings[:3])}")

            countries = influencer.get("membersCountries", [])
            if countries and isinstance(countries, list):
                country_strings = [
                    f"{country.get('name', 'Unknown')} ({float(country.get('value', 0)):.2%})"
                    for country in countries
                    if isinstance(country, dict) and country.get("name")
                ]
                if country_strings:
                    context_parts.append(f"   - Top Countries: {', '.join(country_strings[:3])}")

    return "\n".join(context_parts) if context_parts else "No valid influencer data found"

def create_prompt_template():
    """Create and return the prompt template for the universal social media chatbot."""
    return PromptTemplate(
        template="""
        You are a helpful AI assistant that specializes in finding and analyzing social media influencers.

        **Your Current Task:**
        Based on the user's query and any available chat history data, provide a helpful response.

        **Available Data:**
        Chat History: {ChatHistory}
        User Query: {Query}

        **Instructions:**
        1. If the chat history contains influencer data, analyze it and provide insights, rankings, or summaries as requested.
        2. If the user is asking for new influencer data and no relevant data exists in chat history, indicate that fresh data needs to be fetched.
        3. For general questions about influencers, social media, or marketing, provide helpful information.
        4. Always be clear, helpful, and professional in your responses.
        5. If you need to fetch new influencer data, clearly state what information you need and suggest the next steps.
        6. Don't give any fake influencer data, always fetch from the API.
        

        **Response Guidelines:**
        - Be concise but comprehensive
        - Use clear formatting for any data presentation
        - Provide actionable insights when possible
        - If data is missing, clearly explain what's needed
        """,
        input_variables=['ChatHistory', 'Query']
    )

async def process_influencer_query(input_query, chat_history_dict, user_id):
    """Process a single social media influencer query with automatic fetching."""
    # Check for exit command
    exit_commands = ["exit", "quit", "end"]
    if input_query.lower().strip() in exit_commands:
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

    # Initialize variables
    chat_history = convert_dict_to_langchain_messages(chat_history_dict)
    chat_history.append(HumanMessage(content=input_query))
    
    ranked_influencers = []
    fetch_attempted = False
    api_result = None
    
    # Check if we should fetch new influencer data
    if should_fetch_influencers(input_query, chat_history_dict):
        print("Determining if we should fetch new influencer data...")
        
        # Extract parameters from user query
        niche = extract_niche_from_query(input_query)
        location = extract_location(input_query)
        social_platforms = extract_social_media_platforms(input_query)
        num_results = extract_number_of_results(input_query)
        
        print(f"Extracted parameters - Niche: {niche}, Location: {location}, Platforms: {social_platforms}, Results: {num_results}")
        
        # Fetch influencers immediately if we have extracted parameters
        if niche and niche != "general":
            try:
                print(f"Fetching influencers for niche: {niche}")
                api_result = fetch_social_media_influencers(
                    query=niche,
                    page=1,
                    per_page=min(num_results, 50),
                    sort="-score",
                    location=location,
                    social_types=social_platforms
                )
                
                print(f"API Result received: {len(api_result.get('data', []))} influencers")
                
                if api_result.get("data"):
                    # Parse and add to chat history
                    parsed_data = parse_social_media_data_to_context_string(api_result)
                    chat_history.append(AIMessage(content=parsed_data))
                    fetch_attempted = True
                    
                    # Create ranked influencers
                    ranked_influencers = []
                    for influencer in api_result["data"]:
                        # Safe conversion functions
                        def safe_float(value, default=0.0):
                            if value is None:
                                return default
                            try:
                                return float(value)
                            except (ValueError, TypeError):
                                return default
                        
                        def safe_int(value, default=0):
                            if value is None:
                                return default
                            try:
                                return int(value)
                            except (ValueError, TypeError):
                                return default
                        
                        # Extract values with null safety
                        followers = safe_int(influencer.get("usersCount"))
                        engagement_rate = safe_float(influencer.get("avgER"))
                        avg_interactions = safe_int(influencer.get("avgInteractions"))
                        quality_score = safe_float(influencer.get("qualityScore"))
                        
                        # Calculate ranking score with null safety
                        niche_match = 1 if (niche.lower() in str(influencer.get("name", "")).lower() or 
                                          niche.lower() in str(influencer.get("screenName", "")).lower()) else 0.5
                        
                        ranking_score = (
                            0.35 * niche_match +
                            0.30 * min(engagement_rate / 100, 1) +
                            0.20 * min(followers / 1000000, 1) +
                            0.15 * quality_score
                        ) * 100
                        
                        ranked_influencers.append({
                            "Influencer Name": influencer.get("name", "Unknown"),
                            "Handle": f"@{influencer.get('screenName', 'unknown')}",
                            "Platform": get_social_media_display_name(str(influencer.get("socialType", "Unknown")).upper()),
                            "Followers": followers,
                            "Engagement Rate": engagement_rate,
                            "Average Interactions": avg_interactions,
                            "Quality Score": quality_score,
                            "Verified": bool(influencer.get("verified", False)),
                            "Location": ", ".join([city.get("name", "Unknown") for city in influencer.get("membersCities", [])]) or "N/A",
                            "Profile URL": influencer.get("url", "N/A"),
                            "Ranking Score": ranking_score
                        })
                    
                    # Sort by ranking score
                    ranked_influencers.sort(key=lambda x: x["Ranking Score"], reverse=True)
                    
                else:
                    error_msg = api_result.get("error", "No influencer data returned by API")
                    chat_history.append(AIMessage(content=f"I tried to fetch {niche} influencers but encountered an issue: {error_msg}"))
            
            except Exception as e:
                error_msg = f"Error fetching social media influencer data: {str(e)}"
                chat_history.append(AIMessage(content=error_msg))
                print(f"Debug info: {error_msg}")  # For debugging
                
    # If we don't have fresh data, use LLM to process the query
    if not fetch_attempted:
        prompt_template = create_prompt_template()
        final_prompt = prompt_template.invoke({"ChatHistory": convert_langchain_to_dict(chat_history), "Query": input_query})
        response = call_llm(final_prompt, chat_history)
        
        # Ensure response is a string
        if not isinstance(response, str):
            response = str(response)
        
        chat_history.append(AIMessage(content=response))

    # Format the response
    updated_chat_history = convert_langchain_to_dict(chat_history)
    final_response = format_ranked_influencers(ranked_influencers) if ranked_influencers else updated_chat_history[-1]["content"]

    return {
        "updated_chat_history": updated_chat_history,
        "response": final_response,
        "database_stored": False,
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
            output.append(f"   - Engagement Rate: {influencer.get('Engagement Rate', 0):.4f}%")
            output.append(f"   - Average Interactions: {influencer.get('Average Interactions', 0):,}")
            output.append(f"   - Quality Score: {influencer.get('Quality Score', 0):.4f}")
            output.append(f"   - Verified: {influencer.get('Verified', False)}")
            output.append(f"   - Location: {influencer.get('Location', 'N/A')}")
            output.append(f"   - Profile URL: {influencer.get('Profile URL', 'N/A')}")
            output.append(f"   - Ranking Score: {influencer.get('Ranking Score', 0):.2f}")
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
            "response": result["response"]
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