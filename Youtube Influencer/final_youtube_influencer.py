import json
import os
import re
from typing import List, Dict, Any
import asyncio

from langchain_google_genai import ChatGoogleGenerativeAI
from openai import OpenAI
from final_youtube_api import fetch_channel_with_their_avg_comments, read_and_parse_json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import json
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import PromptTemplate
from final_youtube_api import read_and_parse_json , fetch_channel_with_their_avg_comments , parse_json_to_context_string
from notion_database import create_chat_history, get_chat_history_for_user

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Universal Social Media Influencer Finder API",
    description="Advanced AI-powered influencer discovery and analysis across YouTube and Instagram",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Pydantic models for request/response
class ChatMessage(BaseModel):
    role: str
    content: str

class InfluencerFinderRequest(BaseModel):
    input_query: str
    ChatHistory: List[ChatMessage] = []
    user_id: int

class InfluencerFinderResponse(BaseModel):
    chat_history: List[ChatMessage]
    response: str
    fetch_attempted: bool = False
    ranked_channels: List[Dict[str, Any]] = []
    platform_analyzed: str = "both"
    database_stored: bool = False

class HealthResponse(BaseModel):
    status: str
    service: str
    supported_platforms: List[str]
    version: str

# Your existing utility functions
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
    """Call the OpenAI API with GPT-4o model for multi-platform influencer sourcing."""
    prompt_text = prompt.text if hasattr(prompt, 'text') else str(prompt)
    
    role_mapping = {
        "human": "user",
        "ai": "assistant",
        "system": "system",
        "assistant": "assistant",
        "user": "user"
    }
    
    messages = [
        {"role": "system", "content": """You are an ethical AI Multi-Platform Influencer Sourcing Agent designed to assist users in finding YouTube and Instagram influencers for marketing campaigns or answering general queries. 

        PLATFORM CAPABILITIES:
        • YouTube: Full access to trending videos, niche searches, channel analytics, subscriber counts, video performance, and audience comments
        • Instagram: Access to profile data, follower counts, engagement metrics, post performance, story analytics, and audience demographics

        CORE RESPONSIBILITIES:
        1. Interpret user intent for influencer discovery across YouTube and Instagram
        2. Suggest appropriate platform(s) based on campaign goals and target audience
        3. Provide honest, transparent recommendations respecting privacy and ethical guidelines
        4. Store and analyze results before providing comprehensive responses
        5. Prompt for missing parameters (niche, country, platform preference, budget range)

        PLATFORM SELECTION GUIDANCE:
        • YouTube: Long-form content, tutorials, reviews, educational content, product demonstrations
        • Instagram: Visual content, lifestyle, fashion, food, travel, quick engagement, Stories, Reels
        • Both: Maximum reach, diverse content formats, cross-platform campaigns

        RESPONSE FORMAT:
        - Always specify which platform(s) you're analyzing
        - Provide platform-specific metrics and insights
        - Suggest cross-platform strategies when appropriate
        - Include engagement rate comparisons between platforms
        - Recommend content formats suitable for each platform

        You have access to functions for fetching influencer data from both platforms and must store results before responding. 
        When platform preference is unclear, ask the user or suggest the most suitable platform based on their campaign goals."""},
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

def call_common_llm(messages: List[dict]) -> str:
    """Make a call to OpenAI API"""
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ OpenAI API error: {str(e)}")
        return ""

def GetThingsFromInput(input_query: str) -> list:
    """Extract niche, country, max_results, and platform from user input using a single LLM request."""
    combined_prompt = f"""
    Extract the following information from this text: "{input_query}"
    
    Return the results as a JSON object in the following format:
    {{"niche": "<keyword or phrase>", "country": "<country name>", "max_results": "<number>", "platform": "<platform name>"}}
    
    Rules for extraction:
    1. Niche Extraction Rules:
       - If the niche is explicitly specified in the query (e.g., "AI Tools", "machine learning", "digital marketing"), return the exact phrase as mentioned.
       - If the niche is not specified or unclear, analyze the query carefully to identify any niche, category, topic, or industry mentioned or implied.
    
       IMPORTANT: Focus on the ACTUAL TOPIC/CATEGORY, not generic words:
       - Extract the core subject matter, removing generic words like "products", "items", "things", "stuff", "content", "videos", "channels"
       - Examples:
         * "electronic products" → "electronics"
         * "fitness items" → "fitness"
         * "cooking videos" → "cooking"
         * "tech products" → "technology"
         * "gaming content" → "gaming"
         * "beauty products" → "beauty"
         * "educational content" → "education"
         * "business tools" → "business"
    
       - Think about what type of content or influencers would be most relevant for the user's request. Extract the most relevant niche/category, which should be a meaningful topic or industry (e.g., fitness, food, technology, AI, digital marketing, machine learning, electronics, gaming, beauty, education).
    
       - Analyze the context to determine if the user wants trending results or general information:
         * If the query contains words like "trending", "popular", "viral", "hot", "latest", "current", "what's new", "top", "best", "most popular", or similar trending-related terms, return "trending".
         * If it's a general query without specific trending indicators, return "general".
    
       - Return only the core topic/niche keyword or phrase, nothing else.
       - Avoid generic terms like "products", "items", "things", "stuff", "content", "videos", "channels" in your response.

    
    2. Country:
       - Extract the country and return as a 2-letter ISO country code (e.g., "US", "UK", "CA", "AU", "DE", "FR", "IN", etc.).
       - Convert any country name or variant to its corresponding 2-letter code.
       - If no country is mentioned, return empty string.
    
    3. Max_results:
       - Extract the number of results requested (e.g., from phrases like "find 10 influencers", "get 5 creators", "show 20 people").
       - Return only the number as a string.
       - If no number is mentioned, return empty string.
    
    4. Platform:
       - Analyze the context and content of the query to determine which social media platform(s) the user is interested in.
       - Supported platform values: "youtube", "instagram", "both"
       - Platform analysis guidelines:
         * Carefully read and understand the user's intent and context
         * Consider what type of content, creators, or platform features are being discussed
         * Analyze the overall context to determine the most appropriate platform
         * If the query clearly indicates a specific platform (YouTube or Instagram), return that platform
         * If the query suggests multiple platforms or general social media content, return "both"
         * If the query is about general influencer marketing without platform specifics, return "both"
         * If no platform context can be determined from the query, return "both"
       - Use your understanding of social media platforms and user intent to make the best determination
       - Don't rely solely on keyword matching - analyze the full context and meaning
    
    Return only the JSON object as a string, without any Markdown, code fences, or additional text.
    """
    
    try:
        results = ["", "", "", ""]
        messages = [{"role": "user", "content": combined_prompt}]
        result = call_common_llm(messages).strip()
        
        print(f"Raw LLM response: {result}")
        
        if not result:
            print("Error: LLM response is empty")
            return results
        
        cleaned_result = result
        if result.startswith("```json"):
            cleaned_result = result.replace("```json", "").replace("```", "").strip()
        
        try:
            parsed_result = json.loads(cleaned_result)
        except json.JSONDecodeError as e:
            print(f"Error parsing LLM response as JSON: {str(e)}")
            print(f"Cleaned response: {cleaned_result}")
            return results
        
        # Post-process the results
        niche = parsed_result.get('niche', '')
        if niche:
            results[0] = niche.lower()
        
        results[1] = parsed_result.get('country', '')
        
        max_results = parsed_result.get('max_results', '')
        if max_results.isdigit():
            results[2] = max_results
        
        platform = parsed_result.get('platform', '').lower()
        if platform in ['youtube', 'instagram', 'both']:
            results[3] = platform
        
        return results
        
    except Exception as e:
        print(f"Error extracting information: {str(e)}")
        return ["", "", "", ""]

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
    """Create and return the prompt template for the multi-platform influencer analysis chatbot."""
    return PromptTemplate(
        template="""
        You are an expert Multi-Platform Influencer Analysis Agent designed to analyze and rank YouTube and Instagram influencers for marketing campaigns. Your primary role is to evaluate pre-fetched influencer data and provide comprehensive insights for campaign planning.

        **CRITICAL INSTRUCTIONS**:
        ⚠️ **WORK ONLY WITH REAL DATA FROM CHAT HISTORY** ⚠️
        - Only analyze ACTUAL influencer data provided in the Chat History section
        - NEVER generate fake, dummy, or fabricated influencer information
        - If Chat History is empty or contains no influencer data, inform user that data needs to be fetched first
        - NEVER create sample data, placeholder information, or hypothetical results
        - If user query is 'exit', 'quit', or 'end', respond with 'Session ended. Chat history saved.'

        **PLATFORM EXPERTISE**:
        • **YouTube**: Analyze channel analytics, subscriber metrics, video performance, audience engagement, watch time patterns
        • **Instagram**: Evaluate profile metrics, follower engagement, post performance, story analytics, reel interactions

        **CORE RESPONSIBILITIES**:
        1. Handle general queries gracefully while guiding users toward influencer analysis capabilities
        2. Analyze influencer data from Chat History using advanced ranking algorithms
        3. Provide comprehensive performance insights and campaign recommendations
        4. Rank influencers based on multi-factor scoring system
        5. Generate detailed JSON reports with all available metrics
        6. Suggest platform-specific strategies and cross-platform opportunities

        **ANALYSIS METHODOLOGY**:
        
        **Ranking Algorithm (0-100 Scale)**:
        - **Niche Relevance (40%)**: Content alignment with target category, keyword matching, topic consistency
        - **Audience Engagement (30%)**: Engagement rate, comment quality, interaction depth, community responsiveness
        - **Popularity & Reach (30%)**: Follower/subscriber count, average views, content reach, growth trends

        **Data Processing Rules**:
        
        1. **Query Type Classification**:
           - **General Queries**: Handle with helpful responses while naturally directing to core capabilities
           - **Influencer Analysis Queries**: Proceed to Chat History analysis
           - **Capability Questions**: Explain agent features and guide toward influencer analysis
        
        2. **General Query Handling**:
           - Provide concise, helpful answers to general questions
           - Naturally transition to influencer analysis capabilities
           - Use examples to demonstrate how the agent can help with marketing needs
           - Maintain professional, approachable tone
        
        3. **Smooth Transition Strategies**:
           - Connect general marketing questions to influencer analysis benefits
           - Provide context about how the agent can solve specific marketing challenges
           - Offer to analyze influencer data if available
           - Suggest next steps for getting started with influencer analysis
        
        **Chat History Data**: {ChatHistory}
        **User Query**: {Query}
        
        Remember: 
        - Handle all queries gracefully while guiding toward core capabilities
        - For general queries: Be helpful, then naturally transition to influencer analysis benefits
        - For analysis queries: Focus on data analysis and ranking expertise
        - Always maintain conversational flow and professional tone
        - Provide actionable marketing insights when analyzing data
        - Generate comprehensive, professional reports for influencer analysis
        - Support strategic campaign planning decisions
        - Help users discover how influencer analysis can solve their marketing challenges
        """,
        input_variables=['ChatHistory', 'Query']
    )

def should_fetch_influencers(user_input: str, chat_history: List[Dict]) -> bool:
    """Use LLM to determine if we should fetch new influencer data."""
    chat_context = ""
    if chat_history:
        recent_messages = chat_history[-3:]
        chat_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_messages])
    
    prompt = f"""
    Analyze the user input and chat history to determine if new influencer data should be fetched.

    User Input: "{user_input}"
    
    Recent Chat History:
    {chat_context if chat_context else "No previous chat history"}
    
    Rules:
    1. Return "fetch" if the user is asking to find, search, discover, or get new influencers
    2. Return "fetch" if the user is asking for specific types of influencers not in chat history
    3. Return "fetch" if the user wants different criteria (platform, location, niche) than what's in chat history
    4. Return "not fetch" if the user is asking questions about already provided influencer data
    5. Return "not fetch" if the user is asking for analysis, ranking, or insights on existing data
    6. Return "not fetch" for general questions about social media or marketing
    
    Respond with ONLY "fetch" or "not fetch" - nothing else.
    """
    
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.1,
            max_output_tokens=10
        )
        
        response = llm.invoke(prompt)
        decision = response.content.strip().lower()
        
        return decision == "fetch"
        
    except Exception as e:
        print(f"Error in LLM decision making: {str(e)}")
        return True

async def process_influencer_query(input_query, chat_history_dict, user_id):
    """Process a single influencer query with improved logging and multi-platform support."""
    
    # Check for exit command
    exit_commands = ["exit", "quit", "end"]
    if input_query.lower().strip() in exit_commands:
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
            "ranked_channels": [],
            "platform_analyzed": "both"
        }

    # Normal query processing
    chat_history = convert_dict_to_langchain_messages(chat_history_dict)
    chat_history.append(HumanMessage(content=input_query))
    
    # Create prompt template and get response
    prompt_template = create_prompt_template()
    final_prompt = prompt_template.invoke({
        "ChatHistory": convert_langchain_to_dict(chat_history), 
        "Query": input_query
    })
    response = call_llm(final_prompt, chat_history)
    
    print("LLM Response:", response)
    
    chat_history.append(AIMessage(content=response))
    ranked_channels = []
    fetch_attempted = False
    platform = "both"

    # Extract ranked channels from JSON response
    if "```json" in response or "Store_in_Analysis_Database" in response:
        ranked_channels = extract_json_response_to_list(response)
        print("Extracted Ranked Channels:", ranked_channels)

    # Check if we need to fetch new influencer data
    should_fetch = should_fetch_influencers(input_query, chat_history_dict)

    if should_fetch:
        try:
            # Extract parameters using the improved GetThingsFromInput function
            niche, country, max_results, platform = GetThingsFromInput(input_query)
            
            print(f"Extracted - Niche: {niche}, Country: {country}, Max Results: {max_results}, Platform: {platform}")
            
            # Set defaults
            if not country:
                country = "US"
            if not max_results.isdigit() or int(max_results) <= 0:
                max_results = "2"
            if not platform:
                platform = "both"
            
            # Validate niche requirement
            if niche == "general":
                error_msg = "Please specify your input query with niche, country, and max results."
                chat_history.append(AIMessage(content=error_msg))
            else:
                # Determine method based on niche
                if niche == "trending":
                    method = 1
                    print("Fetching trending channels...")
                    result = fetch_channel_with_their_avg_comments(
                        method, 
                        country=country
                    )
                elif niche:
                    method = 2
                    print(f"Fetching channels in niche: {niche}...")
                    result = fetch_channel_with_their_avg_comments(
                        method, 
                        niche=niche, 
                        country=country
                    )
                
                # Parse the fetched data
                if result:
                    new_data = parse_json_to_context_string(result)
                else:
                    # Fallback to reading from file if API fails
                    new_data = read_and_parse_json("./channel_comments.json")
                
                if new_data:
                    # Remove the last AI message and replace with new data
                    chat_history.pop()
                    platform_info = f"Platform: {platform.title()}" if platform != "both" else "Platform: YouTube & Instagram"
                    chat_history.append(AIMessage(content=f"{new_data}\n\n{platform_info}\n\nDo you want me to rank them based on their popularity and engagement metrics?"))
                    fetch_attempted = True
                else:
                    error_msg = "Error: Unable to fetch or parse influencer data."
                    chat_history.append(AIMessage(content=error_msg))
                    
        except Exception as e:
            error_msg = f"Error fetching influencer data: {str(e)}"
            chat_history.append(AIMessage(content=error_msg))
            print(f"Fetch error: {str(e)}")

    # Prepare final response
    updated_chat_history = convert_langchain_to_dict(chat_history)
    final_response = chat_history[-1].content if chat_history else response
    
    return {
        "updated_chat_history": updated_chat_history,
        "response": final_response,
        "database_stored": False,
        "fetch_attempted": fetch_attempted,
        "ranked_channels": ranked_channels,
        "platform_analyzed": platform
    }

# API Endpoints
@app.post('/Social_Media_Influencer_Finder', response_model=InfluencerFinderResponse)
async def social_media_influencer_finder(request: InfluencerFinderRequest):
    """
    Main endpoint for Universal Social Media Influencer Finder chatbot.
    
    Features:
    - Multi-platform support (YouTube, Instagram)
    - AI-powered influencer discovery and ranking
    - Comprehensive performance analytics
    - Campaign strategy recommendations
    - Real-time data fetching and analysis
    """
    try:
        input_query = request.input_query
        chat_history = [msg.dict() for msg in request.ChatHistory]
        user_id = request.user_id

        # Load existing chat history if not provided
        if not chat_history:
            try:
                result = await get_chat_history_for_user(user_id)
                if result != "No chat history found.":
                    chat_history = result
            except Exception as e:
                print(f"Error loading chat history: {str(e)}")
                chat_history = []
        
        print("Chat History:", chat_history)
        
        if not input_query:
            raise HTTPException(status_code=400, detail="input_query is required")
        
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
                
        # Process the query
        result = await process_influencer_query(input_query, chat_history, user_id)

        if result["database_stored"]:
            print("💾 Chat history successfully stored in database")
        
        if result["fetch_attempted"]:
            print("🔍 Data fetch was attempted")

        if result["ranked_channels"]:
            print("📊 Ranked Channels:", json.dumps(result["ranked_channels"], indent=2))
        
        return InfluencerFinderResponse(
            chat_history=[ChatMessage(**msg) for msg in result["updated_chat_history"]],
            response=result["response"],
            fetch_attempted=result["fetch_attempted"],
            ranked_channels=result["ranked_channels"],
            platform_analyzed=result["platform_analyzed"],
            database_stored=result["database_stored"]
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get('/health', response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify API status and capabilities.
    """
    return HealthResponse(
        status="healthy",
        service="Universal Social Media Influencer Finder",
        supported_platforms=["YouTube", "Instagram", "Cross-Platform Analysis"],
        version="1.0.0"
    )

@app.get('/platforms')
async def get_supported_platforms():
    """
    Get detailed information about supported platforms and their capabilities.
    """
    return {
        "supported_platforms": {
            "youtube": {
                "capabilities": [
                    "Channel analytics",
                    "Subscriber metrics",
                    "Video performance tracking",
                    "Audience engagement analysis",
                    "Comment sentiment analysis",
                    "Trending content discovery"
                ],
                "best_for": [
                    "Long-form content campaigns",
                    "Educational content",
                    "Product demonstrations",
                    "Tutorial-based marketing"
                ]
            },
            "instagram": {
                "capabilities": [
                    "Profile analytics",
                    "Follower engagement metrics",
                    "Post performance tracking",
                    "Story analytics",
                    "Reel performance analysis",
                    "Hashtag effectiveness"
                ],
                "best_for": [
                    "Visual content campaigns",
                    "Lifestyle marketing",
                    "Fashion and beauty",
                    "Food and travel content"
                ]
            },
            "cross_platform": {
                "capabilities": [
                    "Unified analytics dashboard",
                    "Cross-platform performance comparison",
                    "Integrated campaign strategies",
                    "Multi-platform audience insights"
                ],
                "best_for": [
                    "Maximum reach campaigns",
                    "Diverse content formats",
                    "Comprehensive brand awareness",
                    "Multi-touchpoint marketing"
                ]
            }
        }
    }

@app.get('/user/{user_id}/history')
async def get_user_chat_history(user_id: str):
    """
    Retrieve chat history for a specific user.
    """
    try:
        result = await get_chat_history_for_user(user_id)
        if result == "No chat history found.":
            return {"message": "No chat history found for this user", "chat_history": []}
        return {"message": "Chat history retrieved successfully", "chat_history": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving chat history: {str(e)}")

# Main application runner
if __name__ == '__main__':
    import uvicorn
    print("🚀 Universal Social Media Influencer Finder API Started!")
    print("=" * 80)
    print("📋 Available endpoints:")
    print("  - POST /Social_Media_Influencer_Finder  (Main influencer finder)")
    print("  - GET  /health                          (Health check)")
    print("  - GET  /platforms                       (Platform capabilities)")
    print("  - GET  /user/{user_id}/history          (User chat history)")
    print("  - GET  /docs                            (API documentation)")
    print("  - GET  /redoc                           (Alternative documentation)")
    print("=" * 80)
    print("🌐 Supported Platforms:")
    print("  ✅ YouTube    - Channel analytics, subscriber metrics, video performance")
    print("  ✅ Instagram  - Profile analytics, engagement metrics, story/reel tracking")
    print("  ✅ Cross-Platform - Unified analysis and campaign strategies")
    print("=" * 80)
    print("🔧 Features:")
    print("  • AI-powered influencer discovery and ranking")
    print("  • Multi-platform performance analytics")
    print("  • Campaign strategy recommendations")
    print("  • Real-time data fetching and analysis")
    print("  • Comprehensive audience insights")
    print("  • Chat history persistence")
    print("=" * 80)
    
    # Use dynamic module name
    module_name = __file__.split('\\')[-1].split('.')[0] if '\\' in __file__ else __file__.split('/')[-1].split('.')[0]
    uvicorn.run(f"{module_name}:app", host="0.0.0.0", port=5000, reload=True)