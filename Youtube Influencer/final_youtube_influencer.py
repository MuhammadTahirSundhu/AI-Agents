
import json
import os
import re
from typing import List

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
from youtube_apis import read_and_parse_json , fetch_channel_with_their_avg_comments , parse_json_to_context_string
from notion_database import create_chat_history, get_chat_history_for_user

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

def call_llm(prompt, chat_history):
    """Call the OpenAI API with GPT-4o model for multi-platform influencer sourcing."""
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
    """
    Extract niche, country, max_results, and platform from user input using a single LLM request.
    
    Args:
        input_query (str): The user's input text
    
    Returns:
        list: List containing [niche, country, max_results, platform]. Each element is a string,
              empty string if not found.
    """
    # Combined prompt for all extraction tasks
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
        # Initialize result list [niche, country, max_results, platform]
        results = ["", "", "", ""]
        
        # Prepare messages for OpenAI API
        messages = [
            {"role": "user", "content": combined_prompt}
        ]
        
        # Get response from LLM
        result = call_common_llm(messages).strip()
        
        # Debug: Print raw LLM response
        print(f"Raw LLM response: {result}")
        
        # Check if response is empty
        if not result:
            print("Error: LLM response is empty")
            return results
        
        # Strip Markdown code fences if present
        cleaned_result = result
        if result.startswith("```json"):
            cleaned_result = result.replace("```json", "").replace("```", "").strip()
        
        # Parse JSON response
        try:
            parsed_result = json.loads(cleaned_result)
        except json.JSONDecodeError as e:
            print(f"Error parsing LLM response as JSON: {str(e)}")
            print(f"Cleaned response: {cleaned_result}")
            return results
        
        # Post-process the results
        # Niche
        niche = parsed_result.get('niche', '')
        if niche:
            results[0] = niche.lower()  # Preserve multi-word phrases
        
        # Country
        results[1] = parsed_result.get('country', '')
        
        # Max_results
        max_results = parsed_result.get('max_results', '')
        if max_results.isdigit():
            results[2] = max_results
        
        # Platform
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
        
        4. **Priority Check for Analysis**: ALWAYS examine Chat History for influencer data
        5. **Data Availability Responses**:
           - If Chat History contains influencer data → Proceed to immediate analysis and ranking
           - If Chat History is empty → Offer to analyze data once it becomes available
        
        6. **Analysis Process**:
           - Extract all available metrics from each influencer profile
           - Calculate ranking scores using the weighted algorithm
           - Sort results by score (highest to lowest)
           - Generate comprehensive JSON output with all details
        
        4. **Quality Assurance**:
           - Verify all data points are from Chat History
           - Ensure no fabricated or assumed information
           - Maintain data integrity throughout analysis
        
        5. **Platform-Specific Insights**:
           - **YouTube**: Focus on watch time, subscriber growth, video consistency, comment engagement
           - **Instagram**: Emphasize visual content quality, story engagement, reel performance, hashtag effectiveness
        
        **COMPREHENSIVE ANALYSIS OUTPUT**:
        When analyzing influencer data, provide detailed JSON-formatted results:
        
        Store_in_Analysis_Database
        ```json
        [
          {{
            "influencer_id": "actual_id_from_chat_history",
            "platform": "platform_from_chat_history",
            "ranking_position": position_number,
            "overall_score": calculated_score_0_to_100,
            
            "profile_analysis": {{
              "channel_name": "actual_channel_name",
              "handle": "actual_handle",
              "description": "actual_description",
              "profile_image_url": "actual_profile_image",
              "verification_status": "actual_verification_status",
              "account_age": "actual_account_age_if_available"
            }},
            
            "performance_metrics": {{
              "subscribers_followers": actual_number,
              "total_content_views": actual_number,
              "content_count": actual_number,
              "engagement_rate_percentage": actual_percentage,
              "average_likes": actual_number,
              "average_comments": actual_number,
              "average_views_per_content": actual_number,
              "growth_trend": "actual_growth_data_if_available"
            }},
            
            "content_strategy": {{
              "primary_niche": "actual_primary_niche",
              "content_categories": ["actual_categories_from_data"],
              "content_format": "actual_content_format",
              "posting_frequency": "actual_posting_frequency",
              "language_primary": "actual_language",
              "content_quality_score": calculated_quality_score
            }},
            
            "recent_content_analysis": [
              {{
                "content_title": "actual_content_title",
                "performance_views": actual_views,
                "engagement_likes": actual_likes,
                "engagement_comments": actual_comments,
                "publish_date": "actual_publish_date",
                "content_type": "actual_content_type"
              }}
            ],
            
            "geographic_data": {{
              "primary_country": "actual_country",
              "primary_city": "actual_city_if_available",
              "timezone": "actual_timezone_if_available",
              "target_regions": "actual_target_regions_if_available"
            }},
            
            "social_presence": {{
              "primary_platform_url": "actual_platform_url",
              "youtube_channel": "actual_youtube_url_if_available",
              "instagram_profile": "actual_instagram_url_if_available",
              "personal_website": "actual_website_if_available",
              "other_platforms": "actual_other_platforms_if_available"
            }},
            
            "audience_intelligence": {{
              "top_audience_comments": [
                {{
                  "comment_text": "actual_comment_text",
                  "comment_likes": actual_likes,
                  "commenter_handle": "actual_commenter_handle"
                }}
              ],
              "engagement_patterns": "actual_engagement_patterns",
              "audience_demographics": "actual_demographics_if_available",
              "community_sentiment": "calculated_sentiment_analysis"
            }},
            
            "collaboration_potential": {{
              "brand_partnership_history": "actual_brand_partnerships",
              "collaboration_rate_estimate": "actual_collaboration_rate",
              "business_contact": "actual_contact_email_if_available",
              "media_kit_availability": "actual_media_kit_url_if_available",
              "campaign_suitability_score": calculated_suitability_score
            }},
            
            "competitive_analysis": {{
              "niche_relevance_score": calculated_niche_score,
              "engagement_quality_score": calculated_engagement_score,
              "popularity_score": calculated_popularity_score,
              "unique_value_proposition": "identified_unique_aspects",
              "campaign_fit_assessment": "calculated_campaign_fit"
            }},
            
            "platform_specific_insights": {{
              "youtube_specific": {{
                "watch_time_metrics": "actual_watch_time_if_available",
                "subscriber_growth_rate": "actual_growth_rate_if_available",
                "video_consistency": "calculated_consistency_score"
              }},
              "instagram_specific": {{
                "story_engagement": "actual_story_metrics_if_available",
                "reel_performance": "actual_reel_metrics_if_available",
                "hashtag_effectiveness": "calculated_hashtag_score"
              }}
            }},
            
            "recommendations": {{
              "campaign_strategy": "tailored_campaign_recommendations",
              "content_collaboration_ideas": "specific_content_suggestions",
              "budget_tier_suggestion": "estimated_budget_tier",
              "best_collaboration_format": "recommended_collaboration_type"
            }},
            
            "data_freshness": {{
              "last_updated": "actual_last_updated_date",
              "data_collection_timestamp": "actual_timestamp",
              "analysis_performed_at": "current_analysis_timestamp"
            }}
          }}
        ]
        ```
        
        **ANALYSIS COMPLETION**:
        - End every successful analysis with "Analysis Complete"
        - Include summary statistics (total analyzed, top performer, average scores)
        - Provide actionable insights for campaign planning
        
        **PLATFORM-SPECIFIC CONSIDERATIONS**:
        - **YouTube**: Prioritize subscriber retention, video completion rates, comment engagement depth
        - **Instagram**: Focus on visual content quality, story completion rates, reel virality potential
        - **Cross-Platform**: Identify opportunities for integrated campaigns and content repurposing
        
        **RESPONSE GUIDELINES**:
        
        **For General Queries**:
        1. Provide helpful, concise answers to general questions
        2. Naturally connect responses to influencer marketing context
        3. Demonstrate agent capabilities through relevant examples
        4. Guide users toward discovering core functionalities
        5. Maintain conversational, professional tone
        
        **For Analysis Queries**:
        1. Always specify which platform(s) are being analyzed
        2. Provide clear ranking explanations with score breakdowns
        3. Include actionable recommendations for each ranked influencer
        4. Highlight standout performers and explain why they scored highest
        5. Suggest optimal collaboration approaches based on data insights
        6. Maintain professional, analytical tone throughout
        
        **Transition Examples**:
        - Marketing Strategy Question → "That's a great marketing approach! For influencer campaigns specifically, I can analyze YouTube and Instagram creators to find the perfect match for your brand. Would you like me to analyze any influencer data you have?"
        - Social Media Question → "Absolutely! Speaking of social media success, I specialize in analyzing influencer performance across YouTube and Instagram. I can help you identify top-performing creators in any niche."
        - Business Question → "That's an important business consideration. In influencer marketing, I can help you make data-driven decisions by analyzing creator metrics, engagement rates, and audience fit. Have you considered working with influencers for your campaign?"
        
        **ERROR HANDLING**:
        - If Chat History contains incomplete data, work with available information and note limitations
        - If data appears corrupted or inconsistent, flag issues and provide best possible analysis
        - Never assume or interpolate missing data points
        
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
    # Prepare chat history context
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
    5. Return "not fetch" if the user is asking questions about already provided influencer data
    6. Return "not fetch" if the user is asking for analysis, ranking, or insights on existing data
    7. Return "not fetch" for general questions about social media or marketing
    
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
        
        # Return True if LLM says "fetch", False otherwise
        return decision == "fetch"
        
    except Exception as e:
        print(f"Error in LLM decision making: {str(e)}")
        # Fallback to safe default - fetch if unsure
        return True
    
async def process_influencer_query(input_query, chat_history_dict, user_id):
    """Process a single influencer query with improved logging and multi-platform support."""
    
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

    # Extract ranked channels from JSON response
    if "```json" in response or "Store_in_Analysis_Database" in response:
        ranked_channels = extract_json_response_to_list(response)
        print("Extracted Ranked Channels:", ranked_channels)

    # Check if we need to fetch new influencer data
    # This logic would need to be enhanced based on your specific fetch trigger patterns
    should_fetch_influencers = should_fetch_influencers(input_query, chat_history)

    if should_fetch_influencers:
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

    # Prepare final response
    updated_chat_history = convert_langchain_to_dict(chat_history)
    final_response = chat_history[-1].content if chat_history else response
    
    return {
        "updated_chat_history": updated_chat_history,
        "response": final_response,
        "database_stored": False,  # No storage until exit
        "fetch_attempted": fetch_attempted,
        "ranked_channels": ranked_channels,
        "platform_analyzed": platform if 'platform' in locals() else "both"
    }

def main():
    """Test the functionality"""
    print("🧪 Testing YouTube Influencer Finder...")

    input_query = input("Enter your query: ")
    # Extract niche, country, and max_results from input
    niche, country, max_results = GetThingsFromInput(input_query)
    print(f"Extracted Niche: {niche}, Country: {country}, Max Results: {max_results}")
    # result = None
    # if country is None or country == "":
    #     country = "US"
    # if niche == "general":
    #     print("Please specify your input query with like niche, country, and max results.")
    # else:
    #     # Validate max_results
    #     if not max_results.isdigit() or int(max_results) <= 0:
    #         max_results = "2"  # Default to 2 if invalid
    #     if niche == "trending":
    #         print("Fetching trending channels...")
    #         # Test method 1 (trending)
    #         result = fetch_channel_with_their_avg_comments(1, int(max_results), country=country)
    #     elif niche:
    #         print(f"Fetching channels in niche: {niche}...")
    #         # Test method 2 (niche-based)
    #         result = fetch_channel_with_their_avg_comments(2, int(max_results), niche=niche, country=country)
    
    # Parse and display results
    context = read_and_parse_json("./channel_comments.json")
    if context:
        print("\n📋 Parsed Context:")
        print(context)

if __name__ == "__main__":
    main()