import time
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import re
import os
import json
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import PromptTemplate
# from notion_database import create_chat_history, get_chat_history_for_user
import requests
# Load environment variables
load_dotenv()



############################################################ API #################################################################
# Configuration variables
CONFIG = {
    "BASE_URL": "https://yt-api.p.rapidapi.com",
    "API_KEY": os.getenv("YOUTUBE_API_KEY"),
    "API_HOST": "yt-api.p.rapidapi.com",
    "DEFAULT_COUNTRY": "US",
    "DEFAULT_LANGUAGE": "en",
    "TRENDING_TYPE": "now",
    "DEFAULT_NICHE": "fitness",
    "RATE_LIMIT_DELAY": 0.5,
    "DEFAULT_MAX_RESULTS": 1
}

# API headers
HEADERS = {
    "x-rapidapi-key": CONFIG["API_KEY"],
    "x-rapidapi-host": CONFIG["API_HOST"]
}

# Cache for channel details
CHANNEL_CACHE: Dict[str, Dict] = {}

def fetch_trending_videos(country: str, type_filter: str, max_results: int) -> List[Dict]:
    """Fetch trending videos for a country using geo and type parameters."""
    videos = []
    token = None
    endpoint = "/trending"
    
    while len(videos) < max_results:
        params = {"geo": country, "type": type_filter}
        if token:
            params["token"] = token
        
        try:
            response = requests.get(f"{CONFIG['BASE_URL']}{endpoint}", headers=HEADERS, params=params)
            response.raise_for_status()
            data = response.json()
            
            for item in data.get("data", []):
                if item.get("type") != "video" or len(videos) >= max_results:
                    continue
                videos.append({
                    "videoId": item.get("videoId", ""),
                    "title": item.get("title", ""),
                    "channelId": item.get("channelId", ""),
                    "channelTitle": item.get("channelTitle", ""),
                    "description": item.get("description", ""),
                    "viewCount": item.get("viewCount", 0),
                    "publishedTimeText": item.get("publishedTimeText", ""),
                    "lengthText": item.get("lengthText", ""),
                    "thumbnail": item.get("thumbnail", []),
                    "channelThumbnail": item.get("channelThumbnail", [])
                })
            
            token = data.get("continuation")
            if not token:
                break
                
        except requests.exceptions.RequestException as e:
            print(f"Error fetching trending videos: {e}")
            break
        
        time.sleep(CONFIG["RATE_LIMIT_DELAY"])
        
    return videos

def fetch_channel_details(channel_id: str, country: str, language: str) -> Optional[Dict]:
    """Fetch channel information using /channel/about endpoint or return from cache."""
    if channel_id in CHANNEL_CACHE:
        return CHANNEL_CACHE[channel_id]
    
    try:
        response = requests.get(
            f"{CONFIG['BASE_URL']}/channel/about",
            headers=HEADERS,
            params={"id": channel_id, "geo": country, "lang": language}
        )
        response.raise_for_status()
        data = response.json()
        
        channel_data = {
            "channelId": channel_id,
            "title": data.get("title", ""),
            "channelHandle": data.get("channelHandle", ""),
            "description": data.get("description", ""),
            "subscriberCount": data.get("subscriberCount", 0),
            "subscriberCountText": data.get("subscriberCountText", ""),
            "videosCount": data.get("videosCount", 0),
            "videosCountText": data.get("videosCountText", ""),
            "viewCount": data.get("viewCount", 0),
            "joinedDate": data.get("joinedDate", ""),
            "country": data.get("country", ""),
            "isVerified": data.get("isVerified", False),
            "isFamilySafe": data.get("isFamilySafe", False),
            "banner": data.get("banner", ""),
            "avatar": data.get("avatar", ""),
            "keywords": data.get("keywords", []),
            "links": data.get("links", []),
            "contactInfo": data.get("email", "Not provided, check About page")
        }
        
        CHANNEL_CACHE[channel_id] = channel_data
        return channel_data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching channel {channel_id}: {e}")
        return None

def method_1_trending_channels(country: str, type_filter: str, max_results: int) -> List[Dict]:
    """Fetch trending channels based on trending videos."""
    videos = fetch_trending_videos(country, type_filter, max_results)
    channel_ids = list(set(video["channelId"] for video in videos))
    channels = []
    
    for channel_id in channel_ids:
        channel_data = fetch_channel_details(channel_id, country, CONFIG["DEFAULT_LANGUAGE"])
        if channel_data:
            channels.append(channel_data)
    
    
    
    return channels

def method_2_popular_channels_in_niche(niche: str, country: str, language: str, max_results: int) -> List[Dict]:
    """Fetch popular channels in a specific niche sorted by views."""
    videos = []
    token = None
    endpoint = "/search"
    
    while len(videos) < max_results:
        params = {
            "query": niche,
            "type": "video",
            "geo": country,
            "sort_by": "views"
        }
        if token:
            params["token"] = token
        
        try:
            response = requests.get(f"{CONFIG['BASE_URL']}{endpoint}", headers=HEADERS, params=params)
            response.raise_for_status()
            data = response.json()
            
            for item in data.get("data", []):
                if len(videos) >= max_results:
                    break
                if item.get("type") == "video":
                    videos.append({
                        "videoId": item.get("videoId", ""),
                        "title": item.get("title", ""),
                        "channelId": item.get("channelId", ""),
                        "channelTitle": item.get("channelTitle", ""),
                        "description": item.get("description", ""),
                        "viewCount": item.get("viewCount", 0),
                        "publishedTimeText": item.get("publishedTimeText", ""),
                        "lengthText": item.get("lengthText", ""),
                        "thumbnail": item.get("thumbnail", []),
                        "channelThumbnail": item.get("channelThumbnail", [])
                    })
            
            token = data.get("continuation")
            if not token:
                break
                
        except requests.exceptions.RequestException as e:
            print(f"Error fetching videos: {e}")
            break
        
        time.sleep(CONFIG["RATE_LIMIT_DELAY"])
    
    
    
    channel_ids = list(set(video["channelId"] for video in videos))
    channels = []
    
    for channel_id in channel_ids:
        channel_data = fetch_channel_details(channel_id, country, language)
        if channel_data:
            channels.append(channel_data)
    
    
    
    return channels

def get_top_popular_videos_of_channel(channel_id: str, max_results: int) -> List[Dict]:
    """Fetch the top popular videos for a given channel ID."""
    videos = []
    token = None
    endpoint = "/channel/videos"
    
    while len(videos) < max_results:
        params = {
            "id": channel_id,
            "sort_by": "popular"
        }
        if token:
            params["token"] = token
        
        try:
            response = requests.get(f"{CONFIG['BASE_URL']}{endpoint}", headers=HEADERS, params=params)
            response.raise_for_status()
            data = response.json()
            
            for item in data.get("data", []):
                if len(videos) >= max_results:
                    break
                if item.get("type") == "video":
                    videos.append({
                        "videoId": item.get("videoId", ""),
                        "title": item.get("title", ""),
                        "channelId": item.get("channelId", ""),
                        "channelTitle": item.get("channelTitle", ""),
                        "description": item.get("description", ""),
                        "viewCount": item.get("viewCount", 0),
                        "publishedTimeText": item.get("publishedTimeText", ""),
                        "lengthText": item.get("lengthText", ""),
                        "thumbnail": item.get("thumbnail", []),
                        "channelThumbnail": item.get("channelThumbnail", [])
                    })
            
            token = data.get("continuation")
            if not token:
                break
                
        except requests.exceptions.RequestException as e:
            print(f"Error fetching videos for channel {channel_id}: {e}")
            break
        
        time.sleep(CONFIG["RATE_LIMIT_DELAY"])
    
    
    
    return videos

def fetch_popular_video_comments(video_id: str, max_results: int, language: str) -> Tuple[List[Dict], str]:
    """Fetch top comments for a video and return comments with formatted string."""
    comments = []
    token = None
    endpoint = "/comments"
    formatted_comments = "Comments:\n"
    
    while len(comments) < max_results:
        params = {
            "id": video_id,
            "sort_by": "top",
            "lang": language
        }
        if token:
            params["token"] = token
        
        try:
            response = requests.get(f"{CONFIG['BASE_URL']}{endpoint}", headers=HEADERS, params=params)
            response.raise_for_status()
            data = response.json()
            
            for item in data.get("data", []):
                if len(comments) >= max_results:
                    break
                comment = {
                    "commentId": item.get("commentId", ""),
                    "author": item.get("authorText", ""),
                    "authorId": item.get("authorChannelId", ""),
                    "authorThumbnail": item.get("authorThumbnail", ""),
                    "text": item.get("textDisplay", ""),
                    "publishedTimeText": item.get("publishedTimeText", ""),
                    "likes": item.get("likesCount", 0),
                    "replyCount": item.get("replyCount", 0),
                    "replyToken": item.get("replyToken", ""),
                    "authorIsChannelOwner": item.get("authorIsChannelOwner", False)
                }
                comments.append(comment)
                formatted_comments += (
                    f"Comment ID: {comment['commentId']}\n"
                    f"Comment Text: {comment['text']}\n"
                    f"Comment Likes: {comment['likes']}\n"
                    f"Comment Publish Time: {comment['publishedTimeText']}\n"
                    f"---------------------------------\n"
                )
            
            token = data.get("continuation")
            if not token:
                break
                
        except requests.exceptions.RequestException as e:
            print(f"Error fetching comments for video {video_id}: {e}")
            break
        
        time.sleep(CONFIG["RATE_LIMIT_DELAY"])
    
    
    
    return comments, formatted_comments

def get_instagram_stats(account_url):
    """
    Fetch Instagram statistics for a given account URL
    
    Args:
        account_url (str): Instagram account URL
        
    Returns:
        dict: Instagram statistics data or None if failed
    """
    try:
        url = "https://instagram-statistics-api.p.rapidapi.com/community"
        
        querystring = {"url": account_url}
        
        headers = {
            "x-rapidapi-key": os.getenv("INSTAGRAM_SearchUser_API_KEY"),
            "x-rapidapi-host": "instagram-statistics-api.p.rapidapi.com"
        }
        
        response = requests.get(url, headers=headers, params=querystring)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Instagram API error for {account_url}: Status {response.status_code}")
            return ""
            
    except Exception as e:
        print(f"❌ Error fetching Instagram data for {account_url}: {str(e)}")
        return ""

def fetch_channel_instagram_info(result):
    """
    Extract Instagram URLs from channel data using LLM and fetch Instagram statistics
    
    Args:
        result (dict): Channel data containing channels and their information
    
    Returns:
        list: List of Instagram URLs
    """
    
    # Prepare the improved prompt for LLM
    prompt = """
    CRITICAL TASK: Extract or generate Instagram URLs for YouTube channels.

    STRICT PRIORITY RULES:
    1. **FIRST PRIORITY**: If a channel has existing Instagram links in socialLinks or description, YOU MUST USE THEM EXACTLY
    2. **SECOND PRIORITY**: Only create new URLs if NO Instagram link exists anywhere
    3. **DETECTION KEYWORDS**: Look for: "instagram.com", "instagram", "insta", "@" followed by username
    4. **SEARCH THOROUGHLY**: Check socialLinks array, description text, and any other fields for Instagram references

    ANALYSIS PROCESS:
    - Step 1: Scan socialLinks array for any Instagram URLs
    - Step 2: Scan description for Instagram mentions or URLs
    - Step 3: If found, extract and use the EXACT URL
    - Step 4: If NOT found, create URL using format: https://instagram.com/[username]

    USERNAME CONVERSION RULES (only if creating new URLs):
    - Convert to lowercase
    - Replace spaces with underscores or remove them
    - Remove special characters except underscores
    - Keep numbers and letters only

    OUTPUT FORMAT:
    - Return ONLY a Python list: ["url1", "url2", "url3"]
    - No explanations, no code blocks, no additional text
    - One URL per channel in the same order as provided

    CHANNELS TO PROCESS:
    """
    
    # Add detailed channel information to the prompt
    for i, channel_data in enumerate(result["channels"]):
        channel = channel_data["channel"]
        prompt += f"\n--- CHANNEL {i+1} ---"
        prompt += f"\nTitle: {channel['title']}"
        prompt += f"\nChannel ID: {channel['channelId']}"
        prompt += f"\nSubscribers: {channel.get('subscriberCountText', 'Unknown')}"
        
        # Emphasize social links section
        if 'socialLinks' in channel and channel['socialLinks']:
            prompt += f"\n🔍 SOCIAL LINKS (CHECK FOR INSTAGRAM): {channel['socialLinks']}"
        else:
            prompt += f"\n🔍 SOCIAL LINKS: None found"
        
        # Include description for Instagram mentions
        if 'description' in channel and channel['description']:
            description_snippet = channel['description'][:300]
            prompt += f"\n📝 DESCRIPTION (CHECK FOR INSTAGRAM): {description_snippet}..."
        else:
            prompt += f"\n📝 DESCRIPTION: None found"
        
        # Add any other relevant fields that might contain Instagram info
        if 'links' in channel and channel['links']:
            prompt += f"\n🔗 OTHER LINKS: {channel['links']}"
        
        prompt += "\n"
    
    prompt += """
    
    REMEMBER: 
    - Use existing Instagram links if found (PRIORITY #1)
    - Create new ones only if none exist
    - Return exactly one URL per channel
    - Format: ["url1", "url2", "url3"]
    """
    
    # Call LLM with improved system message
    messages = [
        {
            "role": "system", 
            "content": """You are an expert Instagram URL detector and generator. Your PRIMARY job is to find existing Instagram links in the provided data. Only create new URLs if absolutely no Instagram presence is found. 

CRITICAL: Always prioritize existing Instagram links over generated ones. Look carefully in socialLinks, description, and any other fields for Instagram references."""
        },
        {
            "role": "user", 
            "content": prompt
        }
    ]
    
    llm_response = call_common_llm(messages)
    print(f"🤖 LLM Response: {llm_response}")
    
    try:
        # Try to evaluate the response as a Python list
        cleaned_response = llm_response.strip()
        
        # Handle potential code block formatting
        if cleaned_response.startswith('```'):
            lines = cleaned_response.split('\n')
            for line in lines:
                if line.strip().startswith('[') and line.strip().endswith(']'):
                    cleaned_response = line.strip()
                    break
        
        if cleaned_response.startswith('[') and cleaned_response.endswith(']'):
            instagram_urls = eval(cleaned_response)
            
            # Validate that we have the right number of URLs
            if len(instagram_urls) != len(result["channels"]):
                print(f"⚠️ Warning: Expected {len(result['channels'])} URLs, got {len(instagram_urls)}")
            
            # Now fetch Instagram data for each URL and append to corresponding channels
            print(f"📱 Fetching Instagram data for {len(instagram_urls)} URLs...")
            
            for i, instagram_url in enumerate(instagram_urls):
                if i < len(result["channels"]):  # Make sure we don't exceed channel count
                    print(f"  🔍 Channel {i+1}: {instagram_url}")
                    
                    # Fetch Instagram statistics
                    instagram_data = get_instagram_stats(instagram_url)
                    
                    # Add Instagram data to the corresponding channel
                    result["channels"][i]["instagram_url"] = instagram_url
                    result["channels"][i]["instagram_data"] = instagram_data
                    
                    if instagram_data or instagram_data == "":
                        print(f"    ✅ Successfully fetched Instagram data")
                    else:
                        print(f"    ❌ Failed to fetch Instagram data")
                else:
                    print(f"⚠️ Warning: More Instagram URLs than channels. Skipping {instagram_url}")
            
            return instagram_urls
        else:
            print(f"❌ LLM response is not a proper list format: {cleaned_response}")
            return []
    except Exception as e:
        print(f"❌ Error parsing LLM response: {str(e)}")
        print(f"LLM Response: {llm_response}")
        return []

def fetch_channel_with_their_avg_comments(method, max_results, niche=None): 
    """
    Main function to fetch channels, their popular videos, and comments.
    
    Args:
        method (int): 1 for Trending Channels, 2 for Popular Channels in Niche
        max_results (int): Maximum number of results to fetch
        niche (str, optional): Required when method=2, specifies the niche to search
        platform (str): Platform type - "Youtube" or "instagram"
    
    Returns:
        dict: Result containing channels and comments data, plus Instagram URLs if platform is instagram
    
    Raises:
        ValueError: If invalid method is provided or niche is missing for method 2
    """
    # Validate method parameter
    if method not in [1, 2]:
        raise ValueError("Method must be 1 (Trending Channels) or 2 (Popular Channels in Niche)")
    
    # Validate niche parameter for method 2
    if method == 2 and not niche:
        raise ValueError("Niche parameter is required when method=2")
    
    channels = []
    
    if method == 1:
        print("Running Method 1: Trending Channels")
        channels = method_1_trending_channels(
            country=CONFIG["DEFAULT_COUNTRY"],
            type_filter=CONFIG["TRENDING_TYPE"],
            max_results=max_results
        )
    elif method == 2:
        print(f"Running Method 2: Popular Channels in Niche - '{niche}'")
        channels = method_2_popular_channels_in_niche(
            niche=niche,
            country=CONFIG["DEFAULT_COUNTRY"],
            language=CONFIG["DEFAULT_LANGUAGE"],
            max_results=max_results
        )
    
    print(f"Found {len(channels)} channels:")
    for channel in channels:
        print(f"- {channel['title']} ({channel['subscriberCountText']} subscribers)")
    
    result = {
        "channels": [],
        "comments_strings": []
    }
    
    for channel in channels:
        channel_data = {
            "channel": channel,
            "videos": []
        }
        
        popular_videos = get_top_popular_videos_of_channel(channel["channelId"], max_results)
        comments_for_channel = ""
        
        for pop_vid in popular_videos:
            comments, comments_string = fetch_popular_video_comments(
                pop_vid["videoId"],
                max_results,
                CONFIG["DEFAULT_LANGUAGE"]
            )
            video_data = {
                "video": pop_vid,
                "comments": comments,
                "comments_string": comments_string
            }
            channel_data["videos"].append(video_data)
            comments_for_channel += f"Video: {pop_vid['title']}\n{comments_string}\n"
        
        result["channels"].append(channel_data)
        result["comments_strings"].append(comments_for_channel)

    # Handle Instagram platform
    print("🔍 Fetching Instagram URLs and data...")
    instagram_urls = fetch_channel_instagram_info(result)
    result["instagram_urls"] = instagram_urls
    print(f"✅ Generated {len(instagram_urls)} Instagram URLs")
        
        # # Display summary of Instagram data
        # print("\n📊 Instagram Data Summary:")
        # for i, channel_data in enumerate(result["channels"]):
        #     channel_name = channel_data["channel"]["title"]
        #     instagram_url = channel_data.get("instagram_url", "Not found")
        #     instagram_data = channel_data.get("instagram_data")
            
        #     print(f"  {i+1}. {channel_name}")
        #     print(f"     Instagram URL: {instagram_url}")
            
        #     if instagram_data:
        #         # Extract key metrics if available
        #         if isinstance(instagram_data, dict):
        #             followers = instagram_data.get('followers', 'Unknown')
        #             following = instagram_data.get('following', 'Unknown')
        #             posts = instagram_data.get('posts', 'Unknown')
        #             print(f"     Followers: {followers} | Following: {following} | Posts: {posts}")
        #         else:
        #             print(f"     Status: Data fetched successfully")
        #     else:
        #         print(f"     Status: ❌ Failed to fetch Instagram data")
        #     print()
    
    # Save results to file
    with open("channel_comments.json", "w") as f:
        json.dump(result, f, indent=4)
    
    return result

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

def parse_json_to_context_string(json_data: dict) -> str:
    """
    Parse JSON data containing channel, video, and comment information into a single string for LLM context.
    
    Args:
        json_data: Dictionary containing channels and comments_strings data.
    
    Returns:
        A formatted string with channel details, video details, and comments.
    """
    context = f"TOP Influencer's Channels, Their TOP Popular Vedios, Each Vedio's Top Comments \n\n"
    
    # Iterate through each channel
    for channel_data in json_data.get("channels", []):
        channel = channel_data.get("channel", {})
        
        # Add channel details
        context += f"Channel: {channel.get('title', 'Unknown')}\n"
        context += f"Channel ID: {channel.get('channelId', 'Unknown')}\n"
        context += f"Handle: {channel.get('channelHandle', 'Unknown')}\n"
        context += f"Description: {channel.get('description', 'No description')}\n"
        context += f"Subscribers: {channel.get('subscriberCountText', 'Unknown')}\n"
        context += f"Videos Count: {channel.get('videosCountText', 'Unknown')}\n"
        context += f"Total Views: {channel.get('viewCount', 'Unknown')}\n"
        context += f"Joined Date: {channel.get('joinedDate', 'Unknown')}\n"
        context += f"Country: {channel.get('country', 'Unknown')}\n"
        context += f"Is Verified: {channel.get('isVerified', False)}\n"
        context += f"Is Family Safe: {channel.get('isFamilySafe', False)}\n"
        
        # Add links if available
        links = channel.get("links", [])
        if links:
            context += "Links:\n"
            for link in links:
                context += f"  - {link.get('title', 'Unknown')}: {link.get('link', 'Unknown')}\n"
        context += "\n"

        # Add Instagram data if available
        instagram_data = channel_data.get("instagram_data", {}).get("data", {})
        if instagram_data:
            context += "\nInstagram Data:\n"
            context += f"  Instagram URL: {instagram_data.get('url', 'Unknown')}\n"
            context += f"  Name: {instagram_data.get('name', 'Unknown')}\n"
            context += f"  Screen Name: {instagram_data.get('screenName', 'Unknown')}\n"
            context += f"  Description: {instagram_data.get('description', 'No description')}\n"
            context += f"  Followers: {instagram_data.get('usersCount', 'Unknown')}\n"
            context += f"  Verified: {instagram_data.get('verified', False)}\n"
            context += f"  Average Engagement Rate: {instagram_data.get('avgER', 'Unknown')}\n"
            context += f"  Average Likes: {instagram_data.get('avgLikes', 'Unknown')}\n"
            context += f"  Average Comments: {instagram_data.get('avgComments', 'Unknown')}\n"
            context += f"  Quality Score: {instagram_data.get('qualityScore', 'Unknown')}\n"
            context += f"  Fake Followers Percentage: {instagram_data.get('pctFakeFollowers', 'Unknown')}\n"
            context += f"  Audience Severity: {instagram_data.get('audienceSeverity', 'Unknown')}\n"
            context += f"  Contact Email: {instagram_data.get('contactEmail', 'Not provided')}\n"
            
            # Add Instagram tags
            tags = instagram_data.get("tags", [])
            if tags:
                context += f"  Tags: {', '.join(tags)}\n"
            
            # Add top cities
            cities = instagram_data.get("membersCities", [])[:5]  # Limit to top 5
            if cities:
                context += "  Top Cities:\n"
                for city in cities:
                    context += f"    - {city.get('name', 'Unknown')}: {city.get('value', 'Unknown'):.2%}\n"
            
            # Add top countries
            countries = instagram_data.get("membersCountries", [])[:5]  # Limit to top 5
            if countries:
                context += "  Top Countries:\n"
                for country in countries:
                    context += f"    - {country.get('name', 'Unknown')}: {country.get('value', 'Unknown'):.2%}\n"
            
            # Add gender and age distribution
            gender_age = instagram_data.get("membersGendersAges", {}).get("summary", {})
            if gender_age:
                context += "  Gender Distribution:\n"
                context += f"    - Male: {gender_age.get('m', 'Unknown'):.2%}\n"
                context += f"    - Female: {gender_age.get('f', 'Unknown'):.2%}\n"
                context += f"    - Average Age Group: {gender_age.get('avgAges', 'Unknown')}\n"
            
            # Add latest Instagram posts
            last_posts = instagram_data.get("lastPosts", [])[:3]  # Limit to 3 posts
            if last_posts:
                context += "  Latest Instagram Posts:\n"
                for post in last_posts:
                    context += f"    - URL: {post.get('url', 'Unknown')}\n"
                    context += f"      Date: {post.get('date', 'Unknown')}\n"
                    context += f"      Type: {post.get('type', 'Unknown')}\n"
                    context += f"      Likes: {post.get('likes', 'Unknown')}\n"
                    context += f"      Comments: {post.get('comments', 'Unknown')}\n"
                    context += f"      Text: {post.get('text', 'No text')}\n"
                    context += f"      ---------------------------------\n"
        
        context += "\n"
        
        
        # Add video details and comments
        context += "Videos:\n"
        for video_data in channel_data.get("videos", []):
            video = video_data.get("video", {})
            context += f"  Video Title: {video.get('title', 'Unknown')}\n"
            context += f"  Video ID: {video.get('videoId', 'Unknown')}\n"
            context += f"  Description: {video.get('description', 'No description')}\n"
            context += f"  View Count: {video.get('viewCount', 'Unknown')}\n"
            context += f"  Published: {video.get('publishedTimeText', 'Unknown')}\n"
            context += f"  Length: {video.get('lengthText', 'Unknown')}\n"
            context += f"  Comments:\n"
            
            # Add comments for the video
            for comment in video_data.get("comments", []):
                context += f"    Comment ID: {comment.get('commentId', 'Unknown')}\n"
                context += f"    Author: {comment.get('author', 'Unknown')}\n"
                context += f"    Text: {comment.get('text', 'No text')}\n"
                context += f"    Likes: {comment.get('likes', 'Unknown')}\n"
                context += f"    Published: {comment.get('publishedTimeText', 'Unknown')}\n"
                context += f"    ---------------------------------\n"
            context += "\n"
        
        context += "=" * 50 + "\n\n"
    
    return context

def read_and_parse_json(file_path: str = "result.json") -> Optional[str]:
    """
    Read JSON data from a file and pass it to parse_json_to_context_string to generate a formatted string.
    
    Args:
        file_path: Path to the JSON file (default: 'result.json').
    
    Returns:
        A formatted string with parsed JSON data, or None if an error occurs.
    """
    try:
        # Read the JSON file
        with open(file_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
        
        # Pass the JSON data to the parsing function
        return parse_json_to_context_string(json_data)
    
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' contains invalid JSON.")
        return None
    except Exception as e:
        print(f"Error: An unexpected error occurred while processing '{file_path}': {str(e)}")
        return None
   

############################################################ API #################################################################


app = FastAPI(title="YouTube Influencer Finder API", version="1.0.0")

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
        You are an ethical Social Influencer Sourcing Agent which can fetch influencers from the Youtube and instagram. Your task is to interpret user queries, identify the appropriate function to call, and handle general questions with moral, responsible answers. You have access to one function for fetching influencer data. Follow these guidelines strictly:

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
              - If you ask you about what you can do or how you work, explain your role with the platforms you can handle for influencers
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
           
            8. **Ranked Result Format**:
              When ranking channels from Chat History data, extract and show these details in markdown format:
              Store_in_Notion_database  
              ```json
              [
                {{
                    **Youtube Channel Details**
                  "Channel Name": "actual_channel_name_from_chat_history",
                  "Handle": "actual_handle_from_chat_history",
                  "Description": "actual_description_from_chat_history",
                  "Subscribers": actual_subscriber_count_number, 
                  "Total Views": actual_total_views_number, 
                  "Videos Count": actual_video_count_number,
                  "Country": "actual_country_from_chat_history",
                  "Ranking Score": "actual_ranking_score_number",

                   **Instagram Details**
                            -"\nInstagram Data:\n"
                            -  Instagram URL: "actual_from_chatHistory"\n"
                            -  Name: "actual_from_chatHistory"\n"
                            -  Screen Name: "actual_from_chatHistory"\n"
                            -  Description: "actual_from_chatHistory"\n"
                            -  Followers: "actual_from_chatHistory"\n"
                            -  Verified: "actual_from_chatHistory"\n"
                            -  Average Engagement Rate: "actual_from_chatHistory"\n"
                            -  Average Likes: "actual_from_chatHistory"\n"
                            -  Average Comments: "actual_from_chatHistory"\n"
                            -  Quality Score: "actual_from_chatHistory"\n"
                            -  Fake Followers Percentage: "actual_from_chatHistory"\n"
                            -  Audience Severity: "actual_from_chatHistory"\n"
                            -  Contact Email: "actual_from_chatHistory"\n"
                }}
              ]
              ```

        **Response Logic**:
        1. If the user query is 'exit', 'quit', or 'end', respond with 'Session ended. Chat history saved.' and do not process further
        2. Check if Chat History contains influencer data
        3. If YES: Immediately rank the available data and provide markdown output
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
            # chat_history_id = await create_chat_history(
            #     input_query=chat_history_dict[0]["content"] if chat_history_dict else "Help me to find influencers",
            #     user_id=user_id,
            #     chat_history=convert_langchain_to_dict(chat_history),
            #     response="Session ended. Chat history saved."
            # )
            # database_stored = bool(chat_history_id)
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
                    # result = fetch_channel_with_their_avg_comments(method, max_results, niche)
                    # new_data = parse_json_to_context_string(result)
                    # for getting data from file
                    new_data = read_and_parse_json("./channel_comments.json")
                    chat_history.pop()  # Remove last AIMessage
                    chat_history.append(AIMessage(content=f"{new_data} \n\n DO you want me to rank them on basis of their popularity? or Want the Instagram influencers only?"))
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

        # if not chat_history:
        #     result = await get_chat_history_for_user(user_id)
        #     if result != "No chat history found.":
        #         chat_history = result
        
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