import requests
import json
import time
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configuration
CONFIG = {
    "BASE_URL": "https://yt-api.p.rapidapi.com",
    "API_KEY": "c6648bc73fmsh29bec8d170fe1d3p12e4c4jsn516ec2c19f50",
    "API_HOST": "yt-api.p.rapidapi.com",
    "DEFAULT_COUNTRY": "US",
    "DEFAULT_LANGUAGE": "en",
    "TRENDING_TYPE": "now",
    "RATE_LIMIT_DELAY": 0.5
}

# API headers
HEADERS = {
    "x-rapidapi-key": CONFIG["API_KEY"],
    "x-rapidapi-host": CONFIG["API_HOST"]
}

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def call_llm(messages: List[dict], temperature: float = 0.3) -> str:
    """Make a call to OpenAI API"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ OpenAI API error: {str(e)}")
        return ""

def fetch_trending_videos(country: str, type_filter: str) -> List[Dict]:
    """Fetch trending videos for a country"""
    videos = []
    endpoint = "/trending"
    
    print(f"🔍 Fetching trending videos from {country}...")
    
    params = {"geo": country, "type": type_filter}
    
    try:
        response = requests.get(f"{CONFIG['BASE_URL']}{endpoint}", headers=HEADERS, params=params)
        response.raise_for_status()
        data = response.json()
        
        for item in data.get("data", []):
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
        
        print(f"✅ Fetched {len(videos)} trending videos")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching trending videos: {e}")
    
    return videos

def fetch_niche_videos(niche: str, country: str, language: str) -> List[Dict]:
    """Fetch videos for a specific niche"""
    videos = []
    endpoint = "/search"
    
    print(f"🔍 Searching for '{niche}' videos in {country}...")
    
    params = {
        "query": niche,
        "type": "video",
        "geo": country,
        "sort_by": "views"
    }
    
    try:
        response = requests.get(f"{CONFIG['BASE_URL']}{endpoint}", headers=HEADERS, params=params)
        response.raise_for_status()
        data = response.json()
        
        for item in data.get("data", []):
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
        
        print(f"✅ Fetched {len(videos)} niche videos")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching niche videos: {e}")
    
    return videos

def fetch_channel_details(channel_id: str, country: str, language: str) -> Optional[Dict]:
    """Fetch detailed channel information"""
    try:
        response = requests.get(
            f"{CONFIG['BASE_URL']}/channel/about",
            headers=HEADERS,
            params={"id": channel_id, "geo": country, "lang": language}
        )
        response.raise_for_status()
        data = response.json()
        
        return {
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
            "contactInfo": data.get("email", "Not provided")
        }
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching channel {channel_id}: {e}")
        return None

def normalize_country_to_full_name(country: str) -> str:
    """Use LLM to convert country input to full country name"""
    if not country or country.strip() == "":
        return ""
    
    prompt = (
        f"Convert the country input '{country}' to its full country name (e.g., 'US' to 'United States'). "
        "If it's already a full country name, return it unchanged. "
        "If the input is invalid or cannot be recognized, return the original input unchanged. "
        "Return only the full country name or the original input, nothing else."
    )
    messages = [{"role": "user", "content": prompt}]
    
    result = call_llm(messages, temperature=0.3)
    return result.strip() if result else country

def filter_channels_by_country(channels: List[Dict], target_country: str) -> List[Dict]:
    """Filter channels by country with fallback to global channels"""
    target_country_normalized = normalize_country_to_full_name(target_country)

    # Filter exact matches
    exact_matches = [
        ch for ch in channels 
        if ch.get("country") and ch.get("country").upper() == target_country_normalized.upper()
    ]

    if len(exact_matches) >= 3:
        print(f"✅ Found {len(exact_matches)} channels from {target_country_normalized}")
        return exact_matches

    # Include global/international channels
    global_channels = [
        ch for ch in channels 
        if not ch.get("country") or ch.get("country").strip() == ""
    ]

    print(f"⚠️ Only {len(exact_matches)} channels from {target_country_normalized}, including {len(global_channels)} global channels")
    
    return exact_matches + global_channels

def llm_select_best_channels(channels: List[Dict], method: int, niche: str = None, target_country: str = None) -> List[Dict]:
    """Use LLM to select the best channels for influencer marketing"""
    # Filter by country first
    # if target_country:
    #     channels = filter_channels_by_country(channels, target_country)
    
    target_count = min(10, len(channels))
    
    if len(channels) <= target_count:
        return channels
    
    # Prepare channel data for LLM
    channel_summary = []
    for i, channel in enumerate(channels):
        description = channel.get("description", "")
        truncated_desc = description[:200] + "..." if len(description) > 200 else description
        
        channel_summary.append({
            "index": i,
            "title": channel.get("title", ""),
            "subscriberCount": channel.get("subscriberCount", 0),
            "subscriberCountText": channel.get("subscriberCountText", ""),
            "viewCount": channel.get("viewCount", 0),
            "description": truncated_desc,
            "isVerified": channel.get("isVerified", False),
            "country": channel.get("country", "Unknown"),
            "videosCount": channel.get("videosCount", 0)
        })
    
    country_preference = f"HIGHLY PRIORITIZE channels from {target_country} or region-relevant channels" if target_country else ""
    
    system_prompt = f"""You are an expert YouTube influencer marketing analyst. 
    
    Your task is to select the top {target_count} channels from the provided list that have the highest potential for influencer marketing.
    
    Selection Criteria:
    1. **Geographic Relevance**: {country_preference}
    2. **Subscriber Count & Quality**: Prefer channels with substantial but engaged audiences
    3. **Engagement Potential**: Look for channels with good video output and likely high engagement
    4. **Content Relevance**: {'For niche "' + niche + '", prioritize channels relevant to this topic' if niche else 'For trending content, prioritize popular and viral channels'}
    5. **Authenticity**: Prefer verified channels and those with authentic content
    6. **Marketing Potential**: Consider channels suitable for brand partnerships
    7. **Keyword Relevance(MOST Important Factor and Higly Prioritize)**: Include channels which contains specific keywords or phrases relevant to the campaign
    
    Method: {method} ({'Trending Channels' if method == 1 else f'Niche-based Channels - {niche}'})
    
    Return ONLY a JSON array of the selected channel indices (0-based):
    [0, 5, 12, 8, 3, ...]
    
    Select exactly {target_count} channels or fewer if the list is smaller."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Channel Data:\n{json.dumps(channel_summary, indent=2)}"}
    ]
    
    response = call_llm(messages)
    
    try:
        # Extract JSON array from response
        import re
        json_match = re.search(r'\[[\d,\s]*\]', response)
        if json_match:
            selected_indices = json.loads(json_match.group())
            selected_channels = [channels[i] for i in selected_indices if 0 <= i < len(channels)]
            print(f"🤖 LLM selected {len(selected_channels)} channels from {len(channels)} total")
            return selected_channels
    except Exception as e:
        print(f"❌ Error parsing LLM response: {e}")
    
    # Fallback: return top channels by subscriber count
    sorted_channels = sorted(channels, key=lambda x: x.get("subscriberCount", 0), reverse=True)
    return sorted_channels[:target_count]

def get_top_popular_videos_of_channel(channel_id: str, max_results: int) -> List[Dict]:
    """Fetch top popular videos for a channel"""
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
            print(f"❌ Error fetching videos for channel {channel_id}: {e}")
            break
        
        time.sleep(CONFIG["RATE_LIMIT_DELAY"])
    
    return videos

def fetch_video_comments(video_id: str, max_results: int, language: str) -> List[Dict]:
    """Fetch top comments for a video"""
    comments = []
    token = None
    endpoint = "/comments"
    
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
                comments.append({
                    "commentId": item.get("commentId", ""),
                    "author": item.get("authorText", ""),
                    "authorId": item.get("authorChannelId", ""),
                    "text": item.get("textDisplay", ""),
                    "publishedTimeText": item.get("publishedTimeText", ""),
                    "likes": item.get("likesCount", 0),
                    "replyCount": item.get("replyCount", 0),
                    "authorIsChannelOwner": item.get("authorIsChannelOwner", False)
                })
            
            token = data.get("continuation")
            if not token:
                break
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching comments for video {video_id}: {e}")
            break
        
        time.sleep(CONFIG["RATE_LIMIT_DELAY"])
    
    return comments

def format_comments_string(comments: List[Dict], video_title: str) -> str:
    """Format comments into a readable string"""
    if not comments:
        return f"Video: {video_title}\nNo comments available.\n"
    
    formatted = f"Video: {video_title}\nComments:\n"
    for comment in comments:
        formatted += (
            f"Comment ID: {comment['commentId']}\n"
            f"Author: {comment['author']}\n"
            f"Text: {comment['text']}\n"
            f"Likes: {comment['likes']}\n"
            f"Published: {comment['publishedTimeText']}\n"
            f"---------------------------------\n"
        )
    return formatted

def fetch_channel_with_their_avg_comments(method: int, niche: str = None, country: str = CONFIG["DEFAULT_COUNTRY"]) -> dict:
    """
    Main function to fetch channels with LLM-based intelligent selection
    
    Args:
        method (int): 1 for Trending Channels, 2 for Niche-based Channels
        max_results (int): Final number of channels to return
        niche (str, optional): Required for method=2
        country (str): Target country code
    
    Returns:
        dict: Result containing selected channels with videos and comments
    """
    # Validation
    if method not in [1, 2]:
        raise ValueError("Method must be 1 (Trending) or 2 (Niche-based)")
    
    if method == 2 and not niche:
        raise ValueError("Niche parameter required for method=2")
    
    print(f"🚀 Starting Method {method}: {'Trending Channels' if method == 1 else f'Niche-based ({niche})'}")
    print(f"🌍 Country: {country}")
    
    # Fetch videos from API
    if method == 1:
        videos = fetch_trending_videos(country, CONFIG["TRENDING_TYPE"])
    else:
        videos = fetch_niche_videos(niche, country, CONFIG["DEFAULT_LANGUAGE"])
    
    # Extract unique channels
    unique_channels = {}
    for video in videos:
        channel_id = video["channelId"]
        if channel_id not in unique_channels:
            unique_channels[channel_id] = video["channelTitle"]
    
    print(f"📊 Found {len(unique_channels)} unique channels")
    
    # Fetch detailed channel information
    channels = []
    for channel_id, channel_title in unique_channels.items():
        channel_data = fetch_channel_details(channel_id, country, CONFIG["DEFAULT_LANGUAGE"])
        if channel_data:
            channels.append(channel_data)
        time.sleep(CONFIG["RATE_LIMIT_DELAY"])
    
    print(f"✅ Retrieved details for {len(channels)} channels")
    
    # Debug channel information
    for channel in channels:
        print(f"Channel: {channel['title']} | Subscribers: {channel['subscriberCountText']} | Country: {channel['country']} | Verified: {channel['isVerified']}")
    
    # LLM selects best channels
    selected_channels = llm_select_best_channels(channels, method, niche, country)
    
    # Fetch videos and comments for selected channels
    channels_with_data = []
    for channel in selected_channels:
        print(f"🎬 Processing videos for: {channel['title']}")
        
        channel_data = {
            "channel": channel,
            "videos": []
        }
        
        # Get popular videos
        popular_videos = get_top_popular_videos_of_channel(channel["channelId"], 3)
        
        # Get comments for each video
        for video in popular_videos:
            comments = fetch_video_comments(video["videoId"], 3, CONFIG["DEFAULT_LANGUAGE"])
            channel_data["videos"].append({
                "video": video,
                "comments": comments,
                "comments_string": format_comments_string(comments, video["title"])
            })
        
        channels_with_data.append(channel_data)
        time.sleep(CONFIG["RATE_LIMIT_DELAY"])
    
    # Prepare result
    result = {
        "channels": channels_with_data,
        "comments_strings": [
            "\n".join(video["comments_string"] for video in channel["videos"])
            for channel in channels_with_data
        ]
    }
    
    # Save to file
    with open("channel_comments.json", "w") as f:
        json.dump(result, f, indent=4)
    
    print(f"🎉 Successfully processed {len(channels_with_data)} top influencer channels")
    return result

def parse_json_to_context_string(json_data: dict) -> str:
    """Parse JSON data into a formatted context string for LLM"""
    context = "TOP YouTube Influencer Channels Analysis\n"
    context += "=" * 50 + "\n\n"
    
    for channel_data in json_data.get("channels", []):
        channel = channel_data.get("channel", {})
        
        # Channel information
        context += f"Channel: {channel.get('title', 'Unknown')}\n"
        context += f"Handle: {channel.get('channelHandle', 'Unknown')}\n"
        context += f"Subscribers: {channel.get('subscriberCountText', 'Unknown')}\n"
        context += f"Total Views: {channel.get('viewCount', 'Unknown')}\n"
        context += f"Videos Count: {channel.get('videosCountText', 'Unknown')}\n"
        context += f"Country: {channel.get('country', 'Unknown')}\n"
        context += f"Verified: {channel.get('isVerified', False)}\n"
        context += f"Description: {channel.get('description', 'No description')}\n\n"
        
        # Videos and comments
        context += "Popular Videos:\n"
        for video_data in channel_data.get("videos", []):
            video = video_data.get("video", {})
            context += f"  📹 {video.get('title', 'Unknown')}\n"
            context += f"     Views: {video.get('viewCount', 'Unknown')}\n"
            context += f"     Published: {video.get('publishedTimeText', 'Unknown')}\n"
            
            context += f"     Top Comments:\n"
            for comment in video_data.get("comments", []):
                context += f"       • {comment.get('text', 'No text')[:100]}...\n"
                context += f"         👍 {comment.get('likes', 0)} likes\n"
            context += "\n"
        
        context += "=" * 50 + "\n\n"
    
    return context

def read_and_parse_json(file_path: str = "channel_comments.json") -> Optional[str]:
    """Read and parse JSON file to context string"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
        return parse_json_to_context_string(json_data)
    except FileNotFoundError:
        print(f"❌ File '{file_path}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON in '{file_path}'.")
        return None
    except Exception as e:
        print(f"❌ Error processing '{file_path}': {str(e)}")
        return None