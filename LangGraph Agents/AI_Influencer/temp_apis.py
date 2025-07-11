import openai
from openai import OpenAI
import requests
import json
import time
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
import os


# Configuration variables
CONFIG = {
    "BASE_URL": "https://yt-api.p.rapidapi.com",
    "API_KEY": "c8a1bcfdc3msha38b3ca9454dd4ap1a3729jsna2e496f34dc8",
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
            "x-rapidapi-key": "c8a1bcfdc3msha38b3ca9454dd4ap1a3729jsna2e496f34dc8",
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
   

def main():
    """Entry point for the script."""
    load_dotenv()
    # results = fetch_channel_with_their_avg_comments()
    fetch_channel_with_their_avg_comments(2,2,"electronics")
    results = read_and_parse_json("./channel_comments.json")
    print(results)
if __name__ == "__main__":
    main()