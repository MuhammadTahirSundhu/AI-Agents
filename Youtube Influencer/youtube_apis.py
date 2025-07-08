import requests
import json
import time
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
import os


# Configuration variables
CONFIG = {
    "BASE_URL": "https://yt-api.p.rapidapi.com",
    "API_KEY": "8e94ef59e9mshf057ac6d1e2ba59p1790a8jsn4a69885f0ae7",
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

def fetch_channel_with_their_avg_comments(method, max_results, niche=None): 
    """
    Main function to fetch channels, their popular videos, and comments.
    
    Args:
        method (int): 1 for Trending Channels, 2 for Popular Channels in Niche
        max_results (int): Maximum number of results to fetch
        niche (str, optional): Required when method=2, specifies the niche to search
    
    Returns:
        dict: Result containing channels and comments data
    
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
    
    with open("channel_comments.json", "w") as f:
        json.dump(result, f, indent=4)
    
    return result

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
    fetch_channel_with_their_avg_comments(2,2,"technology")
    results = read_and_parse_json("./channel_comments.json")
    print(results)
if __name__ == "__main__":
    main()