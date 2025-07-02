import json
import re
import os
import asyncio
from typing import List, Dict, Any, Union
from notion_client import AsyncClient
from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage

# Initialize Notion client
notion = AsyncClient(auth=os.getenv("NOTION_API_TOKEN"))

def convert_dict_to_langchain_messages(chat_history_dict: List[Dict[str, str]]) -> List[BaseMessage]:
    """
    Convert dictionary format chat history to LangChain message objects.
    
    Args:
        chat_history_dict: List of dictionaries with 'role' and 'content' keys
        
    Returns:
        List of LangChain message objects
    """
    messages = []
    
    for msg in chat_history_dict:
        role = msg.get("role", "").lower()
        content = msg.get("content", "")
        
        if role in ["user", "human"]:
            messages.append(HumanMessage(content=content))
        elif role in ["assistant", "ai"]:
            messages.append(AIMessage(content=content))
        elif role == "system":
            messages.append(SystemMessage(content=content))
        else:
            # Handle unknown roles by defaulting to HumanMessage
            print(f"Warning: Unknown role '{role}', treating as human message")
            messages.append(HumanMessage(content=content))
    
    return messages

def extract_channel_info(content: str) -> List[Dict[str, Any]]:
    """Extract channel information from AI response content."""
    channels = []
    
    # Pattern to match channel blocks
    channel_pattern = r"Channel: ([^\n]+)\n.*?Channel ID: ([^\n]+)\n.*?Handle: ([^\n]+)\n.*?Description: ([^\n]+)\n.*?Subscribers: ([^\n]+)\n.*?Videos Count: ([^\n]+)\n.*?Total Views: ([^\n]+)\n.*?Joined Date: ([^\n]+)\n.*?Country: ([^\n]+)"
    
    channel_matches = re.findall(channel_pattern, content, re.DOTALL)
    
    for match in channel_matches:
        channel_info = {
            "name": match[0].strip(),
            "channel_id": match[1].strip(),
            "handle": match[2].strip(),
            "description": match[3].strip()[:200] + "..." if len(match[3]) > 200 else match[3].strip(),
            "subscribers": match[4].strip(),
            "video_count": match[5].strip(),
            "total_views": match[6].strip(),
            "joined_date": match[7].strip(),
            "country": match[8].strip()
        }
        
        # Extract videos for this channel
        videos = extract_videos_for_channel(content, match[0])
        channel_info["videos"] = videos
        
        channels.append(channel_info)
    
    return channels

def extract_videos_for_channel(content: str, channel_name: str) -> List[Dict[str, Any]]:
    """Extract video information for a specific channel."""
    videos = []
    
    # Find the channel section
    channel_start = content.find(f"Channel: {channel_name}")
    if channel_start == -1:
        return videos
    
    # Find the next channel or end of content
    next_channel = content.find("Channel:", channel_start + 1)
    channel_section = content[channel_start:next_channel] if next_channel != -1 else content[channel_start:]
    
    # Pattern to match videos within the channel section
    video_pattern = r"Video Title: ([^\n]+)\n.*?Video ID: ([^\n]+)\n.*?Description: ([^\n]+)\n.*?View Count: (\d+)\n.*?Published: ([^\n]+)\n.*?Length: ([^\n]+)"
    
    video_matches = re.findall(video_pattern, channel_section, re.DOTALL)
    
    for match in video_matches:
        video_info = {
            "title": match[0].strip(),
            "video_id": match[1].strip(),
            "description": match[2].strip()[:150] + "..." if len(match[2]) > 150 else match[2].strip(),
            "view_count": int(match[3]),
            "published": match[4].strip(),
            "length": match[5].strip()
        }
        
        # Extract comments for this video
        comments = extract_comments_for_video(channel_section, match[0])
        video_info["comments"] = comments
        
        videos.append(video_info)
    
    return videos

def extract_comments_for_video(content: str, video_title: str) -> List[Dict[str, Any]]:
    """Extract comments for a specific video."""
    comments = []
    
    # Find the video section
    video_start = content.find(f"Video Title: {video_title}")
    if video_start == -1:
        return comments
    
    # Find the next video or end of section
    next_video = content.find("Video Title:", video_start + 1)
    video_section = content[video_start:next_video] if next_video != -1 else content[video_start:]
    
    # Pattern to match comments
    comment_pattern = r"Comment ID: ([^\n]+)\n.*?Author: ([^\n]+)\n.*?Text: ([^\n]+)\n.*?Likes: ([^\n]+)\n.*?Published: ([^\n]+)"
    
    comment_matches = re.findall(comment_pattern, video_section, re.DOTALL)
    
    for match in comment_matches:
        comment_info = {
            "comment_id": match[0].strip(),
            "author": match[1].strip(),
            "text": match[2].strip(),
            "likes": match[3].strip(),
            "published": match[4].strip()
        }
        comments.append(comment_info)
    
    return comments

def convert_and_summarize_chat_history(chat_history_dict: List[Dict[str, str]]) -> str:
    """Convert chat history to a structured, human-readable summary."""
    messages = convert_dict_to_langchain_messages(chat_history_dict)
    
    output = ["### Chat History Summary", ""]
    
    for i, msg in enumerate(messages, 1):
        if isinstance(msg, HumanMessage):
            output.append(f"**{i}. User Query:**")
            output.append(f"   {msg.content}")
            output.append("")
            
        elif isinstance(msg, AIMessage):
            output.append(f"**{i}. AI Response:**")
            
            if "no influencer data" in msg.content.lower():
                output.append("   Status: No influencer data found, attempting to fetch trending channels.")
                
            elif "TOP Influencer's Channels" in msg.content:
                channels = extract_channel_info(msg.content)
                
                if channels:
                    output.append(f"   Found {len(channels)} channels with detailed information:")
                    output.append("")
                    
                    for channel in channels:
                        output.append(f"   **Channel: {channel['name']}**")
                        output.append(f"   - Handle: {channel['handle']}")
                        output.append(f"   - Subscribers: {channel['subscribers']}")
                        output.append(f"   - Total Views: {channel['total_views']}")
                        output.append(f"   - Videos: {channel['video_count']}")
                        output.append(f"   - Country: {channel['country']}")
                        output.append(f"   - Joined: {channel['joined_date']}")
                        
                        if channel['videos']:
                            output.append(f"   - Top Videos ({len(channel['videos'])}):")
                            for video in channel['videos'][:3]:  # Show top 3 videos
                                output.append(f"     • {video['title']}")
                                output.append(f"       Views: {video['view_count']:,}, Published: {video['published']}")
                                if video['comments']:
                                    output.append(f"       Top comments: {len(video['comments'])}")
                        output.append("")
                        
            elif "Ranked Channels: []" in msg.content:
                output.append("   Status: No ranked channels provided in response.")
                
            elif "Fetch Influencer:" in msg.content:
                output.append("   Status: Attempting to fetch influencer data with search parameters.")
                
            else:
                # Generic AI response
                content_preview = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                output.append(f"   {content_preview}")
            
            output.append("")
            
        elif isinstance(msg, SystemMessage):
            output.append(f"**{i}. System Message:**")
            output.append(f"   {msg.content}")
            output.append("")
    
    return "\n".join(output)

def create_formatted_chat_output(chat_history_dict: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Create a comprehensive formatted output from chat history.
    
    Returns:
        Dictionary containing both LangChain messages and structured summary
    """
    langchain_messages = convert_dict_to_langchain_messages(chat_history_dict)
    summary = convert_and_summarize_chat_history(chat_history_dict)
    
    return {
        "langchain_messages": langchain_messages,
        "formatted_summary": summary,
        "message_count": len(langchain_messages),
        "conversation_stats": {
            "human_messages": sum(1 for msg in langchain_messages if isinstance(msg, HumanMessage)),
            "ai_messages": sum(1 for msg in langchain_messages if isinstance(msg, AIMessage)),
            "system_messages": sum(1 for msg in langchain_messages if isinstance(msg, SystemMessage))
        }
    }

def split_text(text: str, max_length: int = 1800) -> List[Dict[str, Dict[str, str]]]:
    """Split text into chunks for Notion's rich_text limit (2000 characters per block)."""
    return [{"text": {"content": text[i:i + max_length]}} for i in range(0, len(text), max_length)]

async def create_chat_history(input_query: str, user_id: str, chat_history: List[Dict[str, str]], response: str) -> Union[str, None]:
    """Store or update chat history in the Notion database for a given user_id."""
    try:
        CHAT_HISTORY_DB_ID = os.getenv("CHAT_HISTORY_DB_ID")
        if not CHAT_HISTORY_DB_ID:
            print("Error: CHAT_HISTORY_DB_ID not found in environment variables")
            return None

        chat_history_str = json.dumps(chat_history, ensure_ascii=False, indent=2)
        chat_history_blocks = split_text(chat_history_str) if chat_history_str else []

        properties = {
            "User ID": {"title": [{"text": {"content": user_id}}] if user_id else []},
            "Query": {"rich_text": [{"text": {"content": input_query}}] if input_query else []},
            "Chat History": {"rich_text": chat_history_blocks}
        }

        query_response = await notion.databases.query(
            database_id=CHAT_HISTORY_DB_ID,
            filter={
                "property": "User ID",
                "title": {"equals": user_id}
            }
        )

        if query_response["results"]:
            page_id = query_response["results"][0]["id"]
            await notion.pages.update(page_id=page_id, properties=properties)
            print(f"Updated chat history for user_id {user_id}: {page_id}")
            return page_id
        else:
            create_response = await notion.pages.create(
                parent={"database_id": CHAT_HISTORY_DB_ID},
                properties=properties
            )
            print(f"Created new chat history for user_id {user_id}: {create_response['id']}")
            return create_response["id"]

    except Exception as e:
        print(f"Error in create_chat_history: {str(e)}")
        return None

async def get_chat_history(user_id: str) -> List[Dict[str, Any]]:
    """Retrieve chat history for a specific user."""
    try:
        CHAT_HISTORY_DB_ID = os.getenv("CHAT_HISTORY_DB_ID")
        if not CHAT_HISTORY_DB_ID:
            print("Error: CHAT_HISTORY_DB_ID not found in environment variables")
            return []

        response = await notion.databases.query(
            database_id=CHAT_HISTORY_DB_ID,
            filter={
                "property": "User ID",
                "title": {"equals": user_id}
            }
        )
        return response["results"]
    except Exception as e:
        print(f"Error retrieving chat history: {str(e)}")
        return []

async def get_chat_history_for_user(user_id: str) -> Union[List[Dict[str, str]], str]:
    """Retrieve chat history as a list of dictionaries for a specific user."""
    try:
        # Get raw Notion pages
        pages = await get_chat_history(user_id)
        if not pages:
            print(f"No chat history found for user_id: {user_id}")
            return "No chat history found."

        # Extract chat history from the first page (assuming one page per user_id)
        page = pages[0]
        chat_history_text = "".join(
            block["text"]["content"] for block in page["properties"]["Chat History"]["rich_text"]
        )
        
        # Parse the JSON string into a Python list of dictionaries
        try:
            chat_history_dict = json.loads(chat_history_text)
            return chat_history_dict  # Return the raw dictionary list
        except json.JSONDecodeError as e:
            print(f"Error parsing chat history JSON: {str(e)}")
            return "Error parsing chat history."

    except Exception as e:
        print(f"Error in get_chat_history_for_user: {str(e)}")
        return f"Error: {str(e)}"
    
