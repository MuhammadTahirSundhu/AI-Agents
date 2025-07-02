import os
from dotenv import load_dotenv
from notion_client import AsyncClient
import asyncio
from datetime import datetime

load_dotenv()
notion = AsyncClient(auth=os.getenv("NOTION_API_Token"))

USERS_DB_ID = "2218303fa6a98093a3f6fdc6b74d862e"
QUERIES_DB_ID = "2218303fa6a980a486f5fc3135ba6968"
INFLUENCERS_DB_ID = "2218303fa6a9805ebe83fc03d7d69996"

async def get_database_schema(database_id, database_name):
    try:
        db = await notion.databases.retrieve(database_id=database_id)
        print(f"\nSchema for {database_name}:")
        for prop_name, prop_info in db["properties"].items():
            print(f"  - {prop_name}: {prop_info['type']}")
        return db
    except Exception as e:
        print(f"Error retrieving schema for {database_name}: {str(e)}")
        return None

async def create_user(username, email):
    try:
        response = await notion.pages.create(
            parent={"database_id": USERS_DB_ID},
            properties={
                "Username": {"title": [{"text": {"content": username}}]},
                "Email": {"email": email},
            },
        )
        return response["id"]
    except Exception as e:
        print(f"Error creating user: {str(e)}")
        return None

async def create_query(query, user_id):
    try:
        response = await notion.pages.create(
            parent={"database_id": QUERIES_DB_ID},
            properties={
                "Query": {"title": [{"text": {"content": query}}]},
                "Related User": {"relation": [{"id": user_id}]},
                "Query ID": {"rich_text": [{"text": {"content": f"query_{int(datetime.now().timestamp())}"}}]},
            },
        )
        return response["id"]
    except Exception as e:
        print(f"Error creating query: {str(e)}")
        return None

async def create_influencer(channel_name, channel_url, views=0, subscribers=0, video_count=0, handle="", description="", country="", joined_date=None, top_video_links="", top_comments="", user_id=""):
    try:
        response = await notion.pages.create(
            parent={"database_id": INFLUENCERS_DB_ID},
            properties={
                "Channel Name": {"title": [{"text": {"content": channel_name}}]},
                "Subscribers": {"number": subscribers},
                "Total Views": {"number": views},
                "Vedio Count": {"number": video_count},
                "External Links": {"rich_text": [{"text": {"content": channel_url}}] if channel_url else []},
                "Channel ID": {"rich_text": [{"text": {"content": f"channel_{int(datetime.now().timestamp())}"}}]},
                "Last Updated": {"date": {"start": datetime.now().isoformat()}},
                "Handle": {"rich_text": [{"text": {"content": handle}}] if handle else []},
                "Description": {"rich_text": [{"text": {"content": description}}] if description else []},
                "Country": {"select": {"name": country}} if country else {},
                "Joined Date": {"date": {"start": joined_date}} if joined_date else {},
                "Top Video Links": {"rich_text": [{"text": {"content": top_video_links}}] if top_video_links else []},
                "Top Comments": {"rich_text": [{"text": {"content": top_comments}}] if top_comments else []},
                "User ID": {"rich_text": [{"text": {"content": user_id}}]},
            },
        )
        return response["id"]
    except Exception as e:
        print(f"Error creating influencer: {str(e)}")
        return None

async def get_queries_for_user(user_id):
    try:
        response = await notion.databases.query(
            database_id=QUERIES_DB_ID,
            filter={"property": "Related User", "relation": {"contains": user_id}},
        )
        print("\nQueries for User:")
        for page in response["results"]:
            title = page["properties"]["Query"]["title"][0]["text"]["content"]
            print(f"  - {title} (ID: {page['id']})")
        return response["results"]
    except Exception as e:
        print(f"Error querying queries for user: {str(e)}")
        return []

async def get_influencers_for_query(query_id):
    try:
        response = await notion.databases.query(
            database_id=INFLUENCERS_DB_ID,
            filter={"property": "Related Query", "relation": {"contains": query_id}},
        )
        print("\nInfluencers for Query:")
        for page in response["results"]:
            name = page["properties"]["Channel Name"]["title"][0]["text"]["content"]
            print(f"  - {name} (ID: {page['id']})")
        return response["results"]
    except Exception as e:
        print(f"Error querying influencers for query: {str(e)}")
        return []
