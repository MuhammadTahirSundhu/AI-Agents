import asyncio
import re
import requests
import os
import json
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from youtube_apis import fetch_channel_with_their_avg_comments, parse_json_to_context_string, read_and_parse_json
from notion_database import create_influencer, create_query, create_user, notion
# Load environment variables
load_dotenv()

# Initialize chat history as a list of LangChain message objects
chat_history = []

# File to save chat history
HISTORY_FILE = "chat_history.txt"

def append_to_history(role, message):
    """Append a message to the in-memory chat history."""
    if role == "user":
        chat_history.append(HumanMessage(content=message))
    elif role == "assistant":
        chat_history.append(AIMessage(content=message))

def save_history_to_file():
    """Save the in-memory chat history to a file."""
    with open(HISTORY_FILE, "w", encoding="utf-8") as file:
        history_dict = [{"role": msg.type, "content": msg.content} for msg in chat_history]
        json.dump(history_dict, file, indent=2)

def display_history():
    """Display the current in-memory chat history."""
    print("\nChat History:")
    for msg in chat_history:
        print(f"{msg.type}: {msg.content}")

def call_llm(prompt,chat_history):
    url = "https://206c-20-106-58-127.ngrok-free.app/chat"
    prompt_text = prompt.text if hasattr(prompt, 'text') else str(prompt)
    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an ethical AI Influencer Sourcing Agent designed to assist users in "
                    "finding YouTube influencers for marketing campaigns or answering general queries. "
                    "Your responses must be honest, transparent, and respect privacy. "
                    "You have access to a function for fetching influencer data and must store results before responding. "
                    "Interpret user intent, suggest functions when unclear, and prompt for missing parameters."
                    f"\n**Chat History**\n{chat_history}"
                )
            },
            {
                "role": "user",
                "content": prompt_text
            }
        ],
        "temperature": 0.5,
        "model": "gpt-4o"
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        api_response = response.json()
        # print("Raw API response:", api_response)
        if api_response.get("status") == "success":
            return api_response.get("response")
        else:
            return f"Error: API request failed - {api_response.get('message', 'No message provided')}"
    except requests.RequestException as e:
        return f"Error: Failed to connect to the API - {str(e)}"

def extract_json_response_to_list(input_text):
    """
    Extract JSON data from input text, ignoring all non-JSON content, and store it in a list.
    
    Args:
        input_text: String containing JSON data within ```json and ``` delimiters.
    
    Returns:
        List of dictionaries containing the parsed JSON data, or empty list on error.
    """
    try:
        # Use regex to find JSON block between ```json and ```
        json_pattern = r'```json\n(.*?)\n```'
        match = re.search(json_pattern, input_text, re.DOTALL)
        
        if not match:
            print("Error: No JSON block found in the input text")
            return []
        
        # Extract the JSON string
        json_string = match.group(1)
        
        # Parse JSON string into a Python object
        parsed_data = json.loads(json_string)
        
        # Ensure the parsed data is a list
        if not isinstance(parsed_data, list):
            raise ValueError("JSON data must be a list of objects")
        
        return parsed_data
    
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {str(e)}")
        return []
    except Exception as e:
        print(f"Error: Failed to process JSON - {str(e)}")
        return []


# def call_llm(prompt, chat_history):
#     """Call the external LLM API or fallback to Google Gemini if API fails."""
#     url = "https://206c-20-106-58-127.ngrok-free.app/chat"
#     prompt_text = prompt.text if hasattr(prompt, 'text') else str(prompt)
    
#     # Prepare messages for API or Gemini
#     messages = [
#         SystemMessage(content="""You are an ethical AI Influencer Sourcing Agent designed to assist users in finding YouTube influencers for marketing campaigns or answering general queries. 
#         Your responses must be honest, transparent, and respect privacy. You have access to a function for fetching influencer data and must store results before responding. 
#         Interpret user intent, suggest functions when unclear, and prompt for missing parameters."""),
#         *chat_history,
#         HumanMessage(content=prompt_text)
#     ]

#     # Try external API first
#     try:
#         payload = {
#             "messages": [{"role": msg.type, "content": msg.content} for msg in messages],
#             "temperature": 0.5,
#             "model": "gpt-4o"
#         }
#         response = requests.post(url, json=payload, timeout=10)
#         response.raise_for_status()
#         api_response = response.json()
#         if api_response.get("status") == "success":
#             return api_response.get("response")
#         else:
#             print(f"API request failed: {api_response.get('message', 'No message provided')}")
#     except (requests.RequestException, ValueError) as e:
#         print(f"External API failed: {str(e)}. Switching to Google Gemini.")

#     # Fallback to Google Gemini
#     try:
#         llm = ChatGoogleGenerativeAI(
#             model="gemini-1.5-flash",
#             google_api_key=os.getenv("GOOGLE_API_KEY"),
#             temperature=0.5
#         )
#         response = llm.invoke(messages)
#         return response.content
#     except Exception as e:
#         return f"Error: Failed to connect to Google Gemini - {str(e)}"

def create_prompt_template():
    """Create and return the prompt template for the chatbot."""
    return PromptTemplate(
        template="""
        You are an ethical YouTube Influencer Sourcing Agent. Your task is to interpret user queries, identify the appropriate function to call, and handle general questions with moral, responsible answers. You have access to one function for fetching influencer data. Follow these guidelines strictly:

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

        **Instructions**:
           1. **For trending channels (method=1)**:
              - Only need method and max_results parameters
              - Always set niche as empty string
              - Example response: "Fetch Influencer: method=1, max_results=3, niche= "
              
           2. **For niche-based channels (method=2)**:
              - Need all three parameters: method, max_results, and niche
              - Extract niche from user query (fitness, gaming, tech, etc.)
              - Example response: "Fetch Influencer: method=2, max_results=5, niche=fitness"
              
           3. **Parameter Extraction Rules**:
              - If user mentions "trending", use method=1
              - If user specifies a topic/niche (fitness, gaming, etc.) or not mention "trending", use method=2
              - Extract number from query for max_results
              - If parameters missing, ask user to clarify
              
           4. **General Queries**:
              - For non-influencer queries, provide concise, ethical answers
              
           5. **Ranking Previously Fetched Influencers**:
              - If "Fetch Influencer" was called and context exists, then rank based on:
                - Relevance to Niche (40%): Match keywords, titles, comments
                - Audience Engagement (30%): Comment sentiment and relevance  
                - Popularity (30%): Subscriber count and video views
                - Score (0-100): (0.4 * niche_relevance + 0.3 * engagement + 0.3 * popularity)
                - Rank in descending order, show all details
                - End response with "Finished"
           
           6. **Ranked Result format**:
                Once you rank all the channels, get the below details of all channels and show it in json format so that I can store it in json file:
                - Channel Name        - Title  
                - Channel ID          - Text  
                - Handle              - Text  
                - Description         - Long Text  
                - Subscribers         - Number  
                - Total Views         - Number  
                - Videos Count        - Number  
                - Joined Date         - Date  
                - Country             - Select  
                - Top Video Links     - Text / Multi-line Text  
                - Top Comments        - Long Text  
                - External Links      - Multi-line Text  
                - Last Updated        - Date  
                - 

        **Context**: {Context}
        **Query**: {Query}
        
        Remember: For method=1 (trending), always use empty niche. For method=2, extract niche from user query.
        """,
        input_variables=['Context', 'Query']
    )

def extract_parameters_from_query(query):
    """Extract method, max_results, and niche from user query."""
    method = None
    max_results = None
    niche = None
    
    # Extract number for max_results
    numbers = re.findall(r'\b(\d+)\b', query)
    if numbers:
        max_results = int(numbers[0])
    
    # Determine method based on keywords
    query_lower = query.lower()
    if 'trending' in query_lower:
        method = 1
        niche = ""  # Empty for trending
    elif any(keyword in query_lower for keyword in ['fitness', 'gaming', 'tech', 'cooking', 'beauty', 'music', 'education']) or "popular" in query_lower:
        method = 2
        # Extract niche
        niche_keywords = ['fitness', 'gaming', 'tech', 'technology', 'cooking', 'beauty', 'music', 'education', 'sports']
        for keyword in niche_keywords:
            if keyword in query_lower:
                niche = keyword
                break
    
    return method, max_results, niche

def parse_fetch_command(response):
    """Parse the Fetch Influencer command from LLM response."""
    # Updated regex pattern to handle empty niche
    pattern = r"Fetch Influencer:\s*method=(\d+),\s*max_results=(\d+),\s*niche=([^,\n]*)"
    match = re.search(pattern, response)
    
    if match:
        method = int(match.group(1))
        max_results = int(match.group(2))
        niche_raw = match.group(3).strip()
        
        # Handle empty niche for method=1
        if method == 1:
            niche = None
        else:
            niche = niche_raw if niche_raw else None
            
        return method, max_results, niche
    return None, None, None



async def store_influencers_in_database(influencers):
    try:

        # Create a user
        user_id = await create_user("Jane Doe", "jane@example.com")
        if not user_id:
            return
        
        # Create a query
        query_id = await create_query("Find Social Media Influencers", user_id)
        if not query_id:
            return
        
        # Create influencers from the provided list
        for influencer in influencers:
            influencer_id = await create_influencer(
                channel_name=influencer["Channel Name"],
                channel_url=influencer["External Links"],
                query_id=query_id,
                views=influencer["Total Views"],
                subscribers=influencer["Subscribers"],
                video_count=influencer["Videos Count"],
                handle=influencer["Handle"],
                description=influencer["Description"],
                country=influencer["Country"],
                joined_date=influencer["Joined Date"],
                top_video_links=influencer["Top Video Links"],
                top_comments=influencer["Top Comments"],
            )
            if not influencer_id:
                print(f"Failed to create influencer {influencer['channel_name']}")
                return

    except Exception as e:
        print(f"Error in main: {str(e)}")
    finally:
        await notion.aclose()

def start_chatbot():
    """Start the chatbot to process user queries."""
    prompt_template = create_prompt_template()
    context = ""

    while True:
        user_input = input("Enter your query (or 'exit' to quit):\n")
        if user_input.lower() == "exit":
            break

        append_to_history("user", user_input)
        final_prompt = prompt_template.invoke({"Context": context, "Query": user_input})
        response = call_llm(final_prompt, chat_history)
        append_to_history("assistant", response)

        if "```json" in response:
            ranked_channels = extract_json_response_to_list(response)
            if ranked_channels:
                # print("Ranked channels:", ranked_channels)
                asyncio.run(store_influencers_in_database(ranked_channels))

        if "Fetch Influencer" in response:
            method, max_results, niche = parse_fetch_command(response)
            if method is not None and max_results is not None:
                print(f"Parsed parameters: method={method}, max_results={max_results}, niche={niche}")
                if method == 2 and not niche:
                    print("Error: Niche is required for method=2.")
                    continue

                try:
                    # Call the API function
                    # result = fetch_channel_with_their_avg_comments(method, max_results, niche)
                    # context = parse_json_to_context_string(result)
                    context = read_and_parse_json("./channel_comments.json")
                    # print("Successfully fetched influencer data!")
                except Exception as e:
                    print(f"Error fetching influencer data: {str(e)}")
                    context = ""
            else:
                print("Error: Could not parse Fetch Influencer parameters.")
                print(f"Response was: {response}")
        display_history()
        if "Finished" in response:
            save_history_to_file()
            print("Chat history saved!")
def main():
    """Main function to start the chatbot."""
    print("YouTube Influencer Finder Agent Started!")
    print("You can ask for:")
    print("- Trending channels: 'Give me 3 trending fitness channels'")
    print("- Niche channels: 'Give me 5 gaming channels'")
    print("- Ranking: 'Rank the channels' (after fetching)")
    print("-" * 50)
    start_chatbot()

if __name__ == "__main__":
    main()