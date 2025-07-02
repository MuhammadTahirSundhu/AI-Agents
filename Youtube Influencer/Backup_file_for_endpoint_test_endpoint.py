##################################################  Server #########################################3


# from flask import Flask, request, jsonify
# import asyncio
# import re
# import requests
# import os
# import json
# from dotenv import load_dotenv
# from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
# from langchain.prompts import PromptTemplate
# from langchain_google_genai import ChatGoogleGenerativeAI
# from youtube_apis import fetch_channel_with_their_avg_comments, parse_json_to_context_string, read_and_parse_json
# from notion_database import create_influencer, create_query, create_user, notion

# # Load environment variables
# load_dotenv()

# app = Flask(__name__)

# def convert_dict_to_langchain_messages(chat_history_dict):
#     """Convert dictionary format chat history to LangChain message objects."""
#     messages = []
#     for msg in chat_history_dict:
#         if msg["role"] == "user" or msg["role"] == "human":
#             messages.append(HumanMessage(content=msg["content"]))
#         elif msg["role"] == "assistant" or msg["role"] == "ai":
#             messages.append(AIMessage(content=msg["content"]))
#         elif msg["role"] == "system":
#             messages.append(SystemMessage(content=msg["content"]))
#     return messages

# def convert_langchain_to_dict(chat_history):
#     """Convert LangChain message objects to dictionary format."""
#     return [{"role": msg.type, "content": msg.content} for msg in chat_history]

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
#             model="gemini-2.0-flash",
#             google_api_key=os.getenv("GOOGLE_API_KEY"),
#             temperature=0.5
#         )
#         response = llm.invoke(messages)
#         return response.content
#     except Exception as e:
#         return f"Error: Failed to connect to Google Gemini - {str(e)}"

# # def call_llm(prompt, chat_history):
# #     """Call the external LLM API."""
# #     url = "https://206c-20-106-58-127.ngrok-free.app/chat"
# #     prompt_text = prompt.text if hasattr(prompt, 'text') else str(prompt)
    
# #     # Convert chat history to string format for the system message
# #     chat_history_str = ""
# #     for msg in chat_history:
# #         chat_history_str += f"{msg.type}: {msg.content}\n"
    
# #     payload = {
# #         "messages": [
# #             {
# #                 "role": "system",
# #                 "content": (
# #                     "You are an ethical AI Influencer Sourcing Agent designed to assist users in "
# #                     "finding YouTube influencers for marketing campaigns or answering general queries. "
# #                     "Your responses must be honest, transparent, and respect privacy. "
# #                     "You have access to a function for fetching influencer data and must store results before responding. "
# #                     "Interpret user intent, suggest functions when unclear, and prompt for missing parameters."
# #                     f"\n**Chat History**\n{chat_history_str}"
# #                 )
# #             },
# #             {
# #                 "role": "user",
# #                 "content": prompt_text
# #             }
# #         ],
# #         "temperature": 0.5,
# #         "model": "gpt-4o"
# #     }
    
# #     try:
# #         response = requests.post(url, json=payload)
# #         response.raise_for_status()
# #         api_response = response.json()
# #         if api_response.get("status") == "success":
# #             return api_response.get("response")
# #         else:
# #             return f"Error: API request failed - {api_response.get('message', 'No message provided')}"
# #     except requests.RequestException as e:
# #         return f"Error: Failed to connect to the API - {str(e)}"

# def extract_json_response_to_list(input_text):
#     """Extract JSON data from input text and store it in a list."""
#     try:
#         json_pattern = r'```json\n(.*?)\n```'
#         match = re.search(json_pattern, input_text, re.DOTALL)
        
#         if not match:
#             return []
        
#         json_string = match.group(1)
#         parsed_data = json.loads(json_string)
        
#         if not isinstance(parsed_data, list):
#             raise ValueError("JSON data must be a list of objects")
        
#         return parsed_data
    
#     except (json.JSONDecodeError, ValueError) as e:
#         print(f"Error processing JSON: {str(e)}")
#         return []

# def create_prompt_template():
#     """Create and return the prompt template for the chatbot."""
#     return PromptTemplate(
#         template="""
#         You are an ethical YouTube Influencer Sourcing Agent. Your task is to interpret user queries, identify the appropriate function to call, and handle general questions with moral, responsible answers. You have access to one function for fetching influencer data. Follow these guidelines strictly:

#         **CRITICAL INSTRUCTIONS**:
#         ⚠️ **NEVER PROVIDE FAKE, DUMMY, OR FABRICATED DATA** ⚠️
#         - Only work with REAL data provided in the Chat History section
#         - If no chat history data is available, do NOT create fictional influencer information
#         - If chat history is empty or contains no influencer data, only suggest fetching new data or ask for clarification
#         - NEVER generate sample data, placeholder information, or hypothetical results

#         **Available Functions**:
#            1. **Name**: Fetch Influencers
#               - **Description**: Fetch YouTube influencers based on trending or niche criteria. Results include channel info, popular videos, and comments.
#               - **Parameters**:
#                  - method (integer): 1 for trending channels, 2 for niche-based channels
#                  - max_results (integer): maximum 20 channels to fetch
#                  - niche (string): required ONLY for method=2, leave empty for method=1
#               - **Examples**:
#                  - For trending: "Fetch Influencer: method=1, max_results=5, niche="
#                  - For niche: "Fetch Influencer: method=2, max_results=3, niche=fitness"

#         **Data Processing Rules**:
#            1. **Check Chat History First**: 
#               - ALWAYS examine the Chat History section for existing influencer data
#               - If Chat History contains influencer data, proceed to ranking immediately
#               - If Chat History is empty or contains no influencer data, suggest fetching new data
              
#            2. **For trending channels (method=1)**:
#               - Only need method and max_results parameters
#               - Always set niche as empty string
#               - Example response: "Fetch Influencer: method=1, max_results=3, niche= "
              
#            3. **For niche-based channels (method=2)**:
#               - Need all three parameters: method, max_results, and niche
#               - Extract niche from user query (fitness, gaming, tech, etc.)
#               - Example response: "Fetch Influencer: method=2, max_results=5, niche=fitness"
              
#            4. **Parameter Extraction Rules**:
#               - If user mentions "trending", use method=1
#               - If user specifies a topic/niche (fitness, gaming, etc.) or does not mention "trending", use method=2
#               - Extract number from query for max_results
#               - If parameters missing, ask user to clarify
              
#            5. **General Queries**:
#               - For non-influencer queries, provide concise, ethical answers
              
#            6. **Ranking Available Influencer Data**:
#               - **ONLY rank if Chat History contains actual influencer data**
#               - If Chat History has influencer data, immediately rank based on:
#                 - Relevance to Niche (40%): Match keywords, titles, comments with user's requested niche
#                 - Audience Engagement (30%): Comment sentiment, engagement quality, and relevance  
#                 - Popularity (30%): Subscriber count and video views
#                 - Score (0-100): (0.4 * niche_relevance + 0.3 * engagement + 0.3 * popularity)
#                 - Rank in descending order by score, show all available details
#                 - End response with "Finished"
           
#            7. **Ranked Result Format**:
#                 When ranking channels from Chat History data, extract and show these details in JSON format:
#                 ```json
#                 [
#                   {{
#                     "Channel Name": "actual_channel_name_from_chat_history",
#                     "Channel ID": "actual_channel_id_from_chat_history",
#                     "Handle": "actual_handle_from_chat_history",
#                     "Description": "actual_description_from_chat_history",
#                     "Subscribers": actual_subscriber_count_number,
#                     "Total Views": actual_total_views_number,
#                     "Videos Count": actual_video_count_number,
#                     "Joined Date": "actual_joined_date_from_chat_history",
#                     "Country": "actual_country_from_chat_history",
#                     "Top Video Links": "actual_video_links_from_chat_history",
#                     "Top Comments": "actual_comments_from_chat_history",
#                     "External Links": "actual_external_links_from_chat_history",
#                     "Last Updated": "current_date",
#                     "Ranking Score": calculated_score_0_to_100
#                   }}
#                 ]
#                 ```

#         **Response Logic**:
#         1. First, check if Chat History contains influencer data
#         2. If YES: Immediately rank the available data and provide JSON output
#         3. If NO: Determine if user wants to fetch new data or answer general questions
#         4. Never mix real data with fake data
#         5. Always be transparent about data availability

#         **Chat History Data**: {ChatHistory}
#         **User Query**: {Query}
        
#         Remember: 
#         - Use ONLY real data from Chat History section
#         - For method=1 (trending), always use empty niche
#         - For method=2, extract niche from user query
#         - If Chat History has data, rank it immediately - don't ask to fetch more unless specifically requested
#         """,
#         input_variables=['ChatHistory', 'Query']
#     )

# def parse_fetch_command(response):
#     """Parse the Fetch Influencer command from LLM response."""
#     pattern = r"Fetch Influencer:\s*method=(\d+),\s*max_results=(\d+),\s*niche=([^,\n]*)"
#     match = re.search(pattern, response)
    
#     if match:
#         method = int(match.group(1))
#         max_results = int(match.group(2))
#         niche_raw = match.group(3).strip()
        
#         if method == 1:
#             niche = None
#         else:
#             niche = niche_raw if niche_raw else None
            
#         return method, max_results, niche
#     return None, None, None

# async def store_influencers_in_database(influencers):
#     """Store influencers in the database."""
#     try:
#         # Create a user
#         user_id = await create_user("API User", "api@example.com")
#         if not user_id:
#             return False
        
#         # Create a query
#         query_id = await create_query("Find Social Media Influencers", user_id)
#         if not query_id:
#             return False
        
#         # Create influencers from the provided list
#         for influencer in influencers:
#             influencer_id = await create_influencer(
#                 channel_name=influencer.get("Channel Name", ""),
#                 channel_url=influencer.get("External Links", ""),
#                 query_id=query_id,
#                 views=influencer.get("Total Views", 0),
#                 subscribers=influencer.get("Subscribers", 0),
#                 video_count=influencer.get("Videos Count", 0),
#                 handle=influencer.get("Handle", ""),
#                 description=influencer.get("Description", ""),
#                 country=influencer.get("Country", ""),
#                 joined_date=influencer.get("Joined Date", ""),
#                 top_video_links=influencer.get("Top Video Links", ""),
#                 top_comments=influencer.get("Top Comments", ""),
#             )
#             if not influencer_id:
#                 print(f"Failed to create influencer {influencer.get('Channel Name', 'Unknown')}")
#                 return False
        
#         return True
#     except Exception as e:
#         print(f"Error storing influencers: {str(e)}")
#         return False
#     finally:
#         await notion.aclose()

# def process_influencer_query(input_query, chat_history_dict, context=""):
#     """Process a single influencer query and return updated chat history and response."""
    
#     # Convert dictionary chat history to LangChain messages
#     chat_history = convert_dict_to_langchain_messages(chat_history_dict)
    
#     # Add user message to chat history
#     chat_history.append(HumanMessage(content=input_query))
    
#     # Create prompt and get LLM response
#     prompt_template = create_prompt_template()
#     final_prompt = prompt_template.invoke({"Context": context, "Query": input_query})
#     response = call_llm(final_prompt, chat_history)
    
#     # Add assistant response to chat history
#     chat_history.append(AIMessage(content=response))
    
#     # Handle JSON extraction and database storage
#     database_stored = False
#     if "```json" in response:
#         ranked_channels = extract_json_response_to_list(response)
#         if ranked_channels:
#             try:
#                 database_stored = asyncio.run(store_influencers_in_database(ranked_channels))
#             except Exception as e:
#                 print(f"Error storing in database: {str(e)}")
    
#     # Handle fetch influencer command
#     updated_context = context  # Preserve the input context
#     fetch_attempted = False
#     if "Fetch Influencer" in response:
#         method, max_results, niche = parse_fetch_command(response)
#         if method is not None and max_results is not None:
#             if method == 2 and not niche:
#                 # Add error message to chat history
#                 error_msg = "Error: Niche is required for method=2."
#                 chat_history.append(AIMessage(content=error_msg))
#             else:
#                 try:
#                     # Uncomment the next line to use actual API call
#                     # result = fetch_channel_with_their_avg_comments(method, max_results, niche)
#                     # updated_context = parse_json_to_context_string(result)
                    
#                     # Using mock data for now
#                     fetched_data = read_and_parse_json("./channel_comments.json")
#                     # Append new data to existing context instead of replacing
#                     if context and context.strip():
#                         updated_context = context + "\n\n" + fetched_data
#                     else:
#                         updated_context = fetched_data
#                     fetch_attempted = True
#                 except Exception as e:
#                     error_msg = f"Error fetching influencer data: {str(e)}"
#                     chat_history.append(AIMessage(content=error_msg))
    
#     # Convert back to dictionary format
#     updated_chat_history = convert_langchain_to_dict(chat_history)
    
#     return {
#         "updated_chat_history": updated_chat_history,
#         "response": response,
#         "context": updated_context,  # Return updated context
#         "database_stored": database_stored,
#         "fetch_attempted": fetch_attempted
#     }

# @app.route('/Youtube_Influencer_Finder', methods=['POST'])
# def youtube_influencer_finder():
#     """Main endpoint for YouTube Influencer Finder chatbot."""
#     try:
#         # Get input data
#         data = request.get_json()
        
#         if not data:
#             return jsonify({"error": "No JSON data provided"}), 400
        
#         input_query = data.get('input_query', '')
#         chat_history = data.get('chat_history', [])
        
#         if not input_query:
#             return jsonify({"error": "input_query is required"}), 400
        
#         # Process the query
#         result = process_influencer_query(input_query, chat_history)
        
#         # Return the result with updated context
#         return jsonify({
#             "status": "success",
#             "updated_chat_history": result["updated_chat_history"],
#             "response": result["response"],
#             "context": result["context"],  # Return updated context
#             "database_stored": result["database_stored"],
#             "fetch_attempted": result["fetch_attempted"]
#         })
    
#     except Exception as e:
#         return jsonify({
#             "status": "error",
#             "message": str(e)
#         }), 500

# @app.route('/health', methods=['GET'])
# def health_check():
#     """Health check endpoint."""
#     return jsonify({"status": "healthy", "service": "YouTube Influencer Finder"})

# if __name__ == '__main__':
#     print("YouTube Influencer Finder API Started!")
#     print("Available endpoints:")
#     print("- POST /Youtube_Influencer_Finder")
#     print("- GET /health")
#     print("-" * 50)
#     app.run(debug=True, host='0.0.0.0', port=5000)




####################################################################### Client ##########################################################

# import requests
# import json
# import time
# from datetime import datetime
# from typing import Dict, List, Optional

# # Configuration
# ENDPOINT_URL = "http://localhost:5000/Youtube_Influencer_Finder"
# HEALTH_URL = "http://localhost:5000/health"

# class YouTubeInfluencerClient:
#     """Client for testing the YouTube Influencer Finder API."""
    
#     def __init__(self, base_url: str = "http://localhost:5000"):
#         self.base_url = base_url
#         self.endpoint_url = f"{base_url}/Youtube_Influencer_Finder"
#         self.health_url = f"{base_url}/health"
#         self.chat_history = []
#         self.context = ""
#         self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
#     def check_health(self) -> bool:
#         """Check if the server is healthy."""
#         try:
#             response = requests.get(self.health_url, timeout=10)
#             if response.status_code == 200:
#                 health_data = response.json()
#                 print(f"✅ Server is healthy: {health_data.get('service', 'Unknown')}")
#                 return True
#             else:
#                 print(f"❌ Server health check failed: {response.status_code}")
#                 return False
#         except requests.exceptions.RequestException as e:
#             print(f"❌ Cannot connect to server: {e}")
#             return False
    
#     def send_request(self, query: str, timeout: int = 30) -> Optional[Dict]:
#         """Send a POST request to the endpoint with the query and chat history."""
#         # Create payload with chat history as a list
#         payload = {
#             "input_query": query,
#             "chat_history": self.chat_history  # Send chat_history as is (list)
#         }

#         # If context exists, append it as a system message tos chat_history
#         if self.context:
#             payload["chat_history"].append({
#                 "role": "Context",
#                 "content": self.context
#             })

#         try:
#             print(f"🔄 Sending request...")
#             response = requests.post(
#                 self.endpoint_url, 
#                 json=payload, 
#                 timeout=timeout,
#                 headers={"Content-Type": "application/json"}
#             )
#             response.raise_for_status()
#             return response.json()

#         except requests.exceptions.Timeout:
#             print(f"⏰ Request timed out after {timeout} seconds")
#             return None
#         except requests.exceptions.RequestException as e:
#             print(f"❌ Error communicating with the endpoint: {e}")
#             if hasattr(e, 'response') and e.response is not None:
#                 try:
#                     error_detail = e.response.json()
#                     print(f"Error details: {error_detail}")
#                 except:
#                     print(f"Response text: {e.response.text}")
#             return None
    
#     def process_response(self, response: Dict) -> bool:
#         """Process the API response and update internal state."""
#         if not response:
#             return False
            
#         if response.get("status") != "success":
#             print(f"❌ API Error: {response.get('message', 'Unknown error')}")
#             return False
        
#         # Print the main response
#         api_response = response.get("response", "")
#         print(f"\n🤖 Assistant: {api_response}")
        
#         # Update internal state
#         self.chat_history = response.get("updated_chat_history", self.chat_history)
#         self.context = response.get("context", self.context)
        
#         # Show additional info
#         if response.get("database_stored"):
#             print("💾 Data successfully stored in database")
        
#         if response.get("fetch_attempted"):
#             print("🔍 Data fetch was attempted")
        
#         # Show context info if available
#         if self.context and len(self.context) > 100:
#             print(f"📊 Context updated ({len(self.context)} characters)")
        
#         return True
    
#     def print_chat_history(self):
#         """Print the current chat history in a formatted way."""
#         if not self.chat_history:
#             print("📝 No chat history yet.")
#             return
        
#         print("\n📝 Chat History:")
#         print("-" * 50)
#         for i, msg in enumerate(self.chat_history, 1):
#             role = msg.get("role", "unknown")
#             content = msg.get("content", "")
            
#             # Truncate long messages
#             if len(content) > 100:
#                 content = content[:100] + "..."
            
#             role_emoji = {
#                 "user": "👤",
#                 "human": "👤", 
#                 "assistant": "🤖",
#                 "ai": "🤖",
#                 "system": "⚙️"
#             }.get(role, "❓")
            
#             print(f"{i}. {role_emoji} {role.capitalize()}: {content}")
#         print("-" * 50)
    
#     def save_session(self, filename: Optional[str] = None):
#         """Save the current session to a JSON file."""
#         if not filename:
#             filename = f"session_{self.session_id}.json"
        
#         session_data = {
#             "session_id": self.session_id,
#             "timestamp": datetime.now().isoformat(),
#             "chat_history": self.chat_history,
#             "context": self.context
#         }
        
#         try:
#             with open(filename, 'w', encoding='utf-8') as f:
#                 json.dump(session_data, f, indent=2, ensure_ascii=False)
#             print(f"💾 Session saved to {filename}")
#         except Exception as e:
#             print(f"❌ Failed to save session: {e}")
    
#     def load_session(self, filename: str):
#         """Load a session from a JSON file."""
#         try:
#             with open(filename, 'r', encoding='utf-8') as f:
#                 session_data = json.load(f)
            
#             self.chat_history = session_data.get("chat_history", [])
#             self.context = session_data.get("context", "")
#             self.session_id = session_data.get("session_id", self.session_id)
            
#             print(f"📂 Session loaded from {filename}")
#             print(f"   - Messages: {len(self.chat_history)}")
#             print(f"   - Context length: {len(self.context)}")
            
#         except Exception as e:
#             print(f"❌ Failed to load session: {e}")
    
#     def run_interactive_mode(self):
#         """Run the interactive chatbot mode."""
#         print("🎯 YouTube Influencer Finder Chatbot")
#         print("=" * 50)
#         print("Commands:")
#         print("  'exit' or 'quit' - Exit the chatbot")
#         print("  'history' - Show chat history")
#         print("  'save' - Save current session")
#         print("  'clear' - Clear chat history")
#         print("  'status' - Show current status")
#         print("=" * 50)
        
#         # Check server health first
#         if not self.check_health():
#             print("⚠️  Server is not responding. Please check if the server is running.")
#             return
        
#         while True:
#             try:
#                 # Get user input
#                 print("\n" + "="*30)
#                 user_input = input("👤 Your query: ").strip()
                
#                 if not user_input:
#                     continue
                
#                 # Handle special commands
#                 if user_input.lower() in ['exit', 'quit']:
#                     print("👋 Goodbye!")
#                     break
#                 elif user_input.lower() == 'history':
#                     self.print_chat_history()
#                     continue
#                 elif user_input.lower() == 'save':
#                     self.save_session()
#                     continue
#                 elif user_input.lower() == 'clear':
#                     self.chat_history = []
#                     self.context = ""
#                     print("🗑️ Chat history and context cleared!")
#                     continue
#                 elif user_input.lower() == 'status':
#                     print(f"📊 Status:")
#                     print(f"   - Messages in history: {len(self.chat_history)}")
#                     print(f"   - Context length: {len(self.context)}")
#                     print(f"   - Session ID: {self.session_id}")
#                     continue
                
#                 # Send request to the API
#                 response = self.send_request(user_input)
                
#                 # Process and display response
#                 self.process_response(response)
                
#             except KeyboardInterrupt:
#                 print("\n\n⚠️ Interrupted by user")
#                 save_choice = input("💾 Save session before exiting? (y/n): ").strip().lower()
#                 if save_choice == 'y':
#                     self.save_session()
#                 print("👋 Goodbye!")
#                 break
#             except Exception as e:
#                 print(f"❌ Unexpected error: {e}")

# def run_test_scenarios():
#     """Run predefined test scenarios."""
#     client = YouTubeInfluencerClient()
    
#     if not client.check_health():
#         return
    
#     test_queries = [
#         "Find me 3 trending YouTube influencers",
#         "I need fitness influencers for my campaign",
#         "Show me gaming channels with good engagement",
#         "Find tech reviewers on YouTube",
#         "What are the top beauty influencers?"
#     ]
    
#     print("🧪 Running Test Scenarios")
#     print("=" * 50)
    
#     for i, query in enumerate(test_queries, 1):
#         print(f"\n🔍 Test {i}: {query}")
#         response = client.send_request(query)
        
#         if response:
#             client.process_response(response)
#             time.sleep(2)  # Brief pause between tests
#         else:
#             print(f"❌ Test {i} failed")
        
#         print("-" * 30)
    
#     print("\n✅ Test scenarios completed")
    
#     # Save test session
#     client.save_session("test_session.json")

# def main():
#     """Main function with mode selection."""
#     print("🎯 YouTube Influencer Finder Test Client")
#     print("=" * 50)
#     print("Select mode:")
#     print("1. Interactive mode (recommended)")
#     print("2. Run test scenarios")
#     print("3. Check server health only")
    
#     try:
#         choice = input("\nEnter your choice (1-3): ").strip()
        
#         if choice == "1":
#             client = YouTubeInfluencerClient()
#             client.run_interactive_mode()
#         elif choice == "2":
#             run_test_scenarios()
#         elif choice == "3":
#             client = YouTubeInfluencerClient()
#             client.check_health()
#         else:
#             print("❌ Invalid choice. Please run the script again.")
    
#     except KeyboardInterrupt:
#         print("\n👋 Goodbye!")
#     except Exception as e:
#         print(f"❌ Error: {e}")

# if __name__ == "__main__":
#     main()

