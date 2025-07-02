
import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Optional

# Configuration
ENDPOINT_URL = "http://localhost:5000/Youtube_Influencer_Finder"
HEALTH_URL = "http://localhost:5000/health"

class YouTubeInfluencerClient:
    """Client for testing the YouTube Influencer Finder API."""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.endpoint_url = f"{base_url}/Youtube_Influencer_Finder"
        self.health_url = f"{base_url}/health"
        self.chat_history = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.user_id = "100001"
    
    def check_health(self) -> bool:
        """Check if the server is healthy."""
        try:
            response = requests.get(self.health_url, timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Server is healthy: {health_data.get('service', 'Unknown')}")
                return True
            else:
                print(f"❌ Server health check failed: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Cannot connect to server: {e}")
            return False
    
    def send_request(self, query: str, timeout: int = 30) -> Optional[Dict]:
        """Send a POST request to the endpoint with the query and chat history."""
        # Create payload with chat history as ChatHistory
        payload = {
            "input_query": query,
            "ChatHistory": self.chat_history,
            "user_id":self.user_id
        }

        try:
            print(f"🔄 Sending request...")
            response = requests.post(
                self.endpoint_url, 
                json=payload, 
                timeout=timeout,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            print(f"⏰ Request timed out after {timeout} seconds")
            return None
        except requests.exceptions.RequestException as e:
            print(f"❌ Error communicating with the endpoint: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    print(f"Error details: {error_detail}")
                except:
                    print(f"Response text: {e.response.text}")
            return None
    
    def process_response(self, response: Dict) -> bool:
        """Process the API response and update internal state."""
        if not response:
            return False
            
        if response.get("status") != "success":
            print(f"❌ API Error: {response.get('message', 'Unknown error')}")
            return False
        
        # Print the main response
        api_response = response.get("response", "")
        print(f"\n🤖 Assistant: {api_response}")
        
        # Print ranked channels if available
        ranked_channels = response.get("ranked_channels", [])
        if ranked_channels:
            print("\n📊 Ranked Channels:")
            print("-" * 50)
            for i, channel in enumerate(ranked_channels, 1):
                print(f"{i}. {channel.get('Channel Name', 'Unknown Channel')}")
                print(f"   - Channel ID: {channel.get('Channel ID', 'N/A')}")
                print(f"   - Handle: {channel.get('Handle', 'N/A')}")
                print(f"   - Subscribers: {channel.get('Subscribers', 0)}")
                print(f"   - Total Views: {channel.get('Total Views', 0)}")
                print(f"   - Videos Count: {channel.get('Videos Count', 0)}")
                print(f"   - Country: {channel.get('Country', 'N/A')}")
                print(f"   - Joined Date: {channel.get('Joined Date', 'N/A')}")
                print(f"   - Ranking Score: {channel.get('Ranking Score', 0)}")
                print("-" * 50)
            # Append ranked channels to chat history as an assistant message
            self.chat_history.append({
                "role": "Ranked Channels",
                "content": json.dumps(ranked_channels, indent=2)
            })
        
        # Update internal state with updated chat history from server
        self.chat_history = response.get("updated_chat_history", self.chat_history)
        
        # Show additional info
        if response.get("database_stored"):
            print("💾 Data successfully stored in database")
        
        if response.get("fetch_attempted"):
            print("🔍 Data fetch was attempted")
        
        return True
    
    def print_chat_history(self):
        """Print the current chat history in a formatted way."""
        if not self.chat_history:
            print("📝 No chat history yet.")
            return
        
        print("\n📝 Chat History:")
        print("-" * 50)
        for i, msg in enumerate(self.chat_history, 1):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            # Truncate long messages
            if len(content) > 100:
                content = content[:100] + "..."
            
            role_emoji = {
                "user": "👤",
                "human": "👤", 
                "assistant": "🤖",
                "ai": "🤖",
                "system": "⚙️"
            }.get(role, "❓")
            
            print(f"{i}. {role_emoji} {role.capitalize()}: {content}")
        print("-" * 50)
    
    def save_session(self, filename: Optional[str] = None):
        """Save the current session to a JSON file."""
        if not filename:
            filename = f"session_{self.session_id}.json"
        
        session_data = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "chat_history": self.chat_history
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            print(f"💾 Session saved to {filename}")
        except Exception as e:
            print(f"❌ Failed to save session: {e}")
    
    def load_session(self, filename: str):
        """Load a session from a JSON file."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            self.chat_history = session_data.get("chat_history", [])
            self.session_id = session_data.get("session_id", self.session_id)
            
            print(f"📂 Session loaded from {filename}")
            print(f"   - Messages: {len(self.chat_history)}")
            
        except Exception as e:
            print(f"❌ Failed to load session: {e}")
    
    def run_interactive_mode(self):
        """Run the interactive chatbot mode."""
        print("🎯 YouTube Influencer Finder Chatbot")
        print("=" * 50)
        print("Commands:")
        print("  'exit' or 'quit' - Exit the chatbot")
        print("  'history' - Show chat history")
        print("  'save' - Save current session")
        print("  'clear' - Clear chat history")
        print("  'status' - Show current status")
        print("=" * 50)
        
        # Check server health first
        if not self.check_health():
            print("⚠️ Server is not responding. Please check if the server is running.")
            return
        
        while True:
            try:
                # Get user input
                print("\n" + "="*30)
                user_input = input("👤 Your query: ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in ['exit', 'quit']:
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'history':
                    self.print_chat_history()
                    continue
                elif user_input.lower() == 'save':
                    self.save_session()
                    continue
                elif user_input.lower() == 'clear':
                    self.chat_history = []
                    print("🗑️ Chat history cleared!")
                    continue
                elif user_input.lower() == 'status':
                    print(f"📊 Status:")
                    print(f"   - Messages in history: {len(self.chat_history)}")
                    print(f"   - Session ID: {self.session_id}")
                    continue
                
                # Send request to the API
                response = self.send_request(user_input)
                
                # Process and display response
                self.process_response(response)
                
            except KeyboardInterrupt:
                print("\n\n⚠️ Interrupted by user")
                save_choice = input("💾 Save session before exiting? (y/n): ").strip().lower()
                if save_choice == 'y':
                    self.save_session()
                print("👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")

def run_test_scenarios():
    """Run predefined test scenarios."""
    client = YouTubeInfluencerClient()
    
    if not client.check_health():
        return
    
    test_queries = [
        "Find me 3 trending YouTube influencers",
        "I need fitness influencers for my campaign",
        "Show me gaming channels with good engagement",
        "Find tech reviewers on YouTube",
        "What are the top beauty influencers?"
    ]
    
    print("🧪 Running Test Scenarios")
    print("=" * 50)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: {query}")
        response = client.send_request(query)
        
        if response:
            client.process_response(response)
            time.sleep(2)  # Brief pause between tests
        else:
            print(f"❌ Test {i} failed")
        
        print("-" * 30)
    
    print("\n✅ Test scenarios completed")
    
    # Save test session
    client.save_session("test_session.json")

def main():
    """Main function with mode selection."""
    print("🎯 YouTube Influencer Finder Test Client")
    print("=" * 50)
    print("Select mode:")
    print("1. Interactive mode (recommended)")
    print("2. Run test scenarios")
    print("3. Check server health only")
    
    try:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            client = YouTubeInfluencerClient()
            client.run_interactive_mode()
        elif choice == "2":
            run_test_scenarios()
        elif choice == "3":
            client = YouTubeInfluencerClient()
            client.check_health()
        else:
            print("❌ Invalid choice. Please run the script again.")
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()