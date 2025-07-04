import streamlit as st
import json
import os
from typing import Dict, List, Optional, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
import time

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="SmartGigs - AI Fiverr Assistant",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS with truly fixed input and better design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Reset and base styles */
    * {
        box-sizing: border-box;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container with fixed bottom space */
    .main .block-container {
        padding-bottom: 120px !important;
        max-width: 1200px;
    }
    
    /* Header section */
    .app-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .app-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        animation: shimmer 3s infinite;
        transform: rotate(45deg);
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%) rotate(45deg); }
        100% { transform: translateX(100%) rotate(45deg); }
    }
    
    .app-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
    }
    
    .app-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
        position: relative;
        z-index: 1;
    }
    
    /* Chat container */
    .chat-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        max-height: 60vh;
        overflow-y: auto;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
    }
    
    /* Custom scrollbar */
    .chat-container::-webkit-scrollbar {
        width: 6px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 3px;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 3px;
    }
    
    /* Message styles */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 1rem 0 1rem 25%;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
        animation: slideInRight 0.3s ease-out;
        position: relative;
    }
    
    .user-message::before {
        content: "👤";
        position: absolute;
        right: -30px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 1.2rem;
    }
    
    .ai-message {
        background: rgba(255, 255, 255, 0.1);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 1rem 25% 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        animation: slideInLeft 0.3s ease-out;
        position: relative;
    }
    
    .ai-message::before {
        content: "🤖";
        position: absolute;
        left: -30px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 1.2rem;
    }
    
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    /* Gig cards */
    .gig-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .gig-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.2);
        border-color: rgba(102, 126, 234, 0.5);
    }
    
    .gig-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    .gig-title {
        color: #667eea;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .gig-seller {
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
    
    .gig-price {
        color: #667eea;
        font-size: 1.3rem;
        font-weight: 700;
    }
    
    .gig-rating {
        color: #ffd700;
        font-weight: 600;
    }
    
    .gig-details {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin: 1rem 0;
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.9rem;
    }
    
    .gig-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 1rem;
    }
    
    .tag {
        background: rgba(102, 126, 234, 0.2);
        color: #667eea;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    /* Metrics cards */
    .metrics-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.1);
    }
    
    .metric-value {
        color: #667eea;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.9rem;
    }
    
    /* Section headers */
    .section-header {
        color: white;
        font-size: 1.8rem;
        font-weight: 600;
        text-align: center;
        margin: 3rem 0 2rem 0;
        position: relative;
    }
    
    .section-header::after {
        content: '';
        position: absolute;
        bottom: -10px;
        left: 50%;
        transform: translateX(-50%);
        width: 60px;
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 2px;
    }
    
    /* Sidebar styles */
    .sidebar .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        color: white;
    }
    
    .sidebar .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        width: 100%;
    }
    
    .sidebar .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    
    .status-ready {
        background: rgba(34, 197, 94, 0.2);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    
    .status-processing {
        background: rgba(251, 146, 60, 0.2);
        color: #fb923c;
        border: 1px solid rgba(251, 146, 60, 0.3);
    }
    
    .status-ready::before {
        content: "✅";
        margin-right: 0.5rem;
    }
    
    .status-processing::before {
        content: "⏳";
        margin-right: 0.5rem;
        animation: pulse 1s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* TRULY FIXED INPUT BAR */
    .fixed-input-bar {
        position: fixed !important;
        bottom: 0 !important;
        left: 0 !important;
        right: 0 !important;
        background: rgba(15, 15, 35, 0.95) !important;
        backdrop-filter: blur(20px) !important;
        border-top: 1px solid rgba(102, 126, 234, 0.3) !important;
        padding: 1rem 2rem !important;
        z-index: 9999 !important;
        box-shadow: 0 -10px 30px rgba(0, 0, 0, 0.3) !important;
    }
    
    .input-container {
        max-width: 1200px !important;
        margin: 0 auto !important;
        display: flex !important;
        gap: 1rem !important;
        align-items: center !important;
    }
    
    .input-container .stTextInput {
        flex: 1 !important;
        margin: 0 !important;
    }
    
    .input-container .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 2px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 25px !important;
        padding: 12px 20px !important;
        color: white !important;
        font-size: 16px !important;
        height: 50px !important;
        transition: all 0.3s ease !important;
    }
    
    .input-container .stTextInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.3) !important;
        outline: none !important;
    }
    
    .input-container .stTextInput > div > div > input::placeholder {
        color: rgba(255, 255, 255, 0.5) !important;
    }
    
    .input-container .stButton {
        margin: 0 !important;
    }
    
    .input-container .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        height: 50px !important;
        min-width: 100px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3) !important;
    }
    
    .input-container .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4) !important;
    }
    
    .input-container .stButton > button:disabled {
        opacity: 0.6 !important;
        transform: none !important;
    }
    
    /* Hide form labels */
    .input-container .stTextInput > label {
        display: none !important;
    }
    
    /* Ensure content doesn't overlap with fixed input */
    .main-content-wrapper {
        padding-bottom: 100px !important;
    }
    
    /* Loading animation */
    .loading-dots {
        display: inline-block;
    }
    
    .loading-dots::after {
        content: '';
        animation: dots 1.5s infinite;
    }
    
    @keyframes dots {
        0%, 20% { content: ''; }
        40% { content: '.'; }
        60% { content: '..'; }
        80%, 100% { content: '...'; }
    }
    
    /* Recommendation cards */
    .recommendation-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
        animation: glow 2s ease-in-out infinite alternate;
        position: relative;
        overflow: hidden;
    }
    
    .recommendation-card::before {
        content: '⭐';
        position: absolute;
        top: 1rem;
        right: 1rem;
        font-size: 1.5rem;
        animation: sparkle 2s ease-in-out infinite;
    }
    
    @keyframes glow {
        from { box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3); }
        to { box-shadow: 0 25px 50px rgba(102, 126, 234, 0.5); }
    }
    
    @keyframes sparkle {
        0%, 100% { transform: scale(1) rotate(0deg); }
        50% { transform: scale(1.2) rotate(180deg); }
    }
</style>
""", unsafe_allow_html=True)

# Initialize LLM
@st.cache_resource
def init_llm():
    try:
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    except Exception as e:
        st.error(f"❌ Error initializing LLM: {str(e)}")
        return None

# Mock gigs data for demonstration
MOCK_GIGS = [
    {
        "title": "I will create a professional restaurant website with online ordering",
        "seller": "webdev_pro",
        "rating": "4.9",
        "reviews": "150",
        "price": "$299",
        "category": "Web Development",
        "delivery_time": "7 days",
        "experience_level": "Expert",
        "revisions": "3",
        "tags": "restaurant, website, responsive, modern, ordering"
    },
    {
        "title": "I will design a stunning restaurant logo and brand identity",
        "seller": "design_master",
        "rating": "4.8",
        "reviews": "89",
        "price": "$45",
        "category": "Logo Design",
        "delivery_time": "3 days",
        "experience_level": "Professional",
        "revisions": "5",
        "tags": "logo, restaurant, branding, creative, identity"
    },
    {
        "title": "I will write compelling restaurant menu descriptions",
        "seller": "content_writer",
        "rating": "4.7",
        "reviews": "67",
        "price": "$25",
        "category": "Content Writing",
        "delivery_time": "2 days",
        "experience_level": "Professional",
        "revisions": "2",
        "tags": "menu, writing, restaurant, content, descriptions"
    },
    {
        "title": "I will create restaurant social media marketing strategy",
        "seller": "marketing_guru",
        "rating": "4.9",
        "reviews": "134",
        "price": "$75",
        "category": "Digital Marketing",
        "delivery_time": "5 days",
        "experience_level": "Expert",
        "revisions": "2",
        "tags": "social media, marketing, restaurant, strategy, promotion"
    }
]

# Tools
@tool
def get_gigs(query: str) -> List[Dict[str, str]]:
    """Fetches a list of Fiverr gigs based on the user's query."""
    gigs_file = "gigs.json"
    
    # Try to load from file first
    if os.path.exists(gigs_file):
        try:
            with open(gigs_file, "r") as f:
                gigs_data = json.load(f)
            
            gigs = gigs_data.get("gig_list", {}).get("gigs", [])
            
            filtered_gigs = [
                gig for gig in gigs 
                if query.lower() in gig.get("title", "").lower()
                or query.lower() in gig.get("sub_category_name", "").lower()
                or query.lower() in gig.get("category", "").lower()
            ]
            
            return filtered_gigs or gigs[:10]
        except Exception as e:
            st.error(f"❌ Error reading gigs file: {str(e)}")
    
    # Use mock data as fallback
    filtered_gigs = [
        gig for gig in MOCK_GIGS 
        if query.lower() in gig["title"].lower()
        or query.lower() in gig.get("category", "").lower()
        or query.lower() in gig.get("tags", "").lower()
    ]
    
    return filtered_gigs or MOCK_GIGS

@tool
def chatbot_keyword_search(project_details: str) -> str:
    """Extract relevant keywords from project details."""
    llm = init_llm()
    if not llm:
        return "Error: Unable to initialize LLM"
    
    try:
        response = llm.invoke(
            f"Project details: {project_details}\n\n"
            f"Extract the most relevant Fiverr service keyword from the above project details. "
            f"Respond with just the keyword (e.g., 'website', 'logo', 'marketing', etc.)"
        )
        return response.content.strip().lower()
    except Exception as e:
        return f"Error in keyword search: {str(e)}"

# State management
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "messages": [],
        "graph": None,
        "current_gigs": [],
        "awaiting_recommendation": False,
        "processing": False,
        "last_input": "",
        "chat_history": [],
        "input_key": 0
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def build_chatbot_graph():
    """Build the chatbot graph."""
    llm = init_llm()
    if not llm:
        return None
    
    tools = [get_gigs, chatbot_keyword_search]
    llm_with_tools = llm.bind_tools(tools=tools)
    
    def chatbot(state: State):
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}
    
    builder = StateGraph(State)
    builder.add_node("chatbot", chatbot)
    builder.add_node("tools", ToolNode(tools))
    
    builder.add_edge(START, "chatbot")
    builder.add_conditional_edges(
        "chatbot",
        tools_condition,
        {
            "tools": "tools",
            "__end__": END,
        },
    )
    builder.add_edge("tools", "chatbot")
    
    return builder.compile()

def display_gig_card(gig, is_recommended=False):
    """Display a single gig card."""
    card_class = "recommendation-card" if is_recommended else "gig-card"
    
    st.markdown(f"""
    <div class="{card_class}">
        <div class="gig-title">🎯 {gig.get('title', 'N/A')}</div>
        <div class="gig-seller">👤 {gig.get('seller', 'N/A')}</div>
        
        <div style="display: flex; justify-content: space-between; align-items: center; margin: 1rem 0;">
            <div class="gig-rating">⭐ {gig.get('rating', 'N/A')} ({gig.get('reviews', 'N/A')} reviews)</div>
            <div class="gig-price">{gig.get('price', 'N/A')}</div>
        </div>
        
        <div class="gig-details">
            <div>📦 <strong>Category:</strong><br>{gig.get('category', 'N/A')}</div>
            <div>🚀 <strong>Delivery:</strong><br>{gig.get('delivery_time', 'N/A')}</div>
            <div>🎯 <strong>Level:</strong><br>{gig.get('experience_level', 'N/A')}</div>
            <div>🔄 <strong>Revisions:</strong><br>{gig.get('revisions', 'N/A')}</div>
        </div>
        
        <div class="gig-tags">
            {' '.join([f'<span class="tag">{tag.strip()}</span>' for tag in gig.get('tags', '').split(',') if tag.strip()])}
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_metrics(gigs):
    """Display metrics about the fetched gigs."""
    if not gigs:
        return
    
    total_gigs = len(gigs)
    avg_rating = sum(float(gig.get('rating', 0)) for gig in gigs if gig.get('rating', '0').replace('.', '').isdigit()) / total_gigs
    prices = [int(gig.get('price', '$0').replace('$', '')) for gig in gigs if gig.get('price', '$0').replace('$', '').isdigit()]
    avg_price = sum(prices) / len(prices) if prices else 0
    categories = len(set(gig.get('category', 'N/A') for gig in gigs))
    
    st.markdown(f"""
    <div class="metrics-container">
        <div class="metric-card">
            <div class="metric-value">📊 {total_gigs}</div>
            <div class="metric-label">Total Gigs Found</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">⭐ {avg_rating:.1f}</div>
            <div class="metric-label">Average Rating</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">💰 ${avg_price:.0f}</div>
            <div class="metric-label">Average Price</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">📂 {categories}</div>
            <div class="metric-label">Categories</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def process_user_input(user_input):
    """Process user input and generate response."""
    if not user_input.strip() or user_input == st.session_state.last_input:
        return
    
    st.session_state.processing = True
    st.session_state.last_input = user_input
    
    # Initialize graph if needed
    if not st.session_state.graph:
        st.session_state.graph = build_chatbot_graph()
        if not st.session_state.graph:
            st.error("❌ Failed to initialize chatbot. Please check your API key.")
            st.session_state.processing = False
            return
    
    # Add system message for first interaction
    if not st.session_state.messages:
        system_message = SystemMessage(content="""
        You are SmartGigs, an AI assistant specialized in finding and recommending Fiverr gigs.
        
        When users describe their project needs:
        1. Use chatbot_keyword_search to extract relevant keywords
        2. Use get_gigs to fetch matching gigs
        3. Present gigs in a clear, organized manner
        4. Offer to recommend the top 3 most suitable gigs
        
        Be helpful, professional, and focus on matching user needs with the best available gigs.
        """)
        st.session_state.messages.append(system_message)
    
    # Add user message
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    try:
        # Process with graph
        state = {"messages": st.session_state.messages}
        result = st.session_state.graph.invoke(state)
        
        # Update messages
        st.session_state.messages = result["messages"]
        
        # Get latest AI response
        latest_message = st.session_state.messages[-1]
        if isinstance(latest_message, AIMessage):
            st.session_state.chat_history.append({"role": "assistant", "content": latest_message.content})
            
            # Check for tool calls to extract gigs
            if hasattr(latest_message, 'tool_calls') and latest_message.tool_calls:
                for tool_call in latest_message.tool_calls:
                    if tool_call['name'] == 'get_gigs':
                        gigs = get_gigs(tool_call['args']['query'])
                        st.session_state.current_gigs = gigs
                        break
    
    except Exception as e:
        st.error(f"❌ Error processing request: {str(e)}")
    
    finally:
        st.session_state.processing = False

def main():
    """Main application function."""
    init_session_state()
    
    # Header
    st.markdown("""
    <div class="app-header">
        <h1>🎯 SmartGigs</h1>
        <p>AI-Powered Fiverr Gig Discovery Assistant</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🤖 Assistant Status")
        
        # Status indicator
        if st.session_state.processing:
            st.markdown('<div class="status-indicator status-processing">Processing your request<span class="loading-dots"></span></div>', unsafe_allow_html=True)
        elif st.session_state.current_gigs:
            st.markdown('<div class="status-indicator status-ready">Ready - Gigs loaded</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-indicator status-ready">Ready - Waiting for input</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Stats
        if st.session_state.current_gigs:
            st.markdown("### 📊 Current Search")
            st.markdown(f"**🎯 Gigs Found:** {len(st.session_state.current_gigs)}")
            categories = len(set(gig.get('category', 'N/A') for gig in st.session_state.current_gigs))
            st.markdown(f"**📂 Categories:** {categories}")
            st.markdown("---")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            for key in ["messages", "current_gigs", "awaiting_recommendation", "last_input", "chat_history"]:
                st.session_state[key] = [] if key in ["messages", "current_gigs", "chat_history"] else False if key == "awaiting_recommendation" else ""
            st.session_state.input_key += 1
            st.rerun()
    
    # Main content wrapper
    st.markdown('<div class="main-content-wrapper">', unsafe_allow_html=True)
    
    # Chat container
    if st.session_state.chat_history:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="user-message">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="ai-message">{msg["content"]}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Metrics
    if st.session_state.current_gigs:
        st.markdown('<div class="section-header">📈 Search Analytics</div>', unsafe_allow_html=True)
        display_metrics(st.session_state.current_gigs)
    
    # Gigs display
    if st.session_state.current_gigs:
        st.markdown('<div class="section-header">🎯 Available Gigs</div>', unsafe_allow_html=True)
        
        # Display in grid
        cols = st.columns(2)
        for i, gig in enumerate(st.session_state.current_gigs):
            with cols[i % 2]:
                display_gig_card(gig)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # TRULY FIXED INPUT BAR
    st.markdown('<div class="fixed-input-bar">', unsafe_allow_html=True)
    
    # Use form for proper handling
    with st.form(key=f"chat_form_{st.session_state.input_key}", clear_on_submit=True):
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([5, 1])
        
        with col1:
            user_input = st.text_input(
                "message",
                placeholder="💬 Describe your project... e.g., 'I need a website for my restaurant'",
                label_visibility="collapsed",
                disabled=st.session_state.processing
            )
        
        with col2:
            send_clicked = st.form_submit_button(
                "🚀 Send" if not st.session_state.processing else "⏳",
                disabled=st.session_state.processing,
                use_container_width=True
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process input
    if send_clicked and user_input.strip():
        process_user_input(user_input)
        st.rerun()

if __name__ == "__main__":
    main()
