import json
import random
from typing import Dict, List, Optional, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
import os
from IPython.display import Image, display

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

import random
from typing import List, Dict
from datetime import datetime

def generate_dummy_gigs(num_gigs: int = 100) -> List[Dict[str, str]]:
    # Expanded list of services with more variety
    services = [
    # Web Development
        "design a responsive website",
        "develop a web application",
        "create a WordPress site",
        "build an ecommerce platform",
        "design a landing page",
        "convert PSD to HTML",
        "build a MERN stack app",
        "create a blog",
        "design a mobile-friendly site",
        "optimize website speed",
        "develop a custom CMS",
        "create an admin dashboard",
        "build a progressive web app (PWA)",
        "migrate a website to a new platform",
        "develop a single-page application (SPA)",
        "create a multilingual website",
        "set up a headless CMS",
        "integrate APIs into a website",
        "build a SaaS platform",
        "create a membership website",

        # Mobile Development
        "build a mobile app",
        "build a cross-platform app",
        "develop an iOS app",
        "develop an Android app",
        "create a mobile game",
        "design a mobile app UI/UX",
        "optimize mobile app performance",
        "integrate push notifications in a mobile app",
        "build a Flutter app",
        "build a React Native app",

        # Graphic Design
        "create a logo",
        "design promotional banners",
        "create social media graphics",
        "design email templates",
        "create a brand identity kit",
        "design business cards",
        "create infographics",
        "design brochures",
        "create presentation slides",
        "design product packaging",
        "create vector illustrations",
        "design print media ads",
        "create 3D mockups",

        # UI/UX Design
        "design a UI/UX interface",
        "create wireframes for a website",
        "design a mobile app prototype",
        "conduct user experience research",
        "create interactive prototypes",
        "design a dashboard UI",
        "optimize user flows",

        # Content Creation
        "write content for blogs",
        "write SEO-friendly articles",
        "create social media content",
        "write product descriptions",
        "craft email marketing copy",
        "write technical documentation",
        "create video scripts",
        "write press releases",
        "edit and proofread content",
        "translate content to another language",

        # SEO & Digital Marketing
        "optimize SEO",
        "perform keyword research",
        "set up Google Ads campaigns",
        "manage social media ads",
        "create a content marketing strategy",
        "optimize local SEO",
        "build backlinks for SEO",
        "analyze website analytics",
        "set up email marketing funnels",

        # Game Development
        "develop a game",
        "create a 2D game",
        "develop a 3D game",
        "design game assets",
        "build a multiplayer game",
        "create a VR/AR game",
        "optimize game performance",

        # Backend & DevOps
        "set up a backend",
        "develop a custom API",
        "set up cloud infrastructure",
        "perform database optimization",
        "integrate payment gateways",
        "set up a CI/CD pipeline",
        "configure serverless architecture",
        "deploy a website to a cloud server",
        "set up Docker containers",
        "configure Kubernetes clusters",
        "migrate databases",
        "set up load balancers",

        # AI & Machine Learning
        "create a chatbot",
        "build a machine learning model",
        "develop an AI-powered application",
        "create a recommendation system",
        "perform data analysis with AI",
        "train a computer vision model",
        "build a natural language processing tool",

        # Animation & Video
        "create an animated video",
        "produce a promotional video",
        "create motion graphics",
        "edit a video",
        "create 3D animations",
        "produce a whiteboard animation",
        "add subtitles to a video",

        # Cybersecurity
        "perform a security audit",
        "secure a website",
        "set up two-factor authentication",
        "conduct penetration testing",
        "fix vulnerabilities in code",
        "set up a firewall",

        # Miscellaneous
        "fix bugs in code",
        "perform website maintenance",
        "create a custom CRM",
        "build a booking system",
        "develop an inventory management system",
        "create a virtual tour",
        "set up an online learning platform",
        "build a real-time chat application",
        "create a data visualization dashboard",
        "set up an IoT application",
        "develop a blockchain-based application",
        "create a custom WordPress plugin",
        "build a Shopify app",
        "create a Magento extension"
    ]
    
    # Expanded frameworks and tools
    frameworks = [
        "React.js", "Django", "Flask", "Node.js", "Angular", "Vue.js", "Laravel", "WordPress", "Flutter",
        "React Native", "Spring Boot", "Ruby on Rails", "Express.js", "Svelte", "Next.js", "Gatsby",
        "Shopify", "Magento", "Unity", "Unreal Engine"
    ]
    
    # Expanded list of sellers with more realistic names
    sellers = [
        f"{first}_{last}" for first in ["Alex", "Emma", "Liam", "Olivia", "Noah", "Sophia", "James", "Ava", "William", "Isabella"]
        for last in ["Smith", "Johnson", "Brown", "Taylor", "Wilson", "Davis", "Clark", "Harris", "Lewis", "Walker"]
    ]
    
    # Categories for gigs
    categories = [
        "Web Development", "Mobile Development", "Graphic Design", "UI/UX Design", "SEO & Marketing",
        "Content Writing", "Game Development", "Backend Development", "Frontend Development", "Ecommerce Solutions",
        "Cloud Services", "Animation", "Database Management"
    ]
    
    # Tags for additional metadata
    tags = [
        "responsive", "modern", "fast-delivery", "high-quality", "custom", "professional", "SEO-friendly",
        "mobile-first", "ecommerce", "scalable", "user-friendly", "minimalist", "creative", "optimized",
        "cross-platform", "secure"
    ]
    
    # Experience levels
    experience_levels = ["Beginner", "Intermediate", "Expert"]
    
    # Delivery time options
    delivery_times = ["3 days", "5 days", "7 days", "10 days", "14 days", "21 days"]
    
    # Revision options
    revisions = ["1 revision", "2 revisions", "3 revisions", "Unlimited revisions"]
    
    gigs = []
    for i in range(num_gigs):
        service = random.choice(services)
        # Only include framework if the service is development-related
        framework = random.choice(frameworks) if any(keyword in service for keyword in ["develop", "build", "app", "website", "backend", "game", "API"]) else ""
        title = f"I will {service} {framework}".strip()
        
        # Generate realistic price based on service complexity
        base_price = random.randint(20, 1000)
        if "expert" in service.lower() or "advanced" in service.lower():
            base_price += random.randint(100, 500)
        elif "basic" in service.lower() or "simple" in service.lower():
            base_price = max(20, base_price - random.randint(10, 100))
        
        # Generate realistic rating and reviews
        rating = round(random.uniform(3.5, 5.0), 1)
        reviews = random.randint(5, 1000) if rating >= 4.0 else random.randint(0, 50)
        
        # Select relevant category based on service
        relevant_categories = [cat for cat in categories if any(keyword in service.lower() for keyword in cat.lower().split())]
        category = random.choice(relevant_categories) if relevant_categories else random.choice(categories)
        
        # Select random tags (2-5 tags per gig)
        gig_tags = random.sample(tags, random.randint(2, 5))
        
        gig = {
            "gig_id": f"GIG{str(i+1).zfill(4)}",
            "title": title,
            "seller": random.choice(sellers),
            "category": category,
            "tags": ", ".join(gig_tags),
            "rating": f"{rating:.1f}",
            "reviews": str(reviews),
            "price": f"${base_price}",
            "delivery_time": random.choice(delivery_times),
            "experience_level": random.choice(experience_levels),
            "revisions": random.choice(revisions),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        gigs.append(gig)
    
    return gigs

@tool
def get_gigs(query: str) -> List[Dict[str, str]]:
    """Fetches a list of Fiverr gigs from a JSON file based on the user's query."""
    gigs_file = "gigs.json"
    
    # Check if file exists
    if not os.path.exists(gigs_file):
        raise FileNotFoundError(f"The file {gigs_file} does not exist.")
    
    # Read the JSON file
    with open(gigs_file, "r") as f:
        gigs_data = json.load(f)
    
    print(f"Fetching gigs for query: {query}")
    
    # Extract gigs from the JSON data
    gigs = gigs_data.get("gig_list", {}).get("gigs", [])
    
    # Filter gigs where query matches title or sub_category_name (case-insensitive)
    filtered_gigs = [
        gig for gig in gigs 
        if query.lower() in gig["title"].lower() 
        or query.lower() in gig["sub_category_name"].lower()
    ]
    
    # Return filtered gigs, or up to 10 gigs if no matches found
    return filtered_gigs or gigs[:10]

@tool
def chatbot_keyword_search(project_details: str) -> str:
    """
    Chatbot function to handle keyword search for Fiverr gigs.
    """
    # Invoke the LLM with the project details
    response = llm.invoke(f"project_details: {project_details} \n\nPlease provide the most relevant and most suitable Fiverr keyword based on the above project details. in your response there should be format like \"find the <keyword> gigs\" and nothing else.")
    print(f"Keyword search response: {response.content}")
    return response.content

tools = [get_gigs, chatbot_keyword_search]
llm_with_tools = llm.bind_tools(tools=tools)

class State(TypedDict):
    """
    State for the specific Fiverr Gigs Fetcher Agent.
    """
    messages: Annotated[List[BaseMessage], add_messages]

def chatbot(state: State):
    """
    Chatbot function to handle the conversation with the user.
    """
    # Invoke the LLM with all messages
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def build_chatbot_graph():
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
    
    graph = builder.compile()
    return graph

def display_graph(graph):
    try:
        png_bytes = graph.get_graph().draw_mermaid_png()
        with open("graph_output.png", "wb") as f:
            f.write(png_bytes)
        print("Graph saved as graph_output.png. Open it manually to view.")
    except Exception as e:
        print(f"Could not save graph: {e}")

def run_chatbot():
    """
    Run the chatbot with an initial state.
    """
    graph = build_chatbot_graph()
    system_message = SystemMessage(content="""
            # Fiverr Gig Assistant System Prompt
        
        ## Primary Role
        You are a helpful assistant specialized in fetching and recommending Fiverr gigs based on user queries.
        
        ## Core Functionality
        
        ### General Queries
        - For general questions unrelated to Fiverr projects, respond gracefully with helpful information
        - Maintain a conversational and supportive tone
        
        ### Fiverr-Related Queries
        When users provide:
        - Specific project details
        - Keywords related to Fiverr services
        - Requests in the format "find the [keyword] gigs"
        - Detailed project descriptions (paragraphs explaining what they want done)
        
        **Then you should:**
        1. Use the `chatbot_keyword_search` tool for keyword processing
        2. Use the `get_gigs` tool to fetch relevant gigs
        3. Pass the user's project details or keywords to the appropriate tool
        
        ## Gig Display Process
        
        ### Step 1: Display Fetched Gigs
        - Present all fetched gigs in an **attractive, well-formatted display**
        - Include **all available details** for each gig
        - Use clear headings, bullet points, and organized layout
        
        ### Step 2: Offer Personalized Recommendations
        After displaying the gigs, ask:
        > "Would you like me to analyze these gigs and recommend the 3 most suitable ones for your specific project?"
        
        ### Step 3: Handle User Response
        **If user says YES:**
        - Compare all fetched gigs against the user's project requirements
        - Analyze factors like: relevance, seller ratings, pricing, delivery time, reviews, etc.
        - Display the **3 most suitable gigs** with detailed explanations of why they're recommended in a very clear and organized manner
        
        **If user says NO:**
        - End the conversation gracefully
        - Offer assistance with other queries if needed
        
        ## Additional Support
        
        ### Gig Details
        - If user requests more information about a specific gig, provide comprehensive details
        - Include seller information, package options, reviews, delivery times, etc.
        
        ### New Searches
        - If user asks for different keywords or new project types, restart the process
        - Use the same systematic approach for each new query
        
        ## Response Style
        - Be professional yet friendly
        - Use clear, organized formatting
        - Focus on being helpful and efficient
        - Provide actionable information
        """)
    # Initialize the state with the system message
    state: State = {
        "messages": [system_message]
    }
    display_graph(graph)

    while True:
        user_input = input("Enter your query: ")
        if user_input.lower() == "exit":
            print("Exiting the chatbot.")
            break
        
        # Add user message to state
        state["messages"].append(HumanMessage(content=user_input))
        
        # Run the graph
        result = graph.invoke(state)
        
        # Update state with the result
        state = result
        
        # Print the latest AI response
        latest_message = state["messages"][-1]
        if isinstance(latest_message, AIMessage):
            print("Response:", latest_message.content)
        else:
            print("Response: No response generated")
    
    return state

if __name__ == "__main__":
    history = run_chatbot()
    print(f"\nChat history:\n")
    for message in history["messages"]:
        if isinstance(message, HumanMessage):
            print(f"You: {message.content}")
        elif isinstance(message, AIMessage):
            print(f"AI: {message.content}\n")