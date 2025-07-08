from langchain_google_genai import ChatGoogleGenerativeAI
import os
import json
from dotenv import load_dotenv

load_dotenv()

def GetThingsFromInput(input_query: str) -> list:
    """
    Extract niche, location, platform, and number from user input using a single LLM request.
    
    Args:
        input_query (str): The user's input text
    
    Returns:
        list: List containing [niche, location, platform, number]. Each element is a string,
              empty string if not found.
    """
    # Combined prompt for all extraction tasks
    combined_prompt = f"""
    Extract the following information from this text: "{input_query}"
    
    Return the results as a JSON object in the following format:
    {{"niche": "<keyword or phrase>", "location": "<country name>", "platform": "<platform codes>", "number": "<number>"}}
    
    Rules for extraction:
    1. Niche:
       - If the niche is explicitly specified in the query (e.g., "AI Tools", "machine learning", "digital marketing"), return the exact phrase as mentioned.
       - If the niche is not specified or unclear, extract the most relevant niche/category, which can be a single word or multi-word phrase (e.g., fitness, food, technology, AI Tools, digital marketing, machine learning).
       - If no clear niche is found, return "general".
       - Return only the keyword or phrase, nothing else.
    
    2. Location:
       - Extract the exact country name (e.g., united-states, united-kingdom, canada, australia, germany, france, india, etc.).
       - If no location is mentioned, return empty string.
    
    3. Platform:
       - Extract social media platforms and map to these codes:
         - Instagram: INST
         - Facebook: FB
         - Twitter/X: TW
         - YouTube: YT
         - TikTok: TT
         - Telegram: TG
       - If multiple platforms are mentioned, return them comma-separated (e.g., "INST,YT,TT").
       - If no platforms are mentioned, return empty string.
    
    4. Number:
       - Extract the number of results requested (e.g., from phrases like "find 10 influencers", "get 5 creators", "show 20 people").
       - Return only the number as a string.
       - If no number is mentioned, return empty string.
    
    Return only the JSON object as a string, without any Markdown, code fences, or additional text.
    """
    
    try:
        # Initialize LangChain ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",  # Updated to a valid model
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.1,  # Low temperature for consistent results
            max_output_tokens=200  # Increased to accommodate JSON response
        )
        
        # Initialize result list [niche, location, platform, number]
        results = ["", "", "", ""]
        
        # Get response from LLM
        response = llm.invoke(combined_prompt)
        result = response.content.strip()
        
        # Debug: Print raw LLM response
        print(f"Raw LLM response: {result}")
        
        # Check if response is empty
        if not result:
            print("Error: LLM response is empty")
            return results
        
        # Strip Markdown code fences if present
        cleaned_result = result
        if result.startswith("```json"):
            cleaned_result = result.replace("```json", "").replace("```", "").strip()
        
        # Parse JSON response
        try:
            parsed_result = json.loads(cleaned_result)
        except json.JSONDecodeError as e:
            print(f"Error parsing LLM response as JSON: {str(e)}")
            print(f"Cleaned response: {cleaned_result}")
            return results
        
        # Post-process the results
        # Niche
        niche = parsed_result.get('niche', '')
        if niche:
            results[0] = niche.lower()  # Preserve multi-word phrases
        
        # Location
        results[1] = parsed_result.get('location', '')
        
        # Platform
        platform = parsed_result.get('platform', '')
        if platform:
            valid_codes = {'INST', 'FB', 'TW', 'YT', 'TT', 'TG'}
            codes = [code.strip() for code in platform.split(',') if code.strip()]
            valid_codes_found = [code for code in codes if code in valid_codes]
            results[2] = ','.join(valid_codes_found) if valid_codes_found else ""
        
        # Number
        number = parsed_result.get('number', '')
        if number.isdigit():
            results[3] = number
        
        return results
        
    except Exception as e:
        print(f"Error extracting information: {str(e)}")
        return ["", "", "", ""]

