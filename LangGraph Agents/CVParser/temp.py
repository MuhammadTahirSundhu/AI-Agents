import json
import fitz
import re
import os
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="CV Parsing Agent", description="API for analyzing and ranking CVs")

# Pydantic models for request/response
class ChatMessage(BaseModel):
    role: str
    content: str

class InfluencerFinderRequest(BaseModel):
    input_query: str
    ChatHistory: List[ChatMessage] = []
    user_id: int

class InfluencerFinderResponse(BaseModel):
    chat_history: List[ChatMessage]
    response: str

def convert_dict_to_langchain_messages(chat_history_dict: List[ChatMessage]):
    """Convert dictionary format chat history to LangChain message objects."""
    messages = []
    for msg in chat_history_dict:
        if msg.role in ["user", "human"]:
            messages.append(HumanMessage(content=msg.content))
        elif msg.role in ["assistant", "ai"]:
            messages.append(AIMessage(content=msg.content))
        elif msg.role == "system":
            messages.append(SystemMessage(content=msg.content))
    return messages

def convert_langchain_to_dict(chat_history):
    """Convert LangChain message objects to dictionary format."""
    result = []
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            result.append(ChatMessage(role="user", content=msg.content))
        elif isinstance(msg, AIMessage):
            result.append(ChatMessage(role="assistant", content=msg.content))
        elif isinstance(msg, SystemMessage):
            result.append(ChatMessage(role="system", content=msg.content))
    return result

# Intelligent CV Ranking Prompt Template
intelligent_cv_ranking_prompt = PromptTemplate(
    input_variables=["job_description", "cvs_data", "topk"],
    template="""
You are an expert HR professional and technical recruiter with deep understanding of job requirements and candidate evaluation. 

**JOB DESCRIPTION:**
{job_description}

**CANDIDATE CVs DATA:**
{cvs_data}

**YOUR TASK:**
Analyze the job description to intelligently understand what the role requires, then evaluate and rank {topk} candidates based on their fit for this specific position.

**IMPORTANT - CV PARSING INTELLIGENCE:**
CVs come in many formats and use different terminology. Use your expertise to intelligently identify relevant information regardless of how it's structured or labeled. Be flexible in understanding:

- Different section names and organizational styles
- Varying terminology for similar concepts and roles
- Skills mentioned in project descriptions vs. dedicated skills sections
- Experience described in different contexts (internships, freelance, projects, volunteer work)
- Technical competencies demonstrated through practical applications
- Academic and professional achievements in various formats

Apply your professional judgment to extract meaningful insights from each CV's unique presentation style.

**ANALYSIS PROCESS:**
1. **Job Analysis Phase:**
   - Extract key technical skills required
   - Identify required experience level and type
   - Understand the role's primary responsibilities
   - Determine must-have vs nice-to-have qualifications
   - Identify any domain-specific requirements
   - Assess required soft skills and attributes

2. **Candidate Evaluation Phase:**
   - Apply your expertise to understand each CV's unique format and terminology
   - Extract relevant information regardless of section names or presentation style
   - Recognize skills and experience mentioned throughout the document, not just in dedicated sections
   - Understand the context and depth of experience from project descriptions and achievements
   - Assess the candidate's capabilities based on what they've actually accomplished
   - Consider the progression and growth shown in their career/academic journey
   - Evaluate both technical competencies and soft skills demonstrated through their experiences

3. **Intelligent Ranking Phase:**
   - Score candidates holistically (not just checklist matching)
   - Consider potential for growth and learning
   - Evaluate problem-solving and technical depth
   - Assess real-world application of skills
   - Factor in leadership and collaboration indicators

**OUTPUT FORMAT:**

## JOB ANALYSIS
**Role Summary:** [Brief understanding of the role]
**Key Requirements Identified:**
- **Must-Have Skills:** [List extracted must-have skills]
- **Nice-to-Have Skills:** [List preferred skills]
- **Experience Level:** [Junior/Mid/Senior level expected]
- **Core Responsibilities:** [Main job functions]
- **Success Factors:** [What makes someone successful in this role]

## CANDIDATE RANKINGS

### 🥇 RANK 1: "Place the orignal candidate name here"
**Overall Fit Score:** [Score]/100
**Why This Candidate Wins:**
- **Perfect Match Areas:** [Specific alignments with job needs]
- **Standout Qualities:** [What makes them exceptional for this role]
- **Growth Potential:** [How they can evolve in this position]

**Detailed Assessment:**
- **Technical Alignment:** [Score]/10 - [Explanation]
- **Experience Relevance:** [Score]/10 - [Explanation]
- **Project Quality:** [Score]/10 - [Explanation]
- **Learning Agility:** [Score]/10 - [Explanation]
- **Role Readiness:** [Score]/10 - [Explanation]

**Key Strengths:**
• [Strength 1 with specific example]
• [Strength 2 with specific example]
• [Strength 3 with specific example]

**Potential Concerns:**
• [Any gaps or areas to explore in interview]

---

### 🥈 RANK 2: "Place the orignal candidate name here"
**Overall Fit Score:** [Score]/100
**Why This Candidate is Strong:**
- **Main Strengths:** [Key selling points]
- **Unique Value:** [What they bring that others don't]
- **Development Areas:** [Where they need to grow]

**Detailed Assessment:**
- **Technical Alignment:** [Score]/10 - [Explanation]
- **Experience Relevance:** [Score]/10 - [Explanation]
- **Project Quality:** [Score]/10 - [Explanation]
- **Learning Agility:** [Score]/10 - [Explanation]
- **Role Readiness:** [Score]/10 - [Explanation]

**Key Strengths:**
• [Strength 1 with specific example]
• [Strength 2 with specific example]

**Potential Concerns:**
• [Any gaps or areas to explore in interview]

---

### 🥉 RANK 3: "Place the orignal candidate name here"
**Overall Fit Score:** [Score]/100
**Assessment Summary:**
- **Strengths:** [What they do well]
- **Fit Analysis:** [How they match the role]
- **Development Needs:** [Where they need support]

**Detailed Assessment:**
- **Technical Alignment:** [Score]/10 - [Explanation]
- **Experience Relevance:** [Score]/10 - [Explanation]
- **Project Quality:** [Score]/10 - [Explanation]
- **Learning Agility:** [Score]/10 - [Explanation]
- **Role Readiness:** [Score]/10 - [Explanation]

---
and like that upto {topk}
Would you like to give other job description. If yes please provide details?
"""
)

def call_LLM_for_CV_analyzation(job_description: str, cvs_data: str, topk: int) -> str:
    """Make a call to OpenAI API with a properly formatted message"""
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        prompt = intelligent_cv_ranking_prompt.format(
            job_description=job_description,
            cvs_data=cvs_data,
            topk=topk
        )
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert HR professional and technical recruiter."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ OpenAI API error: {str(e)}")
        return f"Error analyzing CVs: {str(e)}"

def extract_pdfs_to_markdown(folder_path: str) -> List[Dict[str, str]]:
    def process_text_block(block: Dict[str, Any]) -> str:
        lines = []
        for line in block["lines"]:
            line_text = ""
            for span in line["spans"]:
                text = span["text"]
                if not text.strip():
                    continue
                font_size = span["size"]
                font_flags = span["flags"]
                font_name = span["font"].lower()
                
                # Apply formatting based on font properties
                formatted_text = text
                if font_size > 16:
                    formatted_text = f"# {text}"
                elif font_size > 14:
                    formatted_text = f"## {text}"
                elif font_size > 12:
                    formatted_text = f"### {text}"
                
                if font_flags & 16:
                    formatted_text = f"**{formatted_text}**"
                if font_flags & 2:
                    formatted_text = f"*{formatted_text}*"
                if any(mono in font_name for mono in ['mono', 'courier', 'consolas', 'code']):
                    formatted_text = f"`{formatted_text}`"
                
                line_text += formatted_text
            if line_text.strip():
                lines.append(line_text.strip())
        return "\n".join(lines)

    def clean_markdown(text: str) -> str:
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        text = re.sub(r'\n(#{1,6})\s*(.+)', r'\n\n\1 \2\n', text)
        text = '\n'.join(line.rstrip() for line in text.split('\n'))
        return text.strip()

    def extract_pdf_advanced(pdf_path: str) -> str:
        try:
            doc = fitz.open(pdf_path)
            markdown_content = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                blocks = page.get_text("dict")
                
                if page_num > 0:
                    markdown_content.append("\n---\n")
                
                for block in blocks["blocks"]:
                    if "lines" in block:
                        block_text = process_text_block(block)
                        if block_text.strip():
                            markdown_content.append(block_text)
            
            doc.close()
            full_text = "\n\n".join(markdown_content)
            return clean_markdown(full_text)
        except Exception as e:
            return f"Error extracting PDF {pdf_path}: {str(e)}"

    def extract_pdf_simple(pdf_path: str) -> str:
        try:
            doc = fitz.open(pdf_path)
            text_content = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                if page_num > 0:
                    text_content.append("\n---\n")
                
                lines = text.split('\n')
                formatted_lines = []
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    if len(line) < 100 and line.isupper() and len(line) > 3:
                        formatted_lines.append(f"## {line}")
                    else:
                        formatted_lines.append(line)
                
                text_content.append('\n\n'.join(formatted_lines))
            
            doc.close()
            return '\n\n'.join(text_content)
        except Exception as e:
            return f"Error extracting PDF {pdf_path}: {str(e)}"

    results = []
    if not os.path.isdir(folder_path):
        return [{"filename": "", "content": f"Error: '{folder_path}' is not a valid directory"}]
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            try:
                markdown_text = extract_pdf_advanced(pdf_path)
                
                if markdown_text.startswith("Error"):
                    print(f"Advanced extraction failed for {filename}, trying simple extraction...")
                    markdown_text = extract_pdf_simple(pdf_path)
                
                results.append({
                    "filename": filename,
                    "content": markdown_text
                })
                
            except Exception as e:
                results.append({
                    "filename": filename,
                    "content": f"An error occurred: {str(e)}"
                })
    
    return results

def GetThingsFromInput(input_query: str) -> list:
    """
    Extract job description, topk value, and folder path from user input using a single LLM request.
    """
    combined_prompt = f"""
    Analyze this text and extract relevant information: "{input_query}"
    
    Your task is to identify and extract:
    1. Any numerical value that represents a quantity, limit, or count of items (topk_value)
    2. Any reference to file locations, storage paths, directories, or data sources (folder_path)
    3. Everything else should be considered the job description
    
    Guidelines:
    - For topk_value: Look for any number that suggests how many items, results, or entries the user wants
    - For folder_path: Look for any file paths, directory references, or storage locations (like /home/..., ../..., C:\..., etc.)
    - For job_description: Take the remaining text after removing the topk_value and folder_path elements. This should include all job-related requirements, skills, technologies, company details, and any other descriptive text
    
    The job_description should be the original text with only the specific numerical quantity and file paths removed, preserving the natural flow and context of the job requirements.
    
    Return the results as a JSON object in the following format:
    {{"job_description": "<all remaining text after removing topk and paths>", "topk_value": "<extracted number>", "folder_path": "<extracted path>"}}
    
    Important:
    - If topk_value or folder_path aren't present, return empty string for those fields
    - The job_description should read naturally and include all context about the job requirements
    - Don't summarize or extract keywords - preserve the full descriptive text
    - If any parameter value not found just place the parameter value as ""
    
    Return only the JSON object as a string, without any Markdown, code fences, or additional text.
    """
    
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.1,
            max_output_tokens=500
        )
        
        results = ["", "", ""]
        
        response = llm.invoke(combined_prompt)
        result = response.content.strip()
        
        if not result:
            print("Error: LLM response is empty")
            return results
        
        # Strip Markdown code fences if present
        cleaned_result = result
        if result.startswith("```json"):
            cleaned_result = result.replace("```json", "").replace("```", "").strip()
        elif result.startswith("```"):
            cleaned_result = result.replace("```", "").strip()
        
        try:
            parsed_result = json.loads(cleaned_result)
        except json.JSONDecodeError as e:
            print(f"Error parsing LLM response as JSON: {str(e)}")
            return results
        
        # Extract results
        job_description = parsed_result.get('job_description', '')
        if job_description:
            results[0] = job_description.strip()
        
        topk_value = parsed_result.get('topk_value', '')
        if topk_value:
            import re
            numbers = re.findall(r'\d+', str(topk_value))
            if numbers:
                results[1] = numbers[0]
            else:
                results[1] = str(topk_value).strip()
        
        folder_path = parsed_result.get('folder_path', '')
        if folder_path:
            results[2] = folder_path.strip()
        
        return results
        
    except Exception as e:
        print(f"Error extracting information: {str(e)}")
        return ["", "", ""]

CV_PARSING_AGENT_PROMPT = PromptTemplate(
    input_variables=["user_input", "conversation_history", "job_description", "topk_value", "folder_path"],
    template="""
You are an intelligent CV Parsing Agent designed to help users analyze and rank CVs against job requirements. Your primary goal is to collect necessary parameters and perform CV analysis efficiently.

## Core Responsibilities:
1. **Parameter Collection**: Gather job description, topk value, and folder path
2. **General Query Handling**: Answer questions about CV parsing, recruitment, and related topics and other topics
3. **Workflow Guidance**: Guide users through the CV analysis process
4. **Function Execution**: Call the Analyze CVs function when all parameters are complete

## Current Status:
- Job Description: {job_description}
- Top K Value: {topk_value}
- Folder Path: {folder_path}

## Conversation History:
{conversation_history}

## User Input:
{user_input}

## Response Guidelines:

### 1. Parameter Collection Priority
If any required parameters are missing, prioritize collecting them:

**Missing Job Description:**
- Ask: "To analyze CVs effectively, I need the job description. Please provide the job requirements, skills, and qualifications you're looking for."
- Encourage detailed job descriptions for better matching accuracy

**Missing Top K Value:**
- Ask: "How many top-ranked CVs would you like me to return? Please specify a number (e.g., 5, 10, 15)."
- Suggest reasonable ranges: "Typically, users request between 5-20 top matches."

**Missing Folder Path:**
- Ask: "Please provide the folder path where your CV files are stored (e.g., '/path/to/cv/folder' or 'C:\\CVs\\')."
- Remind about supported formats: "Ensure your folder contains PDF, DOC, or DOCX files."

### 2. General Query Handling
For non-parameter questions, provide helpful information about:
- Gracefully answer the question
- CV parsing techniques and best practices
- Recruitment process optimization
- Skills matching algorithms
- Document format requirements
- Tips for better job descriptions

### 3. Workflow Guidance
When users ask about the process:
- Explain the CV analysis workflow
- Describe how matching algorithms work
- Provide tips for organizing CV folders
- Suggest best practices for job descriptions

### 4. Function Execution
When ALL parameters are complete:
- Confirm the parameters with the user
- Explain what the analysis will do
- Indicate readiness to analyze CVs
- Format: "I'll now analyze the CVs with these parameters: Job Description: [summary], Top K: {topk_value}, Folder: {folder_path}"

## Response Format:
- Be conversational and helpful
- Use clear, actionable language
- Provide specific examples when helpful
- Maintain professional tone
- Always acknowledge user inputs

## Special Cases:
- If user wants to modify existing parameters, update accordingly
- If folder path doesn't exist, suggest checking the path
- If topk value is unrealistic, suggest reasonable alternatives
- If job description is too vague, ask for more details
- If user ask about your capablities, explain your role to him

Now, based on the current conversation state and user input, provide an appropriate response that follows these guidelines.
"""
)

def get_cvs_data(folder_path: str) -> str:
    """Extract CV data from the specified folder path."""
    print(f"Extracting CVs from folder: {folder_path}")
    
    if not os.path.exists(folder_path):
        return f"Error: Folder '{folder_path}' does not exist."
    
    extracted_contents = extract_pdfs_to_markdown(folder_path)
    
    if not extracted_contents:
        return "Error: No PDF files found in the specified folder."
    
    all_contents = []
    for result in extracted_contents:
        all_contents.append("=" * 200)
        all_contents.append(f"FILENAME: {result['filename']}")
        all_contents.append("=" * 200)
        all_contents.append(result["content"])

    return "\n".join(all_contents)

def call_llm(prompt: str, chat_history: List[ChatMessage]) -> str:
    """Call the OpenAI API with GPT-4o model using a prompt and chat history."""
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Convert chat history to OpenAI format
        messages = [
            {"role": "system", "content": """You are an ethical AI CV Parsing Agent designed to assist users in processing job descriptions and candidate data for recruitment purposes. 
            Your primary function is to extract key information from user input, including job descriptions, numerical values (e.g., number of candidates), and folder paths for candidate databases. 
            You have access to a function for parsing input data and must process results accurately before responding. 
            Your responses must be precise, transparent, and respect the privacy of candidate information. 
            Interpret user intent, suggest relevant actions when unclear, and prompt for missing parameters such as job description, candidate count, or folder paths."""}
        ]
        
        # Add chat history
        for msg in chat_history:
            if msg.role == "user":
                messages.append({"role": "user", "content": msg.content})
            elif msg.role == "assistant":
                messages.append({"role": "assistant", "content": msg.content})
        
        # Add current prompt
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.5
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"OpenAI API failed: {str(e)}")
        return f"Error: Failed to connect to OpenAI API - {str(e)}"

def format_conversation_history(chat_history: List[ChatMessage]) -> str:
    """Format chat history for the prompt template."""
    if not chat_history:
        return "No previous conversation."
    
    formatted_history = []
    for msg in chat_history:
        role = "User" if msg.role == "user" else "Assistant"
        formatted_history.append(f"{role}: {msg.content}")
    
    return "\n".join(formatted_history)

@app.post("/analyze_cvs", response_model=InfluencerFinderResponse)
async def analyze_cvs(request: InfluencerFinderRequest):
    """
    Main endpoint for CV analysis and chat interaction.
    """
    try:
        # Extract parameters from user input
        job_description, topk_value, folder_path = GetThingsFromInput(request.input_query)
        
        # Convert topk to integer if possible
        topk_int = 0
        if topk_value:
            try:
                topk_int = int(topk_value)
            except ValueError:
                topk_int = 0
        
        # Format conversation history
        conversation_history = format_conversation_history(request.ChatHistory)
        
        # Create prompt for the CV parsing agent
        prompt = CV_PARSING_AGENT_PROMPT.format(
            user_input=request.input_query,
            conversation_history=conversation_history,
            job_description=job_description if job_description else "Not provided",
            topk_value=topk_value if topk_value else "Not provided",
            folder_path=folder_path if folder_path else "Not provided"
        )
        
        # Get response from LLM
        response = call_llm(prompt, request.ChatHistory)
        
        # Check if we should analyze CVs (all parameters present)
        if (job_description and topk_value and folder_path and 
            "I'll now analyze" in response):
            
            # Get CV data
            cvs_data = get_cvs_data(folder_path)
            
            if cvs_data and not cvs_data.startswith("Error"):
                # Perform CV analysis
                cv_analysis = call_LLM_for_CV_analyzation(job_description, cvs_data, topk_int)
                response = cv_analysis
            else:
                response = f"Error accessing CV data: {cvs_data}"
        
        # Update chat history
        updated_history = request.ChatHistory.copy()
        updated_history.append(ChatMessage(role="user", content=request.input_query))
        updated_history.append(ChatMessage(role="assistant", content=response))
        
        return InfluencerFinderResponse(
            chat_history=updated_history,
            response=response
        )
        
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        
        # Still update chat history with error
        updated_history = request.ChatHistory.copy()
        updated_history.append(ChatMessage(role="user", content=request.input_query))
        updated_history.append(ChatMessage(role="assistant", content=error_message))
        
        return InfluencerFinderResponse(
            chat_history=updated_history,
            response=error_message
        )

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "CV Parsing Agent API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "CV Parsing Agent"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)