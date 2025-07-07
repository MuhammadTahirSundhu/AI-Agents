import json
import streamlit as st
import requests
import os
from dotenv import load_dotenv
from typing import List, Dict, Any, TypedDict, Annotated
from dataclasses import dataclass
import dropbox
from dropbox.exceptions import AuthError, ApiError
import tempfile
import io
import PyPDF2

import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
from pdf2image import convert_from_bytes
import pytesseract
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import logging
from datetime import datetime
import re
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Configuration
DROPBOX_ACCESS_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN")  # Your Dropbox access token
DROPBOX_FOLDER_PATH = os.getenv("DROPBOX_FOLDER_PATH", "/CVs")  # Your Dropbox folder path containing CVs

# Initialize sentence transformer for semantic similarity
@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer('all-MiniLM-L6-v2')

# State definition for LangGraph
class CVMatchingState(TypedDict):
    job_requirements: str
    extracted_keywords: List[Dict[str, Any]]  # Updated to include weights
    cv_documents: List[Document]
    candidate_profiles: List[Dict[str, Any]]
    similarity_scores: List[Dict[str, Any]]
    ranked_candidates: List[Dict[str, Any]]
    messages: Annotated[List[Any], add_messages]
    errors: List[str]

@dataclass
class CandidateProfile:
    name: str
    skills: List[str]
    experience: str
    experience_years: float  # New field for extracted years of experience
    education: str
    raw_text: str
    file_path: str
    semantic_embedding: np.ndarray = None

class SkillSynonymMapper:
    """Maps skills to their synonyms for better matching"""
    
    def __init__(self):
        self.skill_synonyms = {
            # Programming Languages
            'python': ['python', 'py', 'python3', 'python programming'],
            'javascript': ['javascript', 'js', 'node.js', 'nodejs', 'ecmascript'],
            'java': ['java', 'java8', 'java11', 'java17', 'spring', 'spring boot'],
            'c++': ['c++', 'cpp', 'c plus plus', 'c/c++'],
            'c#': ['c#', 'csharp', 'c sharp', '.net', 'dotnet'],
            'sql': ['sql', 'mysql', 'postgresql', 'sqlite', 'mssql', 'oracle', 'database'],
            'r': ['r', 'r programming', 'rstudio'],
            'go': ['go', 'golang', 'go programming'],
            'rust': ['rust', 'rust programming'],
            'php': ['php', 'php7', 'php8', 'laravel', 'symfony'],
            # Web Technologies
            'html': ['html', 'html5', 'hypertext markup language'],
            'css': ['css', 'css3', 'cascading style sheets', 'sass', 'scss', 'less'],
            'react': ['react', 'react.js', 'reactjs', 'react native'],
            'vue': ['vue', 'vue.js', 'vuejs', 'vue3'],
            'angular': ['angular', 'angular.js', 'angularjs', 'angular2+'],
            'django': ['django', 'django rest framework', 'drf'],
            'flask': ['flask', 'flask-restful'],
            'express': ['express', 'express.js', 'expressjs'],
            # Cloud & DevOps
            'aws': ['aws', 'amazon web services', 'ec2', 's3', 'lambda', 'cloudformation'],
            'azure': ['azure', 'microsoft azure', 'azure functions', 'azure devops'],
            'gcp': ['gcp', 'google cloud platform', 'google cloud', 'firebase'],
            'docker': ['docker', 'containerization', 'containers'],
            'kubernetes': ['kubernetes', 'k8s', 'container orchestration'],
            'jenkins': ['jenkins', 'ci/cd', 'continuous integration'],
            'terraform': ['terraform', 'infrastructure as code', 'iac'],
            # Data Science & ML
            'machine learning': ['machine learning', 'ml', 'artificial intelligence', 'ai'],
            'deep learning': ['deep learning', 'dl', 'neural networks', 'cnn', 'rnn'],
            'data science': ['data science', 'data analysis', 'data analytics', 'statistics'],
            'pandas': ['pandas', 'data manipulation', 'dataframes'],
            'numpy': ['numpy', 'numerical computing', 'scientific computing'],
            'scikit-learn': ['scikit-learn', 'sklearn', 'machine learning library'],
            'tensorflow': ['tensorflow', 'tf', 'keras'],
            'pytorch': ['pytorch', 'torch', 'deep learning framework'],
            'tableau': ['tableau', 'data visualization', 'business intelligence'],
            'power bi': ['power bi', 'powerbi', 'microsoft power bi'],
            # Databases
            'mongodb': ['mongodb', 'mongo', 'nosql', 'document database'],
            'redis': ['redis', 'caching', 'in-memory database'],
            'elasticsearch': ['elasticsearch', 'elk stack', 'search engine'],
            # Project Management
            'agile': ['agile', 'scrum', 'kanban', 'sprint'],
            'jira': ['jira', 'project management', 'issue tracking'],
            'git': ['git', 'version control', 'github', 'gitlab', 'bitbucket'],
            # Testing
            'testing': ['testing', 'unit testing', 'integration testing', 'qa'],
            'selenium': ['selenium', 'automated testing', 'web testing'],
            'jest': ['jest', 'javascript testing', 'unit testing'],
            'pytest': ['pytest', 'python testing', 'test framework'],
        }
    
    def get_canonical_skill(self, skill: str) -> str:
        """Get the canonical form of a skill"""
        skill_lower = skill.lower().strip()
        for canonical, synonyms in self.skill_synonyms.items():
            if skill_lower in synonyms:
                return canonical
        return skill_lower
    
    def expand_skills(self, skills: List[str]) -> List[str]:
        """Expand a list of skills with their synonyms"""
        expanded = set()
        for skill in skills:
            canonical = self.get_canonical_skill(skill)
            expanded.add(canonical)
            if canonical in self.skill_synonyms:
                expanded.update(self.skill_synonyms[canonical])
        return list(expanded)

class DropboxManager:
    """Handles CV files from Dropbox"""
    
    def __init__(self, access_token: str):
        self.access_token = access_token
        self.dbx = self._authenticate()
    
    def _authenticate(self):
        """Authenticate with Dropbox API"""
        try:
            return dropbox.Dropbox(self.access_token)
        except AuthError as e:
            logger.error(f"Dropbox authentication failed: {str(e)}")
            raise
    
    def list_cv_files(self, folder_path: str = None) -> List[Dict[str, str]]:
        """List all PDF files in the specified folder"""
        try:
            folder_path = folder_path or DROPBOX_FOLDER_PATH
            result = self.dbx.files_list_folder(folder_path)
            files = []
            
            for entry in result.entries:
                if isinstance(entry, dropbox.files.FileMetadata) and entry.name.lower().endswith('.pdf'):
                    files.append({
                        'id': entry.id,
                        'name': entry.name,
                        'size': str(entry.size),
                        'modified': entry.server_modified.isoformat()
                    })
            
            while result.has_more:
                result = self.dbx.files_list_folder_continue(result.cursor)
                for entry in result.entries:
                    if isinstance(entry, dropbox.files.FileMetadata) and entry.name.lower().endswith('.pdf'):
                        files.append({
                            'id': entry.id,
                            'name': entry.name,
                            'size': str(entry.size),
                            'modified': entry.server_modified.isoformat()
                        })
            
            return files
        except Exception as e:
            logger.error(f"Error listing files: {str(e)}")
            return []
    
    def download_cv(self, file_path: str) -> bytes:
        """Download CV content from Dropbox"""
        try:
            _, response = self.dbx.files_download(file_path)
            return response.content
        except Exception as e:
            logger.error(f"Error downloading file {file_path}: {str(e)}")
            return b""
    
    def get_folder_info(self, folder_path: str) -> Dict[str, Any]:
        """Get information about a folder"""
        try:
            metadata = self.dbx.files_get_metadata(folder_path)
            return {
                'id': metadata.id,
                'name': metadata.name,
                'path': metadata.path_display
            }
        except Exception as e:
            logger.error(f"Error getting folder info: {str(e)}")
            return {}

class CVProcessor:
    """Processes CV files and extracts information"""
    
    def __init__(self, skill_mapper: SkillSynonymMapper):
        self.skill_mapper = skill_mapper
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """Extract text from PDF bytes with OCR fallback"""
        retries = 3
        for attempt in range(retries):
            try:
                # Try PyPDF2 first for text-based PDFs
                temp_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
                temp_file_path = temp_file.name
                try:
                    temp_file.write(pdf_content)
                    temp_file.flush()
                    temp_file.close()
                    time.sleep(0.1)
                    
                    with open(temp_file_path, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        text = ""
                        for page in pdf_reader.pages:
                            extracted = page.extract_text()
                            if extracted:
                                text += extracted + "\n"
                        if text.strip():  # If text was extracted, return it
                            logger.info(f"Successfully extracted text using PyPDF2 from {temp_file_path}")
                            return text
                finally:
                    try:
                        os.unlink(temp_file_path)
                    except Exception as e:
                        logger.warning(f"Error deleting temporary file {temp_file_path}: {str(e)}")
                
                # Fallback to OCR if PyPDF2 fails to extract meaningful text
                logger.info("No text extracted with PyPDF2, attempting OCR...")
                images = convert_from_bytes(pdf_content)
                text = ""
                for image in images:
                    text += pytesseract.image_to_string(image) + "\n"
                if text.strip():
                    logger.info("Successfully extracted text using OCR")
                    return text
                else:
                    logger.warning("No text extracted with OCR")
                    return ""
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(0.5)
                    continue
                logger.error(f"Error extracting text from PDF after {retries} attempts: {str(e)}")
                return ""
        return ""
    
    def extract_candidate_info(self, text: str, file_path: str) -> CandidateProfile:
        """Extract candidate information from CV text"""
        # Extract name
        name_match = re.search(r'(?:Name[:\s]+)([A-Za-z\s]+)', text, re.IGNORECASE)
        if name_match:
            name = name_match.group(1).strip()
        else:
            name = os.path.basename(file_path).replace('.pdf', '').replace('_', ' ').title()
        
        # Extract skills
        skills = self._extract_skills(text)
        
        # Extract experience and years
        experience, experience_years = self._extract_experience(text)
        
        # Extract education
        education = self._extract_education(text)
        
        return CandidateProfile(
            name=name,
            skills=skills,
            experience=experience,
            experience_years=experience_years,
            education=education,
            raw_text=text,
            file_path=file_path
        )
    
    def _extract_skills(self, text: str) -> List[str]:
        """Extract skills from CV text"""
        skills = set()
        skill_patterns = [
            r'(?:Skills?|Technical Skills?|Key Skills?|Core Competencies)[:\s]*\n?([^,\n]+(?:[,\n][^,\n]+)*)',
            r'(?:Technologies?|Tools?|Platforms?)[:\s]*\n?([^,\n]+(?:[,\n][^,\n]+)*)',
            r'(?:Programming Languages?|Languages?)[:\s]*\n?([^,\n]+(?:[,\n][^,\n]+)*)',
        ]
        
        for pattern in skill_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                skill_items = re.split(r'[,\n\|•·\-\*]', match)
                for skill in skill_items:
                    cleaned_skill = re.sub(r'[^\w\s+#]', '', skill.strip())
                    if cleaned_skill and len(cleaned_skill) > 1:
                        skills.add(cleaned_skill.lower())
        
        all_skills = set()
        for canonical_skill, synonyms in self.skill_mapper.skill_synonyms.items():
            for synonym in synonyms:
                if re.search(r'\b' + re.escape(synonym) + r'\b', text, re.IGNORECASE):
                    all_skills.add(canonical_skill)
        
        skills.update(all_skills)
        return list(skills)
    
    def _extract_experience(self, text: str) -> tuple[str, float]:
        """Extract experience information and estimate years"""
        exp_patterns = [
            r'(?:Experience|Work Experience|Professional Experience)[:\s]*\n?([^\n]{1,500})',
            r'(?:Employment|Career|Work History)[:\s]*\n?([^\n]{1,500})',
        ]
        
        for pattern in exp_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                experience_text = match.group(1).strip()
                # Extract years of experience
                years_match = re.search(r'(\d+\.?\d*)\s*(?:years?|yrs?)\s*(?:of\s*experience)?', experience_text, re.IGNORECASE)
                years = float(years_match.group(1)) if years_match else 0.0
                return experience_text, years
        
        return "Not specified", 0.0
    
    def _extract_education(self, text: str) -> str:
        """Extract education information"""
        edu_patterns = [
            r'(?:Education|Academic Background|Qualification)[:\s]*\n?([^\n]{1,500})',
            r'(?:Degree|University|College)[:\s]*\n?([^\n]{1,500})',
        ]
        
        for pattern in edu_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()
        
        return "Not specified"

class LLMClient:
    """Handles LLM API calls"""
    
    def __init__(self, api_url: str):
        self.api_url = api_url
    
    def call_llm(self, prompt: str, system_content: str) -> str:
        """Make API call to LLM"""
        payload = {
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "model": "gpt-4o"
        }
        
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            api_response = response.json()
            
            if api_response.get("status") == "success":
                return api_response.get("response")
            else:
                return f"Error: {api_response.get('message', 'Unknown error')}"
        except requests.RequestException as e:
            return f"API Error: {str(e)}"

class SemanticMatcher:
    """Handles semantic similarity matching"""
    
    def __init__(self, sentence_transformer):
        self.sentence_transformer = sentence_transformer
    
    def compute_similarity(self, job_requirements: List[str], candidate_skills: List[str]) -> float:
        """Compute semantic similarity between job requirements and candidate skills"""
        if not job_requirements or not candidate_skills:
            return 0.0
        
        req_embeddings = self.sentence_transformer.encode(job_requirements)
        skill_embeddings = self.sentence_transformer.encode(candidate_skills)
        similarities = cosine_similarity(req_embeddings, skill_embeddings)
        return float(np.mean(np.max(similarities, axis=1)))

# LangGraph Node Functions
import json
from langchain_core.messages import HumanMessage
import logging

logger = logging.getLogger(__name__)

def extract_keywords_node(state: CVMatchingState) -> CVMatchingState:
    """Extract keywords from job requirements with weights"""
    llm_client = LLMClient("https://206c-20-106-58-127.ngrok-free.app/chat")
    
    system_content = """You are a technical assistant that extracts important keywords from job requirements and assigns weights (1-10) based on their importance. Focus on technical skills, tools, technologies, platforms, frameworks, and methodologies. Return a JSON list of objects with 'keyword' and 'weight' fields."""
    
    prompt = f"""Extract technical keywords from this job requirement and assign a weight (1-10) to each based on importance:
    {state['job_requirements']}
    
    Return a JSON list of objects, e.g., [{{\"keyword\": \"python\", \"weight\": 8}}, ...]."""
    
    response = llm_client.call_llm(prompt, system_content)
    try:
        keywords = json.loads(response)
        if not isinstance(keywords, list):
            raise ValueError("Response is not a list")
        state['extracted_keywords'] = [
            {"keyword": kw["keyword"], "weight": min(max(float(kw.get("weight", 1)), 1), 10)}
            for kw in keywords if "keyword" in kw
        ]
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Error parsing LLM response: {str(e)}")
        state['extracted_keywords'] = [{"keyword": kw.strip(), "weight": 1.0} for kw in state['job_requirements'].split(',') if kw.strip()]
    
    state['messages'].append(HumanMessage(content=f"Extracted keywords: {', '.join(kw['keyword'] for kw in state['extracted_keywords'])}"))
    return state

def load_cvs_node(state: CVMatchingState) -> CVMatchingState:
    """Load CVs from Dropbox"""
    try:
        dropbox_manager = DropboxManager(
            access_token=DROPBOX_ACCESS_TOKEN
        )
        cv_processor = CVProcessor(SkillSynonymMapper())
        
        cv_files = dropbox_manager.list_cv_files(folder_path=DROPBOX_FOLDER_PATH)
        candidate_profiles = []
        
        for file_info in cv_files:
            try:
                cv_content = dropbox_manager.download_cv(f"{DROPBOX_FOLDER_PATH}/{file_info['name']}")
                if cv_content:
                    text = cv_processor.extract_text_from_pdf(cv_content)
                    if text.strip():
                        profile = cv_processor.extract_candidate_info(text, file_info['name'])
                        candidate_profiles.append(profile)
                        logger.info(f"Successfully processed {file_info['name']}")
                    else:
                        logger.warning(f"No text extracted from {file_info['name']}")
                        state['errors'].append(f"No text extracted from {file_info['name']}")
                else:
                    logger.warning(f"No content downloaded for {file_info['name']}")
                    state['errors'].append(f"No content downloaded for {file_info['name']}")
            except Exception as e:
                logger.error(f"Error processing {file_info['name']}: {str(e)}")
                state['errors'].append(f"Error processing {file_info['name']}: {str(e)}")
        
        state['candidate_profiles'] = [
            {
                'name': profile.name,
                'skills': profile.skills,
                'experience': profile.experience,
                'experience_years': profile.experience_years,
                'education': profile.education,
                'file_path': profile.file_path
            }
            for profile in candidate_profiles
        ]
        
        state['messages'].append(HumanMessage(content=f"Loaded {len(candidate_profiles)} CVs from Dropbox"))
        
    except Exception as e:
        error_msg = f"Error loading CVs from Dropbox: {str(e)}"
        state['errors'].append(error_msg)
        logger.error(error_msg)
    
    return state

def compute_similarities_node(state: CVMatchingState) -> CVMatchingState:
    """Compute semantic similarities between requirements and candidate skills"""
    sentence_transformer = load_sentence_transformer()
    semantic_matcher = SemanticMatcher(sentence_transformer)
    skill_mapper = SkillSynonymMapper()
    
    # Expand keywords with synonyms
    expanded_keywords = []
    for kw in state['extracted_keywords']:
        canonical = skill_mapper.get_canonical_skill(kw['keyword'])
        synonyms = skill_mapper.skill_synonyms.get(canonical, [canonical])
        for synonym in synonyms:
            expanded_keywords.append({'keyword': synonym, 'weight': kw['weight']})
    
    similarity_scores = []
    
    for candidate in state['candidate_profiles']:
        expanded_skills = skill_mapper.expand_skills(candidate['skills'])
        
        # Compute exact matches with weights
        exact_matches = set(kw['keyword'] for kw in expanded_keywords).intersection(set(expanded_skills))
        exact_score = 0.0
        if expanded_keywords:
            total_weight = sum(kw['weight'] for kw in expanded_keywords)
            match_weight = sum(kw['weight'] for kw in expanded_keywords if kw['keyword'] in exact_matches)
            exact_score = match_weight / total_weight if total_weight > 0 else 0.0
        
        # Compute semantic similarity
        semantic_score = semantic_matcher.compute_similarity(
            [kw['keyword'] for kw in expanded_keywords],
            expanded_skills
        )
        
        # Experience score (normalize years to 0-1 scale, assuming 10 years max)
        experience_score = min(candidate['experience_years'] / 10.0, 1.0)
        
        # Education relevance (basic check for degree keywords)
        education_keywords = ['bachelor', 'master', 'phd', 'degree', 'diploma']
        education_score = 0.5 if any(kw.lower() in candidate['education'].lower() for kw in education_keywords) else 0.0
        
        # Combined score: 50% exact match, 30% semantic, 15% experience, 5% education
        combined_score = (exact_score * 0.5) + (semantic_score * 0.3) + (experience_score * 0.15) + (education_score * 0.05)
        
        # Normalize score to prevent inflation
        combined_score = min(combined_score, 1.0)
        
        similarity_scores.append({
            'candidate': candidate['name'],
            'exact_matches': list(exact_matches),
            'exact_score': exact_score,
            'semantic_score': semantic_score,
            'experience_score': experience_score,
            'education_score': education_score,
            'combined_score': combined_score,
            'skills': candidate['skills'],
            'file_path': candidate['file_path']
        })
    
    state['similarity_scores'] = similarity_scores
    return state

def rank_candidates_node(state: CVMatchingState) -> CVMatchingState:
    """Rank candidates based on similarity scores"""
    ranked = sorted(
        state['similarity_scores'],
        key=lambda x: (-x['combined_score'], x['candidate'])
    )
    
    state['ranked_candidates'] = ranked
    
    ranking_message = "**Candidate Rankings:**\n\n"
    for i, candidate in enumerate(ranked, 1):
        score_percentage = int(candidate['combined_score'] * 100)
        matching_skills = candidate['exact_matches']
        
        ranking_message += f"**Rank {i}:** {candidate['candidate']} - "
        ranking_message += f"Score: {score_percentage}% "
        ranking_message += f"(Matching Skills: {', '.join(matching_skills) if matching_skills else 'None'})\n"
    
    state['messages'].append(AIMessage(content=ranking_message))
    return state

def create_cv_matching_graph():
    """Create the LangGraph workflow"""
    workflow = StateGraph(CVMatchingState)
    
    workflow.add_node("extract_keywords", extract_keywords_node)
    workflow.add_node("load_cvs", load_cvs_node)
    workflow.add_node("compute_similarities", compute_similarities_node)
    workflow.add_node("rank_candidates", rank_candidates_node)
    
    workflow.add_edge(START, "extract_keywords")
    workflow.add_edge("extract_keywords", "load_cvs")
    workflow.add_edge("load_cvs", "compute_similarities")
    workflow.add_edge("compute_similarities", "rank_candidates")
    workflow.add_edge("rank_candidates", END)
    return workflow.compile()


def generate_workflow_graph(output_dir: str = "graphs") -> str:
    """Generate a PNG image of the LangGraph workflow using networkx and matplotlib.
    
    Args:
        output_dir: Directory to save the PNG file
    
    Returns:
        Path to the saved PNG file, or empty string if rendering fails
    """
    try:
        # Create a directed graph
        G = nx.DiGraph()
        
        # Define nodes
        nodes = [
            'START',
            'extract_keywords',
            'load_cvs',
            'compute_similarities',
            'rank_candidates',
            'END'
        ]
        G.add_nodes_from(nodes)
        
        # Define edges
        edges = [
            ('START', 'extract_keywords'),
            ('extract_keywords', 'load_cvs'),
            ('load_cvs', 'compute_similarities'),
            ('compute_similarities', 'rank_candidates'),
            ('rank_candidates', 'END')
        ]
        G.add_edges_from(edges)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"workflow_graph_{timestamp}.png")
        
        # Set up the plot
        plt.figure(figsize=(8, 5))
        
        # Define node positions using a layered layout
        pos = nx.spring_layout(G, k=0.9, iterations=50)
        
        # Draw nodes with custom styles
        node_colors = ['lightgreen' if node == 'START' else 'lightcoral' if node == 'END' else 'skyblue' for node in G.nodes]
        node_shapes = ['o' if node in ['START', 'END'] else 's' for node in G.nodes]
        
        # Draw nodes
        for node, shape in zip(G.nodes, node_shapes):
            nx.draw_networkx_nodes(
                G, pos, nodelist=[node],
                node_color=node_colors[nodes.index(node)],
                node_shape=shape,
                node_size=2000,
                edgecolors='navy'
            )
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='black', arrows=True, arrowsize=20)
        
        # Draw labels
        labels = {
            'START': 'START',
            'extract_keywords': 'Extract\nKeywords',
            'load_cvs': 'Load\nCVs',
            'compute_similarities': 'Compute\nSimilarities',
            'rank_candidates': 'Rank\nCandidates',
            'END': 'END'
        }
        nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')
        
        # Customize plot
        plt.title('CV Matching Workflow', fontsize=14, pad=20)
        plt.axis('off')  # Hide axes for cleaner look
        
        # Save and close plot
        plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    except Exception as e:
        error_msg = f"Failed to generate workflow graph: {str(e)}"
        st.error(error_msg)
        return ""

# Streamlit UI
def main():
    st.set_page_config(page_title="AI-Powered RAG based CV Filtering System", layout="wide")

    st.title("🎯 AI-Powered RAG based CV Filtering System")
    st.markdown("*Revolutionize recruitment with cutting-edge RAG and Agentic AI: intelligent semantic analysis, synonym-aware matching, and seamless Dropbox integration for unmatched efficiency and precision.*")
        
    with st.sidebar:
        st.header("Configuration")
        st.info("🔧 Dropbox Setup Required")
        
        folder_path = st.text_input(
            "Dropbox Folder Path",
            value=DROPBOX_FOLDER_PATH or "/CVs",
            help="Enter the folder path containing your CVs (e.g., /CVs)"
        )
        
        access_token = st.text_input(
            "Dropbox Access Token",
            value=DROPBOX_ACCESS_TOKEN or "",
            type="password",
            help="Enter your Dropbox access token"
        )
        
        if st.button("📁 Test Dropbox Connection"):
            if access_token and folder_path:
                try:
                    dropbox_manager = DropboxManager(access_token=access_token)
                    folder_info = dropbox_manager.get_folder_info(folder_path)
                    if folder_info:
                        st.success(f"✅ Connected to folder: {folder_info.get('name', 'Unknown')}")
                        cv_files = dropbox_manager.list_cv_files(folder_path)
                        st.success(f"📄 Found {len(cv_files)} PDF files")
                        if cv_files:
                            st.write("**Sample files:**")
                            for file_info in cv_files[:5]:
                                st.write(f"• {file_info['name']}")
                            if len(cv_files) > 5:
                                st.write(f"... and {len(cv_files) - 5} more")
                    else:
                        st.error("❌ Could not access folder. Check your folder path and permissions.")
                except Exception as e:
                    st.error(f"❌ Connection failed: {str(e)}")
                    st.info("Make sure you have a valid Dropbox access token and folder path.")
            else:
                st.warning("Please enter both a Dropbox access token and folder path")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Job Requirements")
        job_requirements = st.text_area(
            "Enter the job requirements and desired skills:",
            height=150,
            placeholder="e.g., Python developer with 5+ years of experience in Django, React, AWS, machine learning, and a Bachelor's degree in Computer Science..."
        )
    
    with col2:
        st.header("Actions")
        if access_token and folder_path:
            os.environ["DROPBOX_ACCESS_TOKEN"] = access_token
            os.environ["DROPBOX_FOLDER_PATH"] = folder_path
            
        if st.button("🚀 Match Candidates", type="primary", use_container_width=True):
            if job_requirements.strip():
                if access_token and folder_path:
                    match_candidates(job_requirements, folder_path, access_token)
                else:
                    st.error("Please provide both a Dropbox access token and folder path")
            else:
                st.error("Please enter job requirements")
        
        if st.button("📊 View System Stats", use_container_width=True):
            if access_token and folder_path:
                show_system_stats(folder_path, access_token)
            else:
                st.error("Please provide both a Dropbox access token and folder path")
        
        # In the main() function, under the "Actions" section in col2

        if st.button("📈 View Workflow Graph", use_container_width=True):
            graph_path = generate_workflow_graph()
            if graph_path:
                st.subheader("🗺️ LangGraph Workflow")
                st.image(graph_path)
                st.write(f"Graph saved to: {graph_path}")
            else:
                st.warning("Failed to generate workflow graph. Check error messages above.")

def match_candidates(job_requirements: str, folder_path: str, access_token: str):
    """Execute the CV matching workflow"""
    with st.spinner("🔄 Processing CVs and matching candidates..."):
        os.environ["DROPBOX_ACCESS_TOKEN"] = access_token
        os.environ["DROPBOX_FOLDER_PATH"] = folder_path
        
        initial_state = CVMatchingState(
            job_requirements=job_requirements,
            extracted_keywords=[],
            cv_documents=[],
            candidate_profiles=[],
            similarity_scores=[],
            ranked_candidates=[],
            messages=[],
            errors=[]
        )
        
        graph = create_cv_matching_graph()
        final_state = graph.invoke(initial_state)
        
        if final_state['errors']:
            st.error("⚠️ Some errors occurred:")
            for error in final_state['errors']:
                st.error(error)
        
        if final_state['ranked_candidates']:
            st.success("✅ Candidate matching completed!")
            
            st.subheader("🔍 Extracted Keywords")
            st.write(", ".join(kw['keyword'] for kw in final_state['extracted_keywords']))
            
            st.subheader("🏆 Candidate Rankings")
            
            for i, candidate in enumerate(final_state['ranked_candidates'], 1):
                score_percentage = int(candidate['combined_score'] * 100)
                
                with st.expander(f"Rank {i}: {candidate['candidate']} - Score: {score_percentage}%"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Matching Skills:**")
                        if candidate['exact_matches']:
                            for skill in candidate['exact_matches']:
                                st.write(f"✅ {skill}")
                        else:
                            st.write("No exact matches found")
                    
                    with col2:
                        st.write("**All Skills:**")
                        for skill in candidate['skills'][:10]:
                            st.write(f"• {skill}")
                        if len(candidate['skills']) > 10:
                            st.write(f"... and {len(candidate['skills']) - 10} more")
                    
                    st.write(f"**Exact Match Score:** {int(candidate['exact_score'] * 100)}%")
                    st.write(f"**Semantic Similarity:** {int(candidate['semantic_score'] * 100)}%")
                    st.write(f"**Experience Score:** {int(candidate['experience_score'] * 100)}%")
                    st.write(f"**Education Score:** {int(candidate['education_score'] * 100)}%")
                    st.write(f"**File:** {candidate['file_path']}")
        else:
            st.warning("No candidates found or processed. Check if CVs contain extractable text.")

def show_system_stats(folder_path: str, access_token: str):
    """Display system statistics"""
    try:
        dropbox_manager = DropboxManager(access_token=access_token)
        cv_files = dropbox_manager.list_cv_files(folder_path=folder_path)
        
        st.subheader("📈 System Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total CVs", len(cv_files))
        
        with col2:
            st.metric("Skill Synonyms", len(SkillSynonymMapper().skill_synonyms))
        
        with col3:
            st.metric("Last Updated", datetime.now().strftime("%Y-%m-%d %H:%M"))
        
        st.subheader("📁 CV Files")
        for file_info in cv_files[:10]:
            st.write(f"• {file_info['name']}")
        
        if len(cv_files) > 10:
            st.write(f"... and {len(cv_files) - 10} more files")
            
    except Exception as e:
        st.error(f"Error fetching statistics: {str(e)}")

if __name__ == "__main__":
    main()