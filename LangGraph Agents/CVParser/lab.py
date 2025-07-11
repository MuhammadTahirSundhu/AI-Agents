import gdown
import os
import re
import fitz
from typing import List, Dict, Any
from urllib.parse import urlparse, parse_qs
import requests
from io import BytesIO
from dotenv import load_dotenv
load_dotenv()
import os
import re
import requests
from urllib.parse import urlparse, parse_qs
from io import BytesIO
from typing import List, Dict, Any
import fitz  # PyMuPDF


def extract_pdfs_to_markdown(folder_path: str = None, shared_link: str = None, api_key: str = None) -> List[Dict[str, str]]:
    """
    Extracts PDF content to markdown from either a local folder or Google Drive folder.
    
    Args:
        folder_path (str, optional): Path to local folder containing PDFs
        shared_link (str, optional): Google Drive shared folder link
        api_key (str, optional): Google Drive API key for listing files
    
    Returns:
        List[Dict[str, str]]: List of dictionaries containing filename and extracted markdown content
    """
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

    def extract_pdf_advanced(pdf_source, filename: str) -> str:
        try:
            # Handle both file path and stream
            if isinstance(pdf_source, str):
                doc = fitz.open(pdf_source)
            else:
                doc = fitz.open(stream=pdf_source, filetype="pdf")
            
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
            return f"Error extracting PDF {filename}: {str(e)}"

    def extract_pdf_simple(pdf_source, filename: str) -> str:
        try:
            # Handle both file path and stream
            if isinstance(pdf_source, str):
                doc = fitz.open(pdf_source)
            else:
                doc = fitz.open(stream=pdf_source, filetype="pdf")
            
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
            return f"Error extracting PDF {filename}: {str(e)}"

    # Process Google Drive folder
    if shared_link:
        # Use environment variable if no API key is provided
        api_key = api_key or os.getenv("GOOGLE_DRIVE_API_KEY")
        if not api_key:
            return [{"filename": "", "content": "Error: Google Drive API key is required. Set GOOGLE_DRIVE_API_KEY environment variable or pass api_key parameter. Create an API key in Google Cloud Console under project 'elated-drive-465209-h6'."}]

        # Extract folder ID from the shared link
        parsed_url = urlparse(shared_link)
        query_params = parse_qs(parsed_url.query)
        folder_id = query_params.get('id', [None])[0]
        if not folder_id:
            folder_id = parsed_url.path.split('/')[-1]
        
        if not folder_id:
            return [{"filename": "", "content": "Error: Could not extract folder ID from the shared link"}]
        
        # Use Google Drive API to list files in the folder
        list_url = f"https://www.googleapis.com/drive/v3/files?q='{folder_id}'+in+parents&fields=files(id,name,mimeType)&key={api_key}"
        
        try:
            response = requests.get(list_url)
            response.raise_for_status()
            files = response.json().get('files', [])
        except Exception as e:
            return [{"filename": "", "content": f"Error listing files in folder: {str(e)}. Ensure the folder is public ('Anyone with the link') and the API key is valid for project 'elated-drive-465209-h6'."}]

        results = []
        # Process each PDF file from Google Drive
        for file in files:
            if file.get('mimeType') == 'application/pdf':
                filename = file['name']
                file_id = file['id']
                download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
                
                try:
                    # Stream the PDF content
                    session = requests.Session()
                    response = session.get(download_url, stream=True)
                    
                    # Handle Google Drive download confirmation if required
                    if response.headers.get('Content-Type', '').startswith('text/html'):
                        # Extract confirmation token
                        confirm_token = re.search(r'confirm=([^&]+)', response.text)
                        if confirm_token:
                            download_url += f"&confirm={confirm_token.group(1)}"
                            response = session.get(download_url, stream=True)
                    
                    response.raise_for_status()
                    pdf_stream = BytesIO(response.content)
                    
                    # Process the PDF stream
                    markdown_text = extract_pdf_advanced(pdf_stream, filename)
                    
                    if markdown_text.startswith("Error"):
                        print(f"Advanced extraction failed for {filename}, trying simple extraction...")
                        pdf_stream.seek(0)  # Reset stream position
                        markdown_text = extract_pdf_simple(pdf_stream, filename)
                    
                    results.append({
                        "filename": filename,
                        "content": markdown_text
                    })
                    
                except Exception as e:
                    results.append({
                        "filename": filename,
                        "content": f"An error occurred: {str(e)}"
                    })
            else:
                # Handle non-PDF files (e.g., .doc, .docx)
                filename = file['name']
                if filename.lower().endswith(('.doc', '.docx')):
                    results.append({
                        "filename": filename,
                        "content": f"Error: Extraction of {filename} (.doc/.docx) is not supported in this implementation."
                    })

        return results

    # Process local folder
    elif folder_path:
        results = []
        if not os.path.isdir(folder_path):
            return [{"filename": "", "content": f"Error: '{folder_path}' is not a valid directory"}]
        
        for filename in os.listdir(folder_path):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(folder_path, filename)
                try:
                    markdown_text = extract_pdf_advanced(pdf_path, filename)
                    
                    if markdown_text.startswith("Error"):
                        print(f"Advanced extraction failed for {filename}, trying simple extraction...")
                        markdown_text = extract_pdf_simple(pdf_path, filename)
                    
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
    
    else:
        return [{"filename": "", "content": "Error: Either folder_path or shared_link must be provided"}]


def run_extractor(shared_folder_link:str):
    """
    Example function to run the extractor with Google Drive
    """
    shared_folder_link = "https://drive.google.com/drive/folders/14VrlgFXFXxRWgtsLKxjIin2UPGxf61GW"
    api_key = os.getenv("GOOGLE_DRIVE_API_KEY")  # Replace with your API key from project 'elated-drive-465209-h6'
    extracted_content = extract_pdfs_to_markdown(shared_link=shared_folder_link, api_key=api_key)
    return extracted_content


# Example usage:
# For local folder:
# results = extract_pdfs_to_markdown(folder_path="/path/to/pdf/folder")

# For Google Drive folder:
# results = extract_pdfs_to_markdown(shared_link="https://drive.google.com/drive/folders/your_folder_id", api_key="your_api_key")

