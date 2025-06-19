import base64
import io
import os
import re
import yaml
import time
import shutil
import logging
import requests
from PIL import Image

from pathlib import Path
from dotenv import load_dotenv
from fastapi import UploadFile
from typing import List, Dict
from src.ocr import MathpixOCR
from src.pydantic_models import QuestionAnswer
from tempfile import NamedTemporaryFile
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from urllib.parse import urlparse, unquote
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Redirect tqdm logging to the root logger
logging_redirect_tqdm()

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Tractian - Document Ingestor")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("faiss").setLevel(logging.WARNING)

# Load prompts
with open("src/prompts.yaml", "r", encoding="utf-8") as f:
    prompts = yaml.safe_load(f)

class DocumentIngestor:
    """
    This class is used to parse a PDF file to markdown, chunk it and create embeddings for each chunk.
    It also creates the chain for the retrieval of the most relevant chunks for a given question.
    """
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
        self.llm_fallback = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=os.getenv("GOOGLE_API_KEY"))
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
        self.cached_files = []
        self.documents_json = []
        self.stored_chunks = []
        self.stored_images = []
        self.output_parser = StrOutputParser(pydantic_object=QuestionAnswer)

    def convert_pdf_to_markdown(self, file: UploadFile) -> str:
        """
        Process a single PDF FastAPI UploadFile: save to temp, convert to markdown, cleanup temp file.
        Returns the markdown string.
        """
        logger.info(f"Converting PDF to markdown...")
        suffix = Path(file.filename).suffix or ".tmp"
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            try:
                shutil.copyfileobj(file.file, tmp)
            finally:
                file.file.close()
        try:
            markdown, pdf_json = MathpixOCR().pdf_to_markdown(tmp_path)
            self.documents_json.append(pdf_json)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            
        return markdown
    
    def encode_image(self, image_url):
        """Encode image to base64."""            
        headers = {
            'User-Agent': 'AddImageDescription/1.0 (your@email.com) PythonRequests/2.25.1'
        }
        
        try:
            response = requests.get(image_url, headers=headers)
            response.raise_for_status()
            
            # Use a more efficient approach for common image formats
            try:
                img = Image.open(io.BytesIO(response.content))
                
                # Use a smaller size for faster processing
                max_size = 800  # Reduced from 1024 for better performance
                if max(img.size) > max_size:
                    img.thumbnail((max_size, max_size))
                
                # Use a more efficient compression
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG', quality=85)  # Reduced from 85
                img_byte_arr = img_byte_arr.getvalue()
                
                if len(img_byte_arr) > 20 * 1024 * 1024:
                    raise ValueError("Image size exceeds 20MB limit")
                
                encoded_image = base64.b64encode(img_byte_arr).decode('utf-8')
                return encoded_image
            
            except Exception as e:
                print(f"Error processing image format: {str(e)}")
                return None
                
        except Exception as e:
            print(f"Error downloading image from {image_url}: {str(e)}")
            raise
    
    def extract_md_image(self, markdown: str) -> List[str]:
        """Return a list with each raw Markdown image string."""
        _IMAGE_RX = re.compile(r'!\[[^\]]*\]\(\s*[^)\s]+(?:\s+"[^"]*")?\s*\)', re.VERBOSE | re.MULTILINE)
        return [m.group(0) for m in _IMAGE_RX.finditer(markdown)]

    def has_md_image(self, markdown: str) -> bool:
        """Return True if a Markdown image pattern is found, else False."""
        _IMAGE_RX = re.compile(r'!\[[^\]]*\]\(\s*[^)\s]+(?:\s+"[^"]*")?\s*\)', re.VERBOSE | re.MULTILINE)
        return bool(_IMAGE_RX.search(markdown))
    
    def extract_image_metadata(self, markdown_or_url: str) -> int:
        """
        Given either a complete ![](...) tag *or* a bare URL, return the page number.

        Raises ValueError on any malformed input.
        """
        IMG_TAG_RX = re.compile(r"!\[(?:[^\]]*)\]\(\s*(?P<url>[^)\s]+)(?:\s+\"[^\"]*\")?\s*\)", re.MULTILINE)
        
        # Pull out the URL if we're handed a full Markdown tag
        m = IMG_TAG_RX.search(markdown_or_url)
        url = m.group("url") if m else markdown_or_url.strip()

        # Isolate the last path component from image url: "...-42.jpg"
        last_seg = os.path.basename(unquote(urlparse(url).path))
        if not last_seg:
            raise ValueError(f"Cannot parse path component from {url!r}")

        # Split on the last hyphen
        stem, _ext = os.path.splitext(last_seg)
        if "-" not in stem:
            raise ValueError(f"No hyphen in file name {last_seg!r}")

        img_id, page_str = stem.rsplit("-", 1)
        if not page_str.isdigit():
            raise ValueError(f"Trailing chunk after hyphen is not numeric: {page_str!r}")

        return int(page_str)
    
    def ensure_not_empty(self, response):
        """Ensure the response is not empty."""
        if not getattr(response, "content", "").strip():
            raise ValueError("LLM returned empty content")
        return response
    
    def describe_image(self, image: str) -> str:
        """Describe the images in the list."""
        _URL_MD_RX = re.compile(r'!\[[^\]]*\]\(\s*([^) \t\n\r]+)')
        
        url = _URL_MD_RX.search(image).group(1)
        try:
            encoded_image = self.encode_image(url)
        
            if encoded_image:
                messages = [
                    SystemMessage(content=prompts["describe_images"]["system"]),
                    HumanMessage(content=[
                        {"type": "text", "text": prompts["describe_images"]["user"]},
                        {"type": "image", "source_type": "base64", "data": encoded_image, "mime_type": "image/jpeg"}
                    ])
                ]
                
                primary   =  self.llm          | self.ensure_not_empty
                secondary =  self.llm_fallback | self.ensure_not_empty

                llm_with_fallback = primary.with_fallbacks(
                    [secondary], exceptions_to_handle=(Exception, ValueError)
                )

                response = llm_with_fallback.invoke(messages)
                return response.content

        except Exception as e:
            print(f"Error while describing image {url}: {e}")
            return None
    
    def organize_image_metadata(self, descriptions: List[Dict]) -> List[Dict]:
        """Organize the image metadata number by page."""
        page_counts = {}
        for description in descriptions:
            page = description["page"]
            if page not in page_counts:
                page_counts[page] = 1
            else:
                page_counts[page] += 1
            description["number"] = page_counts[page]
        
        return descriptions
    
    def process_image(self, image):
        description = self.describe_image(image)
        page = self.extract_image_metadata(image)
        
        # Sleep to avoid rate limit
        time.sleep(1)
        return {
            "image": image,
            "description": description,
            "page": page
        }
    
    def describe_images(self, markdown: str) -> List[str]:
        """Describe the images from the markdown using ThreadPoolExecutor and tqdm."""
        images = self.extract_md_image(markdown)
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(tqdm(executor.map(self.process_image, images), total=len(images), desc="Describing images..."))

        descriptions = self.organize_image_metadata(results)

        return descriptions

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks based on RecursiveCharacterTextSplitter strategy."""
        logger.info(f"Chunking text...")
        text_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.MARKDOWN, chunk_size=1250, chunk_overlap=0)
        chunks = text_splitter.create_documents([text])
        return chunks

    def parse_document(self, pdf_file) -> Dict:
        """Process a PDF document: convert to markdown, chunk it, and create embeddings."""
        logger.info(f"Ingesting document...")
        text = self.convert_pdf_to_markdown(pdf_file)
        chunks = self.chunk_text(text)
        
        self.stored_chunks.append({"file": pdf_file.filename, "chunks": chunks})
        
        if self.has_md_image(text):
            logger.info(f"Processing images...")
            images = self.describe_images(text)

            for image in images:
                image["file"] = pdf_file.filename
                
            self.stored_images.extend(images)
            print("Stored images: ", self.stored_images)
        
        self.cached_files.append(pdf_file.filename)
        
        return {
            "chunks_processed": len(chunks) + (len(images) if self.has_md_image(text) else 0)
        }
    
    def prepare_prompt_template(self):
        """Prepare the prompt template."""
        return ChatPromptTemplate.from_messages([("system", prompts["prompts"]["system"]), ("user", prompts["prompts"]["user"])])

    def format_docs(self, docs):
        """Format the documents into a string."""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def chain_retrieval(self) -> RetrievalQA:
        """Find the most relevant chunks for a given question."""
        if not self.stored_chunks:
            return None

        # Initialize the list of documents
        all_docs: list[Document] = []

        # Add text chunks to all_docs
        for chunk in self.stored_chunks:
            all_docs.extend(chunk["chunks"])

        # Turn each image description into a Document to identify where it came from
        print("Stored images: ", self.stored_images)
        if self.stored_images:
            all_docs.extend(
                Document(
                    page_content=f"{desc['description']}",
                    metadata={"origin": "image_description", "file": desc["file"], "page": desc["page"], "number": desc["number"]}
                )
                for desc in self.stored_images
                if desc       
            )

        vector_store = FAISS.from_documents(all_docs, self.embeddings)

        retriever = vector_store.as_retriever()
        
        prompt = self.prepare_prompt_template()

        # Set base inputs
        base_inputs = {
            "question": lambda x: x["question"],
            "context":  lambda x: self.format_docs(x["context"]),
        }

        # Set fallback chain
        primary   = base_inputs | prompt | self.llm        .with_structured_output(QuestionAnswer)
        secondary = base_inputs | prompt | self.llm_fallback.with_structured_output(QuestionAnswer)

        # Automatic retry on *any* Exception (you can narrow this tuple)
        rag_chain_from_docs = primary.with_fallbacks([secondary], exceptions_to_handle=(Exception,))
        
        retrieve_docs = (lambda x: x["question"]) | retriever

        chain = RunnablePassthrough.assign(context=retrieve_docs).assign(
            answer=rag_chain_from_docs
        )

        return chain