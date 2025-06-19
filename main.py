# Imports
import logging
import logfire
import uvicorn

from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException

from src.document_ingestor import DocumentIngestor
from src.pydantic_models import QuestionAnswer, QuestionRequest

import warnings
from urllib3.exceptions import InsecureRequestWarning

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Tractian - Document-based Chatbot")

# Suppress InsecureRequestWarning from Mathpix
warnings.simplefilter("ignore", InsecureRequestWarning)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Tractian - Document-based Chatbot", description="RAG-based chatbot for Tractian")

# Configure Logfire
logfire.configure()
logfire.instrument_fastapi(app)
logging.basicConfig(handlers=[logfire.LogfireLoggingHandler()])

logger = logging.getLogger(__name__)

# Initialize DocumentIngestor
pdf_parser = DocumentIngestor()

# Routes
@app.get("/")
async def root():
    return {"message": "Tractian - Document-based Chatbot"}

@app.post("/documents")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload and process PDF documents."""
    try:
        total_chunks = 0
        for i, file in enumerate(files):
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF. Please upload a PDF file.")

            # Check if the file is already in the cache
            if file.filename in pdf_parser.cached_files:
                logger.info(f"File {file.filename} already processed. Skipping...")
                file_chunks = next((item["chunks"] for item in pdf_parser.stored_chunks if item["file"] == file.filename), [])
                total_chunks += len(file_chunks)
                continue

            logger.info(f"Starting processing of file {i+1}/{len(files)}: '{file.filename}'")
            
            result = pdf_parser.parse_document(file)
            total_chunks += result["chunks_processed"]

            logger.info(f"Processed file {i+1}: '{file.filename}' with {result['chunks_processed']} chunks")

        return {
            "message": "Documents processed successfully",
            "documents_indexed": len(files),
            "total_chunks": total_chunks
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/question", response_model=QuestionAnswer)
async def answer_question(request: QuestionRequest):
    """Answer a question based on the uploaded documents."""
    try:
        if not pdf_parser.stored_chunks:
            return QuestionAnswer(
                answer="No relevant information found. Please upload some documents first.",
                references=[]
            )
        
        # Get response from LLM chain
        response = pdf_parser.chain_retrieval().invoke({"question": request.question})
        
        # Parse the response from the chain
        parsed_response = response['answer']
        
        return parsed_response.model_dump()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True) 