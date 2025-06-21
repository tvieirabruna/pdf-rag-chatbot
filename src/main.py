# Imports
from pathlib import Path
import logging
import uvicorn

from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from core.document_ingestor import DocumentIngestor
from core.schemas import QuestionAnswer, QuestionRequest
from config import SERVER, LOGGING, DOCUMENT
from middleware.logging_middleware import LoggingMiddleware

import warnings
from urllib3.exceptions import InsecureRequestWarning

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOGGING["level"]),
    format=LOGGING["format"]
)
logger = logging.getLogger(SERVER["TITLE"])

# Suppress InsecureRequestWarning from Mathpix
warnings.simplefilter("ignore", InsecureRequestWarning)

# Initialize FastAPI app
app = FastAPI(
    title=SERVER["TITLE"], 
    description=SERVER["DESCRIPTION"]
)

# Add logging middleware
app.add_middleware(LoggingMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize DocumentIngestor
pdf_parser = DocumentIngestor()

# Routes
@app.get("/")
async def root():
    return {"message": SERVER["TITLE"]}

@app.post("/documents")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload and process PDF documents."""
    try:
        total_chunks = 0
        for i, file in enumerate(files):
            if not any(file.filename.lower().endswith(fmt) for fmt in DOCUMENT["supported_formats"]):
                raise HTTPException(
                    status_code=400, 
                    detail=f"File {file.filename} is not a supported format. Please upload one of: {', '.join(DOCUMENT['supported_formats'])}"
                )

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
        
        # Add messages to the chat history
        pdf_parser.add_message_to_history("user", request.question)
        pdf_parser.add_message_to_history("assistant", parsed_response.answer)
                
        return parsed_response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host=SERVER["HOST"], 
        port=SERVER["PORT"], 
        reload=SERVER["RELOAD"]
    ) 