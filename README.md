# Document-based Chatbot

A RAG-based chatbot that allows you to chat with your PDF documents. Built with FastAPI, Streamlit, and LangChain.

## Features

- 📄 PDF Document Processing
- 🔍 OCR Support via Mathpix
- 💬 Interactive Chat Interface
- 🔄 Large Language Model (LLM) Fallback (OpenAI GPT-4o-mini with Google Gemini as fallback)
- 🔗 RAG (Retrieval Augmented Generation)
- 🖼️ Image Analysis Support
- 📊 Document Chunking and Embedding
- 🚀 Fast API Backend
- 🎯 Streamlit Frontend
- 🐳 Docker Support

## Prerequisites

- Python 3.11+
- [Mathpix](https://mathpix.com/) account for OCR
- OpenAI API key for embeddings and chat completion
- Google API key for LLM fallback (optional)

## Environment Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd pdf-rag-chatbot
```

2. Create a `.env` file in the root directory with your API keys:
```env
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
MATHPIX_APP_ID=your_mathpix_app_id_here
MATHPIX_APP_KEY=your_mathpix_app_key_here
MATHPIX_APP_URL="https://api.mathpix.com"
```

## Running the Application

You have two options to run the application: using Docker (recommended) or running locally.

### Option 1: Using Docker (Recommended)

1. Make sure Docker and Docker Compose are installed on your system
2. Run the application:
```bash
docker-compose up --build
```

The application will be available at:
- Backend API: `http://localhost:8000`
- Frontend (Streamlit): `http://localhost:8501`

### Option 2: Running Locally

#### Backend Setup

1. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/MacOS
python -m venv venv
source venv/bin/activate
```

2. Install backend dependencies:
```bash
cd src
pip install -r requirements.txt
```

3. Start the FastAPI backend:
```bash
python main.py
```
The API will be available at `http://localhost:8000`

#### Frontend Setup

1. In a new terminal, navigate to the streamlit directory:
```bash
cd streamlit
```

2. Install frontend dependencies:
```bash
pip install -r requirements.txt
```

3. Start the Streamlit frontend:
```bash
streamlit run app.py
```
The web interface will be available at `http://localhost:8501`

**Note:** When running locally, you need to update the API_URL in `streamlit/app.py` from `http://backend:8000` to `http://localhost:8000`.

## Usage

1. Open your browser and navigate to `http://localhost:8501`
2. Use the sidebar to upload PDF documents
3. Click "Process Documents" to analyze them
4. Start asking questions in the chat interface
5. View answers with relevant references from your documents

## API Endpoints

- `GET /`: Health check endpoint
- `POST /documents`: Upload and process PDF documents
- `POST /question`: Ask questions about the processed documents

## Examples

### API Examples

1. **Health Check**
```bash
curl http://localhost:8000/
```
Response:
```json
{
    "message": "Tractian - Document-based Chatbot"
}
```

2. **Upload Documents**
```bash
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: multipart/form-data" \
  -F "files=@/path/to/your/document.pdf"
```
Response:
```json
{
    "message": "Documents processed successfully",
    "documents_indexed": 1,
    "total_chunks": 15
}
```

3. **Ask a Question**
```bash
curl -X POST http://localhost:8000/question \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the recommended maintenance interval?"}'
```
Response:
```json
{
    "answer": "According to the manual, the recommended maintenance interval is every 3 months or 500 operating hours, whichever comes first. This includes inspecting bearings, lubricating moving parts, and checking belt tension.",
    "references": [
        "Regular maintenance should be performed every 3 months or 500 operating hours.",
        "Maintenance schedule diagram showing 3-month interval checkpoints for bearings and belt inspection.",
        "Lubrication of moving parts must be done at 500-hour intervals to ensure optimal performance."
    ]
}
```

## Project Structure

```
ml-challenge/
├── src/                           # Backend source code
│   ├── main.py                    # FastAPI backend application
│   ├── config.py                  # Configuration settings
│   ├── requirements.txt           # Backend dependencies
│   ├── Dockerfile                 # Backend Docker configuration
│   ├── core/                      # Core functionality
│   │   ├── document_ingestor.py   # Document processing and RAG implementation
│   │   ├── ocr.py                 # OCR functionality
│   │   ├── schemas.py             # Pydantic data models
│   │   └── prompts.yml            # System prompts
│   └── middleware/                # Custom middleware
│       └── logging_middleware.py  # Request logging middleware
├── streamlit/                     # Frontend source code
│   ├── app.py                     # Streamlit frontend application
│   ├── requirements.txt           # Frontend dependencies
│   └── Dockerfile                 # Frontend Docker configuration
├── docker-compose.yml             # Docker Compose configuration
├── .env                          # Environment variables (create this)
└── README.md                     # This file
```

## Dependencies

### Backend Dependencies
- FastAPI: Web framework for the backend
- LangChain: RAG implementation and LLM integration
- OpenAI: Language model and embeddings
- Google GenAI: Fallback language model
- Mathpix: OCR processing
- FAISS: Vector storage
- PyYAML: YAML file handling
- Uvicorn: ASGI server

### Frontend Dependencies
- Streamlit: Frontend interface
- Requests: HTTP client for API communication

## Configuration

The application uses a configuration system located in `src/config.py`. Key configurations include:

- **Server Settings**: Host, port, and server details
- **LLM Settings**: Primary (OpenAI) and fallback (Google) models
- **Document Processing**: Chunk size and overlap settings
- **Image Processing**: OCR and image handling parameters

## Troubleshooting

1. **Docker Issues**
   - Ensure Docker and Docker Compose are running
   - Check if ports 8000 and 8501 are available
   - Verify the `.env` file exists and contains valid API keys

2. **File Upload Issues**
   - Ensure files are in PDF format
   - Check file size limits
   - Verify Mathpix API credentials

3. **OCR Issues**
   - Verify Mathpix credentials in `.env`
   - Check PDF quality and format
   - Look for error messages in the logs

4. **Chat Issues**
   - Verify OpenAI API key
   - Check if documents were processed successfully
   - Ensure backend is running

5. **Local Development Issues**
   - When running locally, update `API_URL` in `streamlit/app.py`
   - Ensure both backend and frontend are running
   - Check virtual environment activation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 