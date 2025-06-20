# Document-based Chatbot

A RAG-based chatbot that allows you to chat with your PDF documents. Built with FastAPI, Streamlit, and LangChain.

## Features

- 📄 PDF Document Processing
- 🔍 OCR Support via Mathpix
- 💬 Interactive Chat Interface
- 🔄 Large Language Model (LLM) Fallback
- 🔗 RAG (Retrieval Augmented Generation)
- 🖼️ Image Analysis Support
- 📊 Document Chunking and Embedding
- 🚀 Fast API Backend
- 🎯 Streamlit Frontend

## Prerequisites

- Python 3.11+
- [Mathpix](https://mathpix.com/) account for OCR
- OpenAI API key for embeddings and chat completion

## Environment Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd ml-challenge
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/MacOS
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with your API keys:
```env
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
MATHPIX_APP_ID=your_mathpix_app_id_here
MATHPIX_APP_KEY=your_mathpix_app_key_here
MATHPIX_APP_URL="https://api.mathpix.com"
```

## Running the Application

The application consists of two parts: a FastAPI backend and a Streamlit frontend.

1. Start the FastAPI backend:
```bash
python main.py
```
The API will be available at `http://localhost:8000`

2. In a new terminal, start the Streamlit frontend:
```bash
streamlit run src/app.py
```
The web interface will be available at `http://localhost:8501`

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
├── src/
│   ├── app.py              # Streamlit frontend
│   ├── document_ingestor.py # Document processing and RAG implementation
│   ├── ocr.py             # OCR functionality
│   ├── pydantic_models.py  # Data models
│   └── prompts.yaml        # System prompts
├── main.py                 # FastAPI backend
├── requirements.txt        # Project dependencies
└── README.md              # This file
```

## Dependencies

Key dependencies include:
- FastAPI: Web framework for the backend
- Streamlit: Frontend interface
- LangChain: RAG implementation
- OpenAI: Language model and embeddings
- Mathpix: OCR processing
- FAISS: Vector storage
- PyYAML: YAML file handling
- Logfire: Logging service

## Troubleshooting

1. **File Upload Issues**
   - Ensure files are in PDF format
   - Check file size limits
   - Verify Mathpix API credentials

2. **OCR Issues**
   - Verify Mathpix credentials in `.env`
   - Check PDF quality and format
   - Look for error messages in the logs

3. **Chat Issues**
   - Verify OpenAI API key
   - Check if documents were processed successfully
   - Ensure backend is running

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 