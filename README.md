# Document Processing Service

A lightweight Python service that processes documents (PDF, DOCX, TXT, HTML, JSON, XML, Markdown) and makes them searchable using AI-powered vector embeddings. Built with ChromaDB for efficient vector storage and retrieval.

## Why Vector Document Storage?

### The Problem
Traditional document storage and search systems have limitations:
- Keyword-based search misses semantic meaning
- Different document formats require different handling
- Difficult to find related content without exact matches
- Not optimized for AI applications

### The Solution: Vector Document Storage
This service converts documents into vector embeddings, which:
1. **Captures Semantic Meaning**: Understands the actual meaning of content, not just keywords
2. **Format Agnostic**: Handles any text-based format uniformly
3. **AI-Ready**: Perfect for:
   - Semantic search
   - Document similarity matching
   - Question answering systems
   - Content recommendation
   - Automated document analysis

### Benefits
- **Better Search Results**: Find relevant documents even when keywords don't match
- **Unified Processing**: One system for all document types
- **AI Integration**: Ready for modern AI applications
- **Scalable**: Efficient storage and retrieval with ChromaDB
- **Future-Proof**: Easy to upgrade embedding models as AI advances

## How It Works

### 1. Document Processing Pipeline
```mermaid
graph LR
    A[Input Document] --> B[Text Extraction]
    B --> C[Text Chunking]
    C --> D[Vector Embedding]
    D --> E[ChromaDB Storage]
```

#### Step 1: Text Extraction
- Detects document format automatically
- Uses specialized extractors for each format:
  - PDF/DOCX/TXT/HTML: Unstructured library
  - JSON: JSON parser with formatting
  - XML: XML parser with text extraction
  - Markdown: Direct text extraction

#### Step 2: Text Chunking
- Splits text into optimal segments
- Maintains context and readability
- Ensures chunks fit embedding model limits

#### Step 3: Vector Embedding
- Converts text chunks to numerical vectors
- Uses AI to capture semantic meaning
- Normalizes vectors for consistent comparison

#### Step 4: Storage
- Stores vectors in ChromaDB
- Organizes by collections
- Maintains efficient index for fast retrieval

### 2. Search and Retrieval
```mermaid
graph LR
    A[Search Query] --> B[Query Embedding]
    B --> C[Vector Search]
    C --> D[Results Ranking]
    D --> E[Return Results]
```

## Features

- 📝 **Multi-Format Support**: PDF, DOCX, TXT, HTML, JSON, XML, Markdown
- 🔍 **Smart Search**: AI-powered semantic search
- 💾 **Efficient Storage**: Vector database storage using ChromaDB
- 🔄 **Easy Processing**: Simple command-line interface
- 📚 **Collection Based**: Organize documents in collections

## Quick Start

1. **Clone & Setup**
```bash
# Clone repository
git clone https://github.com/yourusername/document-processing-service.git
cd document-processing-service

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate # Linux/Mac

# Install dependencies
cd python-services
pip install -r requirements.txt
```

2. **Configure**
Create `.env` in root directory:
```env
GOOGLE_GEMINI_API_KEY=your_api_key_here
```

3. **Process Documents**
```bash
# Basic usage
python process_document.py ../docs/your-file.pdf collection_name

# Examples
python process_document.py ../docs/report.pdf reports
python process_document.py ../docs/notes.txt notes
python process_document.py ../docs/page.html web_docs
```

4. **Search Documents**
```bash
# Basic search
python query_documents.py "your search query" collection_name

# Example
python query_documents.py "Find information about project budgets" reports
```

## Project Structure

```
.
├── docs/                    # Document storage
├── python-services/
│   ├── process_document.py # Main processing script
│   ├── extraction.py      # Text extraction
│   ├── chunking.py       # Text chunking
│   ├── embedding_function.py # AI embeddings
│   └── query_documents.py  # Search interface
└── vector-database/        # ChromaDB storage
```

## Use Cases

1. **Enterprise Document Management**
   - Semantic search across all company documents
   - Find related documents automatically
   - Extract insights from document collections

2. **Content Management Systems**
   - Smart content recommendations
   - Related article suggestions
   - Content categorization

3. **Research and Analysis**
   - Find related research papers
   - Cross-reference documents
   - Identify patterns across documents

4. **Customer Support**
   - Quick access to relevant documentation
   - Automated answer suggestion
   - Knowledge base enhancement

## Requirements

- Python 3.9+
- Dependencies listed in `requirements.txt`
- API key for embedding model

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Support

Open an issue if you:
- Found a bug
- Want a new feature
- Need help using the service
- Want to add support for new formats

## License

MIT License - feel free to use in your own projects!
