# DocuChat AI - RAG Document System

A production-grade RAG (Retrieval-Augmented Generation) system with a beautiful single-page frontend. Upload documents, get AI-powered summaries, and chat with your documents.

![DocuChat AI](https://img.shields.io/badge/AI-Powered-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green) ![OpenAI](https://img.shields.io/badge/OpenAI-API-orange)

## ✨ Features

- **📤 Multi-Format Upload**: PDF, Word, images (with OCR), and text files
- **📝 Auto-Summarization**: AI generates summaries and key points
- **💬 Document Chat**: Ask questions and get AI-powered answers
- **🔍 Similarity Search**: Find relevant passages in your documents
- **🗑️ Auto-Cleanup**: Documents automatically expire after 15 days
- **⚡ Fast & Async**: Parallel processing for optimal performance

## 🖥️ Screenshots

The frontend features:
- Dark mode interface with gradient accents
- Drag-and-drop file upload
- Real-time processing status
- Beautiful summary display with key points
- Interactive chat interface

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key
- Supabase account (free tier works)

### Local Development

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/docuchat-ai.git
cd docuchat-ai

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp env.example .env
# Edit .env with your API keys

# 5. Run the server
python run.py --reload
```

Visit http://localhost:8000 to see the app!

## 🔧 Configuration

### Environment Variables

Create a `.env` file with:

```env
# Required
OPENAI_API_KEY=sk-your-api-key-here
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
SUPABASE_SERVICE_KEY=your-service-role-key

# Optional (with defaults)
CHROMA_PERSIST_DIRECTORY=./data/chroma
UPLOAD_DIRECTORY=./data/uploads
DOCUMENT_RETENTION_DAYS=15
MAX_FILE_SIZE_MB=25
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### Supabase Setup

1. Create a new Supabase project at [supabase.com](https://supabase.com)

2. Run this SQL in the SQL Editor:

```sql
-- Documents table
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename TEXT NOT NULL,
    file_type TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    status TEXT DEFAULT 'pending',
    chunk_count INTEGER DEFAULT 0,
    storage_path TEXT,
    error_message TEXT,
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Summaries table
CREATE TABLE document_summaries (
    document_id UUID PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    summary TEXT NOT NULL,
    key_points JSONB DEFAULT '[]',
    word_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Add new columns for enhanced summary features (run after initial setup)
ALTER TABLE document_summaries 
ADD COLUMN IF NOT EXISTS summary_paragraphs JSONB DEFAULT '[]'::jsonb;

ALTER TABLE document_summaries 
ADD COLUMN IF NOT EXISTS personal_info TEXT DEFAULT '';

-- Indexes
CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_documents_expires_at ON documents(expires_at);
```

3. Create a storage bucket:
   - Go to Storage → Create new bucket
   - Name: `document-files`
   - Set to private

## 🌐 Deploy for Free

### Option 1: Render (Recommended)

1. Fork this repository
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Render will auto-detect `render.yaml`
5. Add environment variables in the dashboard:
   - `OPENAI_API_KEY`
   - `SUPABASE_URL`
   - `SUPABASE_KEY`
   - `SUPABASE_SERVICE_KEY`
6. Deploy!

**Free tier notes:**
- 750 hours/month free
- Sleeps after 15 min inactivity
- First request after sleep takes ~30s

### Option 2: Railway

1. Fork this repository
2. Go to [railway.app](https://railway.app) → New Project
3. Choose "Deploy from GitHub repo"
4. Add environment variables:
   - `OPENAI_API_KEY`
   - `SUPABASE_URL`
   - `SUPABASE_KEY`
   - `SUPABASE_SERVICE_KEY`
5. Deploy!

**Free tier notes:**
- $5 free credit/month
- No sleep on free tier
- Better for demos

### Do You Need Docker?

**No!** Both Render and Railway can deploy directly from your Python code:
- Render uses `render.yaml` or auto-detection
- Railway uses `nixpacks.toml` or auto-detection

Docker is optional but included if you need it for other platforms.

## 📡 API Endpoints

### Documents
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/documents/upload` | Upload a document |
| GET | `/api/v1/documents/{id}` | Get document status |
| GET | `/api/v1/documents` | List all documents |
| DELETE | `/api/v1/documents/{id}` | Delete a document |
| GET | `/api/v1/documents/{id}/summary` | Get summary |

### Chat
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/chat` | Chat with document |
| POST | `/api/v1/chat/stream` | Stream chat response |
| POST | `/api/v1/chat/search` | Search passages |

### Health
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Full health check |
| GET | `/health/live` | Liveness probe |
| GET | `/health/ready` | Readiness probe |

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    Frontend (Single Page)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │    Upload    │  │   Summary    │  │      Chat        │  │
│  │    Zone      │  │   Display    │  │    Interface     │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   OpenAI    │  │  ChromaDB   │  │     Supabase        │ │
│  │ Embeddings  │  │   Vector    │  │  Metadata + Files   │ │
│  │  + Chat     │  │   Store     │  │                     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└────────────────────────────────────────────────────────────┘
```

## 💰 Cost Estimation

**Free tier usage (OpenAI):**
- Embeddings: ~$0.0001 per 1K tokens
- Chat: ~$0.002 per 1K tokens (GPT-3.5)
- A typical document: ~$0.01-0.05

**Monthly estimate for light use:**
- 50 documents/month: ~$2-5
- OpenAI has $5 free credit for new accounts

## 🛠️ Development

```bash
# Run with auto-reload
python run.py --reload

# Run tests
pytest

# Format code
black .
isort .
```

## 📝 License

MIT License - feel free to use for personal or commercial projects!

---

**Made with ❤️ using FastAPI, LangChain, and OpenAI**