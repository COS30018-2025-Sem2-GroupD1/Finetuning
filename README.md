# MedBot RAG (MongoDB Atlas + FastAPI + Gemini)

A medical chatbot powered by **Retrieval-Augmented Generation (RAG)**:
- **MongoDB Atlas Vector Search** for embedding storage and retrieval.
- **Sentence-Transformers MiniLM-L6-v2 (384d)** for vector generation.
- **Gemini API** to generate final answers from retrieved context.
- **FastAPI** for serving the `/chat` endpoint.

---

## 🚀 Setup

### 1. Environment
Install dependencies:
```bash
pip install -r requirements.txt
Create a .env file and fill in:

env
MONGODB_URI=mongodb+srv://<user>:<pass>@<cluster>/?retryWrites=true&w=majority
GEMINI_API_KEY=<your_gemini_api_key>


2. Create Vector Search Index in MongoDB Atlas
Go to Atlas → Vector Search → Create Index → JSON Editor.
Select:

Database: medbot
Collection: medical_chunks
Index Name: chunks_vector_index

Paste the following JSON:
json
{
  "fields": [
    { "type": "vector", "path": "vector", "numDimensions": 384, "similarity": "cosine" },
    { "type": "filter", "path": "parent_id" },
    { "type": "filter", "path": "chunk_index" },
    { "type": "filter", "path": "task" },
    { "type": "filter", "path": "source" },
    { "type": "filter", "path": "meta.tags" }
  ]
}
3. Ingest data (Chunk → Embed → Upsert)
Prepare your dataset in .jsonl format.

Run ingest:

bash
python -m scripts.run_ingest \
  --jsonl data/processed/your_dataset.jsonl \
  --target medbot.medical_chunks \
  --max-chars 1000 --overlap 150 \
  --tags chunked,MiniLM
When finished you’ll see:

yaml
Embedded vectors: XXXXX | Written to: medbot.medical_chunks
Each raw document is split into multiple chunks:
_id = docId#<chunk_index>, with fields parent_id, chunk_index, vector, task, source, meta.tags.

4. Run the FastAPI server
bash
uvicorn scripts.main:app --reload --port 8000
Endpoints:

POST /chat → Chat with the RAG pipeline.

GET /healthz → Health check.

POST /query → Alias for /chat.

5. Example request
bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Patient with persistent cough and fever, next step?",
    "top_k": 7,
    "numCandidates": 200
  }'
Example response:

json
{
  "answer": "...",
  "contexts": [
    {
      "_id": "doc123#0",
      "parent_id": "doc123",
      "chunk_index": 0,
      "score": 0.91,
      "text": "..."
    },
    ...
  ],
  "stats": {
    "insufficient": false,
    "max": 0.91,
    "mean": 0.73
  }
}
⚙️ Configuration (config.py)
python

INDEX_NAME = "chunks_vector_index"
NUM_CANDIDATES = 200
TOP_K = 7
THRESHOLD = 0.35
DEFAULT_FILTERS = {
    "task": "medical_dialogue",
    "source": "healthcaremagic"
}
SAVE_VECTORS = True
📂 Project structure
pgsql
scripts/
 ├─ run_ingest.py  
 ├─ main.py         
src/core/
 ├─ db.py        
 ├─ embed.py      
 ├─ gemini.py      
 └─ rag.py           
      
📝 Notes
Ingest multiple datasets into the same collection (medical_chunks), distinguishing them by source/task/tags.

If you prefer to keep datasets isolated, create separate collections under medbot and build an index for each.

Current setup uses MiniLM-L6-v2 (384d). If you switch models, update numDimensions in Atlas Index.