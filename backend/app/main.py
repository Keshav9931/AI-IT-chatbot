from fastapi import FastAPI
from backend.app.rag_engine import RAGEngine

app = FastAPI()

rag_engine = None


@app.on_event("startup")
async def startup_event():
    global rag_engine

    print("🚀 Starting API...")

    rag_engine = RAGEngine(
        data_path="data/knowledge_base.csv",
        qdrant_url="http://localhost:6333",
        collection_name="it_helpdesk",
        google_api_key="AQ.Ab8RN6L1Qo7M7se21BTE-LyGutmTOIKz5NyWFBtmePticEaYvA"
    )

    await rag_engine.initialize()

    print("✅ System Ready")


@app.get("/")
def home():
    return {"message": "AI IT Helpdesk API running 🚀"}


@app.post("/query")
async def query(payload: dict):
    global rag_engine

    question = payload.get("question")

    if not question:
        return {"error": "Question is required"}

    answer = rag_engine.query(question)

    return {
        "question": question,
        "answer": answer
    }
