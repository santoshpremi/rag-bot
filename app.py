from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag import RAGSystem
import uvicorn

app = FastAPI(title="RAG Chatbot API")

# Initialize RAG system
try:
    rag = RAGSystem()
    DOCUMENTS = [
    "COVID-19 is an infectious disease caused by the SARS-CoV-2 virus. (WHO Fact Sheet)",
    "Most common symptoms: fever, cough, tiredness, loss of taste or smell. (WHO Report 2023)",
    "Vaccines approved by WHO include Pfizer-BioNTech, Moderna, AstraZeneca, etc. (WHO Vaccine Guidance)",
    "Primary transmission methods: respiratory droplets, surface contact. (WHO Transmission Guidelines)",
    "Prevention measures: vaccination, masks, hand hygiene. (WHO Prevention Protocol)"
    ]
    rag.add_documents(DOCUMENTS)
except Exception as e:
    print(f"❌ Initialization failed: {str(e)}")
    raise SystemExit(1)

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
async def read_root():
    return {
        "message": "Welcome to the COVID-19 Chatbot API",
        "docs": "Visit /docs for API documentation",
        "endpoint": "POST /ask with {'question': 'your question'}"
    }

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        answer = rag.query(request.question)
        return {
            "question": request.question,
            "answer": answer.split("Answer:")[-1].strip(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1
    )