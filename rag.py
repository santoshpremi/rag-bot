import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import faiss

class RAGSystem:
    def __init__(self):
        self.doc_db = []
        
        # 1. Load retriever
        self.retriever = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        
        # 2. Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            "distilgpt2",
            device_map="cpu"
        )
        
        # 3. Load and merge LoRA adapters
        self.model = PeftModel.from_pretrained(
            base_model,
            "./fine-tuned-model",
            device_map="cpu"
        )
        self.model = self.model.merge_and_unload()  # Critical change
        
        # 4. Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 5. Create pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            framework="pt"
        )
        
        # 6. Initialize FAISS
        self.index = faiss.IndexFlatL2(384)

    def add_documents(self, documents):
        self.doc_db.extend(documents)
        embeddings = self.retriever.encode(documents)
        self.index.add(np.array(embeddings).astype('float32'))

    def query(self, question: str, k: int = 2):
        query_embedding = self.retriever.encode(question)
        _, indices = self.index.search(
            np.array([query_embedding]).astype('float32'), 
            k
        )
        context = "\n".join([self.doc_db[i] for i in indices[0]])
        prompt = f"Question: {question}\nContext: {context}\nAnswer:"
        
        return self.generator(
            prompt,
            max_new_tokens=50,
            num_return_sequences=1,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id
        )[0]['generated_text']