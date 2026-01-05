from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os

MODEL_PATH = os.getenv("MODEL_PATH", "model")

app = FastAPI(title="XSUM Summarizer API")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
model.eval()

class SummarizeRequest(BaseModel):
    text: str

@app.post("/summarize")
def summarize(req: SummarizeRequest):
    inputs = tokenizer(req.text, return_tensors="pt", truncation=True, max_length=192)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_length=48, num_beams=1)
    summary = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    return {"summary": summary}
