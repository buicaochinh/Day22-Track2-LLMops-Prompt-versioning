
"""
Step 3 — RAGAS Evaluation
===========================
TASK:
  1. Run all 50 QA pairs through BOTH prompt versions, capturing answers + contexts
  2. Build EvaluationDataset with SingleTurnSample objects
  3. Evaluate with 4 RAGAS metrics: faithfulness, answer_relevancy,
     context_recall, context_precision
  4. Print a V1 vs V2 comparison table
  5. Save results to data/ragas_report.json

DELIVERABLE: faithfulness ≥ 0.8 for at least one prompt version
             + data/ragas_report.json file saved
"""

import os
import json
import warnings
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# ── 1. Imports ───────────────────────────────────────────────────────────────
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import Client

# Import QA_PAIRS
from qa_pairs import QA_PAIRS

# ── 2. Environment setup ────────────────────────────────────────────────────
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "day22-lab-eval")

llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)

llm_eval = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)

embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)

# ── 3. Pull Prompts from Hub ────────────────────────────────────────────────
client = Client()
try:
    print("📥 Pulling prompts from LangSmith Hub...")
    PROMPT_V1 = client.pull_prompt("rag-prompt-v1")
    PROMPT_V2 = client.pull_prompt("rag-prompt-v2")
except Exception as e:
    print(f"⚠️ Could not pull from Hub ({e}), using local fallback.")
    SYSTEM_V1 = "You are a concise assistant. Use context: {context}. Question: {question}"
    SYSTEM_V2 = "You are a technical expert. Use context: {context}. Question: {question}"
    PROMPT_V1 = ChatPromptTemplate.from_messages([("system", SYSTEM_V1), ("human", "{question}")])
    PROMPT_V2 = ChatPromptTemplate.from_messages([("system", SYSTEM_V2), ("human", "{question}")])

PROMPTS = {
    "v1": PROMPT_V1,
    "v2": PROMPT_V2,
}

# ── 4. Build vectorstore ────────────────────────────────────────────────────
def build_vectorstore():
    kb_path = Path("data/knowledge_base.txt")
    text = kb_path.read_text()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    return FAISS.from_texts(chunks, embeddings)

# ── 5. Run RAG and capture outputs + contexts ────────────────────────────────
def run_rag(retriever, llm, prompt, question: str) -> dict:
    """
    Returns: {"answer": str, "contexts": list[str]}
    """
    docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]
    ctx_str = "\n\n".join(contexts)
    
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": ctx_str, "question": question})
    
    return {"answer": answer, "contexts": contexts}

def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    prompt = PROMPTS[prompt_version]
    results = []
    
    print(f"\n🏃 Running 50 questions with prompt {prompt_version} ...")
    for i, qa in enumerate(QA_PAIRS, 1):
        out = run_rag(retriever, llm, prompt, qa["question"])
        results.append({
            "question":  qa["question"],
            "reference": qa["reference"],
            "answer":    out["answer"],
            "contexts":  out["contexts"],
        })
        if i % 10 == 0:
            print(f"  [{i:02d}/50] completed...")
            
    return results

# ── 6. Build RAGAS EvaluationDataset ────────────────────────────────────────
def build_ragas_dataset(rag_results: list):
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]
    return EvaluationDataset(samples=samples)

# ── 7. Run RAGAS evaluation ──────────────────────────────────────────────────
def run_ragas_eval(rag_results: list, version: str) -> dict:
    print(f"\n📐 Running RAGAS evaluation for prompt {version} (this may take a few minutes) ...")
    dataset = build_ragas_dataset(rag_results)
    
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm_eval,
        embeddings=embeddings,
    )
    
    scores = {}
    for key in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        raw = result[key]
        # raw is a list of floats
        scores[key] = float(np.mean([v for v in raw if v is not None]))
        
    print(f"Summary for {version}:")
    for k, v in scores.items():
        star = " ⭐" if k == "faithfulness" and v >= 0.8 else ""
        print(f"  {k:30s}: {v:.4f}{star}")
        
    return scores

# ── 8. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 3: RAGAS Evaluation")
    print("=" * 60)

    # 1. Build vectorstore
    vectorstore = build_vectorstore()

    # 2. Collect outputs for V1 and V2
    v1_results = collect_rag_outputs(vectorstore, "v1")
    v2_results = collect_rag_outputs(vectorstore, "v2")

    # 3. Run RAGAS evaluation
    v1_scores = run_ragas_eval(v1_results, "v1")
    v2_scores = run_ragas_eval(v2_results, "v2")

    # 4. Print comparison table
    print("\n" + "=" * 60)
    print(f"{'METRIC':30s} | {'V1 SCORE':10s} | {'V2 SCORE':10s} | {'WINNER'}")
    print("-" * 60)
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        s1, s2 = v1_scores[metric], v2_scores[metric]
        winner = "V1" if s1 > s2 else "V2"
        print(f"{metric:30s} | {s1:.4f}     | {s2:.4f}     | {winner}")

    # 5. Check faithfulness target
    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        print(f"\n✅ Target met: faithfulness = {best_faith:.4f}")
    else:
        print(f"\n⚠️  Below target ({best_faith:.4f}). Try adjusting chunking or prompts.")

    # 6. Save JSON report
    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": best_faith >= 0.8,
    }
    report_path = Path("data/ragas_report.json")
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n💾 Saved report to {report_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
