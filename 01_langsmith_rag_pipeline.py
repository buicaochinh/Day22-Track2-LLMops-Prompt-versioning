
"""
Step 1 — LangSmith-instrumented RAG Pipeline
=============================================
TASK:
  1. Load your dataset, split into chunks, index with FAISS
  2. Build a RAG chain: retriever → prompt → LLM → output parser
  3. Decorate the query function with @traceable so every call is traced
  4. Run all 50 questions → generates ≥ 50 LangSmith traces

DELIVERABLE: Open https://smith.langchain.com and confirm traces appear.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── 1. Environment setup ────────────────────────────────────────────────────
load_dotenv()

# Set LangSmith environment variables BEFORE importing LangChain
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "day22-lab")
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# ── 2. LangChain + LangSmith imports ────────────────────────────────────────
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

# Import QA_PAIRS for testing
from qa_pairs import QA_PAIRS

# ── 3. LLM and Embeddings ───────────────────────────────────────────────────
llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)

embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)

# ── 4. Build FAISS vector store ─────────────────────────────────────────────
def build_vectorstore():
    """
    Load the knowledge base, split into chunks, embed and index with FAISS.
    """
    kb_path = Path("data/knowledge_base.txt")
    if not kb_path.exists():
        raise FileNotFoundError(f"Knowledge base not found at {kb_path}")
        
    text = kb_path.read_text()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    print(f"✅ Split knowledge base into {len(chunks)} chunks")

    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

# ── 5. RAG prompt template ──────────────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the context below to answer the user's question accurately. If you don't know the answer based on the context, say that you don't know.\n\nContext:\n{context}"),
    ("human",  "{question}"),
])

# ── 6. Build the RAG chain ──────────────────────────────────────────────────
def build_rag_chain(vectorstore):
    """
    Build a LangChain RAG chain using LCEL.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever

# ── 7. Traced query function ────────────────────────────────────────────────
@traceable(name="rag-query", tags=["rag", "step1"])
def ask(chain, question: str) -> str:
    """
    Run the RAG chain on a single question and trace it in LangSmith.
    """
    return chain.invoke(question)

# ── 9. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 1: LangSmith RAG Pipeline")
    print("=" * 60)

    # Build the vectorstore
    vectorstore = build_vectorstore()

    # Build the RAG chain
    chain, _ = build_rag_chain(vectorstore)

    # Run all 50 questions
    print(f"\n🚀 Running {len(QA_PAIRS)} questions through the RAG pipeline...")
    
    for i, qa in enumerate(QA_PAIRS, 1):
        question = qa["question"]
        answer = ask(chain, question)
        
        # Print progress
        print(f"[{i:02d}/{len(QA_PAIRS)}] Q: {question[:60]}...")
        # print(f"       A: {answer[:80]}...") # Optional: uncomment to see answers

    print("\n" + "=" * 60)
    print(f"✅ {len(QA_PAIRS)} traces sent to LangSmith project '{os.environ['LANGCHAIN_PROJECT']}'")
    print("   Open https://smith.langchain.com to view traces.")
    print("=" * 60)

if __name__ == "__main__":
    main()
