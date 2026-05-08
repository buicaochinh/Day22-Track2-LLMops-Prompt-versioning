
"""
Step 2 — Prompt Hub & A/B Routing
==================================
TASK:
  1. Write two system prompts (V1: concise, V2: structured)
  2. Push both to LangSmith Prompt Hub
  3. Pull them back to use in your chain
  4. Implement deterministic A/B routing using hashlib.md5
  5. Run 50 questions; log which version was used

DELIVERABLE: 
  - LangSmith Prompt Hub shows 2 versions
  - Console shows [prompt-v1] or [prompt-v2] for each query
"""

import os
import hashlib
from pathlib import Path
from dotenv import load_dotenv
from langsmith import Client, traceable
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import QA_PAIRS
from qa_pairs import QA_PAIRS

# ── 1. Environment setup ────────────────────────────────────────────────────
load_dotenv()
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "day22-lab-ab-testing")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT

# ── 2. Initialize LangSmith Client and LLM ──────────────────────────────────
client = Client(api_key=LANGSMITH_API_KEY)

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

# ── 3. Define Two System Prompts ─────────────────────────────────────────────
# Prompt V1: Concise and direct
SYSTEM_V1 = """You are a concise assistant. Answer the question using ONLY the provided context. 
Be brief and do not include unnecessary information.

Context:
{context}"""

# Prompt V2: Structured and detailed
SYSTEM_V2 = """You are a technical expert. Provide a detailed and structured answer based on the context provided.
Use bullet points or numbered lists if appropriate. Ensure technical terms are correctly used.
If the context doesn't contain the answer, explain what is missing.

Context:
{context}"""

PROMPT_V1 = ChatPromptTemplate.from_messages([("system", SYSTEM_V1), ("human", "{question}")])
PROMPT_V2 = ChatPromptTemplate.from_messages([("system", SYSTEM_V2), ("human", "{question}")])

# ── 4. Push to Prompt Hub ───────────────────────────────────────────────────
def push_prompts():
    print("📤 Pushing prompts to LangSmith Prompt Hub...")
    # Note: Replace 'my-rag-prompt' with a unique name if needed
    client.push_prompt("rag-prompt-v1", object=PROMPT_V1)
    client.push_prompt("rag-prompt-v2", object=PROMPT_V2)
    print("✅ Prompts pushed successfully.")

# ── 5. A/B Routing Logic ────────────────────────────────────────────────────
def get_prompt_version(request_id: str) -> str:
    """
    Deterministically route to v1 or v2 based on the request_id.
    """
    h = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    return "rag-prompt-v1" if h % 2 == 0 else "rag-prompt-v2"

# ── 6. Build Vector Store (Same as Step 1) ──────────────────────────────────
def build_vectorstore():
    kb_path = Path("data/knowledge_base.txt")
    text = kb_path.read_text()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    return FAISS.from_texts(chunks, embeddings)

# ── 7. Traced RAG Execution with A/B Routing ────────────────────────────────
@traceable(name="ab-routed-query")
def ask_with_router(chain_map, question: str, request_id: str) -> tuple:
    version_name = get_prompt_version(request_id)
    chain = chain_map[version_name]
    answer = chain.invoke(question)
    return answer, version_name

# ── 8. Main Execution ───────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 2: Prompt Hub & A/B Routing")
    print("=" * 60)

    # 1. Push prompts to Hub
    push_prompts()

    # 2. Pull prompts back from Hub (to ensure we use versioned templates)
    print("\n📥 Pulling prompts back from Hub...")
    pulled_v1 = client.pull_prompt("rag-prompt-v1")
    pulled_v2 = client.pull_prompt("rag-prompt-v2")

    # 3. Setup Vector Store and Chains
    vectorstore = build_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Build two chains, one for each pulled prompt
    chains = {
        "rag-prompt-v1": (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | pulled_v1
            | llm
            | StrOutputParser()
        ),
        "rag-prompt-v2": (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | pulled_v2
            | llm
            | StrOutputParser()
        )
    }

    # 4. Run questions through the router
    print(f"\n🚦 Running {len(QA_PAIRS)} questions with deterministic A/B routing...")
    
    v1_count = 0
    v2_count = 0

    for i, qa in enumerate(QA_PAIRS, 1):
        request_id = f"req_{i}" # Using index as unique request ID
        answer, version = ask_with_router(chains, qa["question"], request_id)
        
        if version == "rag-prompt-v1":
            v1_count += 1
        else:
            v2_count += 1
            
        print(f"[{i:02d}/50] [{version}] Q: {qa['question'][:50]}...")

    print("\n" + "=" * 60)
    print(f"📊 Routing Results: V1 = {v1_count}, V2 = {v2_count}")
    print(f"✅ All traces sent to LangSmith project '{os.environ['LANGCHAIN_PROJECT']}'")
    print("=" * 60)

if __name__ == "__main__":
    main()
