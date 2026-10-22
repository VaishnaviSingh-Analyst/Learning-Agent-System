"""
Setup verification for the Learning Agent System.

Checks imports, Ollama connectivity, model availability, embeddings, and
workflow construction. Run from the project root:
    venv\\Scripts\\python.exe scripts\\verify_setup.py
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

results = []


def check(name, fn):
    try:
        detail = fn() or ""
        print(f"{PASS} {name}{(' - ' + str(detail)) if detail else ''}")
        results.append(True)
    except Exception as e:
        print(f"{FAIL} {name} - {type(e).__name__}: {e}")
        results.append(False)


def check_imports():
    import langgraph, langchain, chromadb, sentence_transformers  # noqa: F401
    import streamlit, httpx, dotenv, langsmith  # noqa: F401
    from langchain_ollama import OllamaLLM
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    return "all core packages importable"


def check_src_imports():
    from src import (  # noqa: F401
        models, llm_service, context_processor, document_processor,
        workflow, workflow_nodes, feynman_teaching, dynamic_materials,
        web_search, custom_topics, sample_data, user_interaction,
        langsmith_config, file_upload, context_validation, multi_checkpoint, main
    )
    return "all 17 src modules importable"


def check_ollama_server():
    import httpx
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    r = httpx.get(f"{base}/api/tags", timeout=3.0)
    r.raise_for_status()
    models = [m["name"] for m in r.json().get("models", [])]
    if not models:
        raise RuntimeError("server up but no models pulled (run: ollama pull llama3.1)")
    return f"server up, models: {', '.join(models)}"


def check_embeddings():
    from sentence_transformers import SentenceTransformer
    name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    model = SentenceTransformer(name)
    vec = model.encode(["setup verification"])
    return f"'{name}' loaded, dim={len(vec[0])}"


def check_chromadb():
    import chromadb, tempfile, shutil, time
    td = tempfile.mkdtemp(prefix="chroma_verify_")
    try:
        client = chromadb.PersistentClient(path=td)
        client.get_or_create_collection(name="verify_setup")
        return "PersistentClient works"
    finally:
        for _ in range(3):
            try:
                shutil.rmtree(td, ignore_errors=True)
                break
            except Exception:
                time.sleep(0.5)


def check_workflow():
    from src.workflow import create_unified_workflow, create_question_generation_workflow
    w1 = create_unified_workflow()
    w2 = create_question_generation_workflow()
    return f"unified + question_generation workflows compiled"


def check_llm_invoke():
    import asyncio
    from src.llm_service import LLMService
    svc = LLMService()
    resp = asyncio.run(asyncio.to_thread(svc.llm.invoke, "Reply with exactly: OK"))
    text = str(getattr(resp, "content", resp)).strip()
    if not text:
        raise RuntimeError("empty LLM response")
    return f"LLM responded ({len(text)} chars)"


print("=" * 60)
print("Learning Agent System - Setup Verification")
print("=" * 60)

check("Core package imports", check_imports)
check("src module imports", check_src_imports)
check("ChromaDB persistent client", check_chromadb)
check("Embedding model load", check_embeddings)
check("Ollama server + model", check_ollama_server)
check("Workflow compilation", check_workflow)
check("LLM invoke round-trip", check_llm_invoke)

ok = all(results)
print("=" * 60)
print(f"RESULT: {sum(results)}/{len(results)} checks passed "
      + ("- ALL GOOD" if ok else "- see failures above"))
sys.exit(0 if ok else 1)
