import os
import json
import hashlib
import pathlib
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
import ollama  

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Config
PDF_FOLDER = "../KBCC_pytesseract"
OUTPUT_FOLDER = "./contextual_chunks"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MODEL_NAME = ""  
OLLAMA_BASE_URL = "http://ollama123:11434" 
LOG_FILE = "contextual_chunking.log"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
ollama.base_url = OLLAMA_BASE_URL

# Logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
llm_logger = logging.getLogger("llm_calls")
llm_logger.setLevel(logging.DEBUG)
llm_handler = logging.FileHandler(LLM_LOG_FILE, mode="w", encoding="utf-8")
llm_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
llm_logger.addHandler(llm_handler)
logging.info("Starting contextual chunk creation.")


# Loading pdfs
pdf_files = list(Path(PDF_FOLDER).rglob("*.pdf"))
docs = []
for pdf_path in tqdm(pdf_files, desc="Loading PDFs...", unit="pdf"):
    loader = PyPDFLoader(str(pdf_path))
    docs.extend(loader.load())

logging.info(f"Loaded {len(docs)} PDF pages from {len(pdf_files)} files.")


# Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    add_start_index=True,
)
chunks = splitter.split_documents(docs)
logging.info(f"Created {len(chunks)} text chunks before contextualization.")



by_page = defaultdict(list)
for d in chunks:
    by_page[(d.metadata.get("source"), d.metadata.get("page"))].append(d)

for key in by_page:
    by_page[key].sort(key=lambda d: (d.metadata.get("start_index") or 0))


# Prompt
def build_prompt(whole_doc: str, chunk_text: str) -> str:
    return f"""
<document>
{whole_doc}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_text}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. 
Answer in French, only with the succinct context and nothing else.
""".strip()


def get_context_from_ollama(whole_doc: str, chunk_text: str, model=MODEL_NAME) -> str:
    prompt = build_prompt(whole_doc, chunk_text)
    try:
        response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"].strip()
    except Exception as e:
        logging.error(f"Ollama call failed: {e}")
        return ""


# Context
enriched = []
ids = []

for (src, page), page_chunks in tqdm(by_page.items(), desc="Contextualizing pages..."):
    # Get full text of the page as "whole document" context
    whole_doc = " ".join([c.page_content for c in page_chunks])

    for idx, d in enumerate(page_chunks):
        src_path = d.metadata.get("source", "")
        file_name = pathlib.Path(src_path).name if src_path else "unknown"
        start_index = d.metadata.get("start_index") or 0
        end_index = start_index + len(d.page_content)
      
        h = hashlib.sha256()
        h.update(str(src_path).encode("utf-8"))
        h.update(str(page).encode("utf-8"))
        h.update(str(start_index).encode("utf-8"))
        h.update(str(d.page_content[:200]).encode("utf-8"))
        chunk_id = h.hexdigest()[:16]

       
        context_text = get_context_from_ollama(whole_doc, d.page_content)
        contextualized_chunk = (
            f"{context_text}\n\n{d.page_content}" if context_text else d.page_content
        )

        d.metadata.update(
            {
                "file_name": file_name,
                "chunk_index": idx,
                "total_chunks_per_page": len(page_chunks),
                "end_index": end_index,
                "ingested_at": datetime.now().isoformat(),
                "chunk_id": chunk_id,
                "context_generated": bool(context_text),
            }
        )

        enriched.append(
            {
                "id": chunk_id,
                "text": contextualized_chunk,
                "metadata": d.metadata,
            }
        )
        ids.append(chunk_id)

logging.info(f"Generated contextualized text for {len(enriched)} chunks.")


# Save
output_path = Path(OUTPUT_FOLDER) / f"contextual_chunks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(enriched, f, ensure_ascii=False, indent=2)

logging.info(f"Saved contextualized chunks to {output_path}")
print(f"Saved contextualized chunks to: {output_path}")
print(f"🪵 Log file: {LOG_FILE}")


-----------

def get_context_from_ollama(whole_doc: str, chunk_text: str, model: str, metadata: dict) -> str:
    chunk_info = f"{metadata.get('file_name', 'unknown')} | page {metadata.get('page')} | chunk {metadata.get('chunk_index')}"
    prompt = build_prompt(whole_doc, chunk_text)

    try:
        response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
        result = response["message"]["content"].strip()
        llm_logger.info(f"[SUCCESS] {chunk_info}\nPROMPT LEN: {len(prompt)}\nRESPONSE:\n{result}\n{'='*80}")
        return result
    except Exception as e:
        err_msg = str(e)
        llm_logger.warning(f"[RETRY] {chunk_info} initial Ollama call failed: {err_msg}")

        
        max_attempts = 5
        step_ratio = 0.8  
        trimmed = whole_doc

        for attempt in range(1, max_attempts + 1):
            trimmed = trimmed[: int(len(trimmed) * step_ratio)]
            prompt = build_prompt(trimmed, chunk_text)
            try:
                response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
                result = response["message"]["content"].strip()
                llm_logger.info(f"[SUCCESS after {attempt}] {chunk_info}\nPROMPT LEN: {len(prompt)}\nRESPONSE:\n{result}\n{'='*80}")
                return result
            except Exception as e2:
                llm_logger.warning(f"[FAIL {attempt}] {chunk_info} retry failed: {e2}")
                continue

        llm_logger.error(f"[GIVEUP] {chunk_info} failed after {max_attempts} retries.")
        return ""


# Write to jsonl
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_PATH = Path(OUTPUT_FOLDER) / f"contextual_chunks_{timestamp}.jsonl"

with open(OUTPUT_PATH, "a", encoding="utf-8") as fout:
    total_chunks = 0

    for (src, page), page_chunks in tqdm(by_page.items(), desc="Contextualizing pages..."):
        full_doc = " ".join([c.page_content for c in page_chunks])

        for idx, d in enumerate(page_chunks):
            src_path = d.metadata.get("source", "")
            file_name = pathlib.Path(src_path).name if src_path else "unknown"
            start_index = d.metadata.get("start_index") or 0
            end_index = start_index + len(d.page_content)

            h = hashlib.sha256()
            h.update(str(src_path).encode("utf-8"))
            h.update(str(page).encode("utf-8"))
            h.update(str(start_index).encode("utf-8"))
            h.update(str(d.page_content[:200]).encode("utf-8"))
            chunk_id = h.hexdigest()[:16]

            d.metadata.update(
                {
                    "file_name": file_name,
                    "chunk_index": idx,
                    "page": page,
                    "total_chunks_per_page": len(page_chunks),
                    "start_index": start_index,
                    "end_index": end_index,
                    "ingested_at": datetime.now().isoformat(),
                    "chunk_id": chunk_id,
                }
            )

            # Call Ollama dynamically
            context_text = get_context_from_ollama(full_doc, d.page_content, model=MODEL_NAME, metadata=d.metadata)
            contextualized_chunk = f"{context_text}\n\n{d.page_content}" if context_text else d.page_content

            # Save immediately (crash-safe)
            fout.write(json.dumps({"id": chunk_id, "text": contextualized_chunk, "metadata": d.metadata}, ensure_ascii=False) + "\n")
            fout.flush()

            total_chunks += 1

logging.info(f" Completed contextualization for {total_chunks} chunks.")
logging.info(f"Chunks saved incrementally to {OUTPUT_PATH}")


-----
from langchain_community.vectorstores import PGVector
from langchain_community.embeddings import OllamaEmbeddings



embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)

vectorstore = PGVector(
    connection_string=CONNECTION_STRING,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
)


print("📡 Connecting to Postgres...")
conn = psycopg2.connect(CONNECTION_STRING)
cur = conn.cursor()

cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public';")
tables = [r[0] for r in cur.fetchall()]
print("\n📋 Tables in database:", tables)

if COLLECTION_NAME not in tables:
    print(f"⚠️ Table '{COLLECTION_NAME}' not found! Check your collection name.")
else:
    cur.execute(f"SELECT COUNT(*) FROM {COLLECTION_NAME};")
    count = cur.fetchone()[0]
    print(f"✅ Total chunks stored in '{COLLECTION_NAME}': {count}")

    cur.execute(f"SELECT id, LEFT(text, 200), metadata FROM {COLLECTION_NAME} LIMIT 3;")
    rows = cur.fetchall()
    print("\n🔍 Sample entries:")
    for r in rows:
        print(f"\nID: {r[0]}")
        print(f"Text preview: {r[1][:150].replace('\\n',' ')}...")
        print(f"Metadata: {r[2]}")

cur.close()
conn.close()

print("\n🧠 Testing similarity search via LangChain...")

embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
vectorstore = PGVector(
    connection_string=CONNECTION_STRING,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
)

query = "ACME Corp revenue growth in Q2 2023"
docs = vectorstore.similarity_search(query, k=3)

for i, doc in enumerate(docs, 1):
    print(f"\nResult {i}:")
    print(f"Text: {doc.page_content[:250]}...")
    print(f"Metadata: {doc.metadata}")




---------------
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)  # allow all levels internally

# File handler → full logs (INFO and above)
file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

# Console handler → only warnings & errors
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.WARNING)
console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

# Apply both handlers
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

# Dedicated LLM call log (always full detail)
llm_logger = logging.getLogger("llm_calls")
llm_logger.setLevel(logging.DEBUG)
llm_file_handler = logging.FileHandler(LLM_LOG_FILE, mode="w", encoding="utf-8")
llm_file_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
llm_logger.addHandler(llm_file_handler)

# ---------------- GROUP BY WHOLE PDF ---------------- #
by_pdf = defaultdict(list)
for d in chunks:
    by_pdf[d.metadata.get("source")].append(d)

for src in by_pdf:
    by_pdf[src].sort(key=lambda d: ((d.metadata.get("page") or 0), (d.metadata.get("start_index") or 0)))


# ---------------- PROMPT BUILDER ---------------- #
def build_prompt(whole_doc: str, chunk_text: str) -> str:
    return f"""
<document>
{whole_doc}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_text}
</chunk>

Please give a short, succinct context (50–100 tokens) to situate this chunk within the overall document for improving retrieval accuracy.
Answer ONLY with the succinct context and nothing else.
""".strip()


# ---------------- OLLAMA CALL (per-chunk) ---------------- #
def get_context_from_ollama(whole_doc: str, chunk_text: str, model: str, metadata: dict) -> str:
    """
    Call Ollama once per chunk with WHOLE-PDF context.
    If the prompt is too long, retry by trimming the end of whole_doc.
    The chunk text is never truncated.
    """
    chunk_info = f"{metadata.get('file_name', 'unknown')} | page {metadata.get('page')} | chunk {metadata.get('chunk_index')}"
    trimmed = whole_doc
    max_attempts = 5
    step_ratio = 0.8  # trim to 80% length each retry

    for attempt in range(0, max_attempts + 1):
        prompt = build_prompt(trimmed, chunk_text)
        try:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"num_ctx": NUM_CTX},
            )
            result = response["message"]["content"].strip()
            llm_logger.info(
                f"[SUCCESS{' after '+str(attempt)+' retry' if attempt else ''}] {chunk_info}\n"
                f"PROMPT:\n{prompt}\n---\nRESPONSE:\n{result}\n{'='*80}"
            )
            return result
        except Exception as e:
            llm_logger.warning(f"[RETRY {attempt}] {chunk_info} failed: {e}")
            if attempt < max_attempts:
                new_len = max(1000, int(len(trimmed) * step_ratio))
                if new_len >= len(trimmed):
                    break
                trimmed = trimmed[:new_len]  # trim from end
                continue
            llm_logger.error(f"[GIVEUP] {chunk_info} after {max_attempts} retries.")
            return ""


# ---------------- STREAMING JSONL WRITE ---------------- #
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_PATH = Path(OUTPUT_FOLDER) / f"contextual_chunks_{timestamp}.jsonl"

total_chunks = 0
with open(OUTPUT_PATH, "a", encoding="utf-8") as fout:
    for src, file_chunks in tqdm(by_pdf.items(), desc="Contextualizing PDFs...", unit="pdf"):
        src_path = src or "unknown"
        file_name = pathlib.Path(src_path).name if src_path else "unknown"

        # WHOLE-PDF context: join all chunks' text
        whole_doc = " ".join([c.page_content for c in file_chunks])

        # Precompute total chunks per page for this PDF
        page_chunk_counts = defaultdict(int)
        for c in file_chunks:
            pg = c.metadata.get("page")
            page_chunk_counts[pg] += 1

        for idx, d in enumerate(file_chunks):
            page = d.metadata.get("page")
            start_index = d.metadata.get("start_index") or 0
            end_index = start_index + len(d.page_content)

            # stable unique id
            h = hashlib.sha256()
            h.update(str(src_path).encode("utf-8"))
            h.update(str(page).encode("utf-8"))
            h.update(str(start_index).encode("utf-8"))
            h.update(str(d.page_content[:200]).encode("utf-8"))
            chunk_id = h.hexdigest()[:16]

            # enrich metadata with both per-page and per-file counts
            d.metadata.update(
                {
                    "file_name": file_name,
                    "chunk_index": idx,
                    "page": page,
                    "total_chunks_per_page": page_chunk_counts.get(page, 0),
                    "total_chunks_per_file": len(file_chunks),
                    "start_index": start_index,
                    "end_index": end_index,
                    "ingested_at": datetime.now().isoformat(),
                    "chunk_id": chunk_id,
                    "context_scope": "whole_pdf",
                }
            )

            # LLM call (whole PDF context)
            context_text = get_context_from_ollama(whole_doc, d.page_content, model=MODEL_NAME, metadata=d.metadata)
            contextualized_chunk = f"{context_text}\n\n{d.page_content}" if context_text else d.page_content

            record = {"id": chunk_id, "text": contextualized_chunk, "metadata": d.metadata}

            # write one line per chunk (crash-safe)
            try:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
            except UnicodeEncodeError as ue:
                safe_line = (json.dumps(record, ensure_ascii=False)).encode("utf-8", errors="replace").decode("utf-8")
                fout.write(safe_line + "\n")
                fout.flush()
                logging.warning(f"Unicode issue writing chunk {chunk_id}; wrote with replacement. {ue}")

            total_chunks += 1

logging.info(f"✅ Completed contextualization for {total_chunks} chunks.")
logging.info(f"Chunks saved incrementally to {OUTPUT_PATH}")

print(f"\n✅ Done! Saved contextualized chunks incrementally to: {OUTPUT_PATH}")
print(f"🪵 Logs:\n  - Main log: {LOG_FILE}\n  - LLM call log: {LLM_LOG_FILE}")
