import os
import json
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Response
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from psycopg import sql

from src.server import ask_rag, ingest_document
from src.db import fetch_recent_evaluations, get_connection
from src.metrics import start_metrics_server, record_evaluation_metrics
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

# FastAPI Lifespan context manager replacing deprecated @app.on_event("startup")
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: synchronize historical metrics
    try:
        sync_metrics_from_db()
    except Exception as e:
        print(f"[Lifespan Startup] Notice: {e}")
    yield
    # Shutdown logic if needed

app = FastAPI(title="RAG Evaluation & Viewport API", version="2.0.0", lifespan=lifespan)

# Security Fix: Support explicit allowed origins from environment variable while keeping local dev functional
default_origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001"
]
env_origins = os.getenv("ALLOWED_ORIGINS")
origins = [o.strip() for o in env_origins.split(",") if o.strip()] if env_origins else default_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins else ["*"],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Merge and mount full multi-project API endpoints from hello-mcp (MathRouter & Doc Intelligence)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hello-mcp"))
try:
    import app as hello_app
    existing_paths = {r.path for r in app.router.routes if hasattr(r, 'path')}
    for r in hello_app.app.router.routes:
        if getattr(r, 'name', '') != 'public' and hasattr(r, 'path'):
            # Replace placeholder recordings/evals routes with authoritative implementations from hello_app
            if r.path not in existing_paths or r.path in ['/api/recordings', '/api/evals/compare']:
                app.router.routes = [ex for ex in app.router.routes if getattr(ex, 'path', '') != r.path]
                app.router.routes.append(r)
    print(f"[API Mount] Successfully unified {len(app.router.routes)} endpoints across MathRouter, Doc Intelligence & RAG.")
except Exception as e:
    print(f"[API Mount Warning] Could not attach hello-mcp endpoints: {e}")


def sync_metrics_from_db():
    """Populates Prometheus registry with all existing historical evaluation logs from PostgreSQL."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT model_id, faithfulness_score, context_relevance_score,
                           answer_relevance_score, confidence_score, verdict, latency_ms
                    FROM rag_evaluations
                    ORDER BY id ASC;
                """)
                rows = cur.fetchall()
                for r in rows:
                    record_evaluation_metrics(
                        model_id=r["model_id"] or "gpt-6-astra",
                        faithfulness=float(r["faithfulness_score"] or 0.0),
                        context_relevance=float(r["context_relevance_score"] or 0.0),
                        answer_relevance=float(r["answer_relevance_score"] or 0.0),
                        confidence=float(r["confidence_score"] or 0.0),
                        verdict=r["verdict"] or "HIGH_CONFIDENCE",
                        latency_ms=int(r["latency_ms"] or 500)
                    )
                print(f"[Prometheus] Synchronized {len(rows)} evaluations from PostgreSQL into Prometheus.")
    except Exception as e:
        print(f"[Prometheus] Metrics sync warning: {e}")

@app.get("/metrics")
@app.get("/metrics/")
def get_metrics_endpoint():
    """Direct Prometheus metrics endpoint without any 307 redirects."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/api/metrics/sync")
def trigger_metrics_sync():
    """Triggers an on-demand sync of all DB evaluation stats into Prometheus."""
    sync_metrics_from_db()
    return {"status": "ok", "message": "Prometheus registry synchronized with PostgreSQL"}

class QueryRequest(BaseModel):
    query: str
    model_id: str = "gpt-6-astra"
    dataset_name: str = "default_kb"
    top_k: int = 8
    async_eval: bool = False
    document_name: Optional[str] = None

class IngestRequest(BaseModel):
    file_path: Optional[str] = None
    filename: Optional[str] = None
    content: Optional[str] = None
    dataset_name: str = "default_kb"

@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "rag-evaluation-v2"}

@app.get("/api/evaluations")
def get_evaluations(limit: int = 50, low_confidence_only: bool = False):
    """Returns recent evaluation records with document name, atomic claims, and scores."""
    rows = fetch_recent_evaluations(limit=limit, low_confidence_only=low_confidence_only)
    formatted = []
    for r in rows:
        row_dict = dict(r)
        if "created_at" in row_dict and row_dict["created_at"]:
            row_dict["created_at"] = row_dict["created_at"].isoformat()
        # Parse JSON fields safely if they are strings
        for json_col in ["retrieved_contexts", "unsupported_claims", "explanations", "claims_analysis"]:
            if json_col in row_dict and isinstance(row_dict[json_col], str):
                try:
                    row_dict[json_col] = json.loads(row_dict[json_col])
                except Exception:
                    pass
        formatted.append(row_dict)
    return formatted

@app.get("/api/evaluations/{eval_id}")
def get_evaluation_detail(eval_id: int):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM rag_evaluations WHERE id = %s", (eval_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Evaluation not found")
            res = dict(row)
            if res.get("created_at"):
                res["created_at"] = res["created_at"].isoformat()
            for col in ["retrieved_contexts", "unsupported_claims", "explanations", "claims_analysis"]:
                if col in res and isinstance(res[col], str):
                    try:
                        res[col] = json.loads(res[col])
                    except Exception:
                        pass
            return res

@app.get("/api/questions")
def get_unique_questions():
    """Returns unique questions from the evaluation logs along with benchmark questions."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT query, MAX(id) as last_id, MAX(model_id) as last_model, COUNT(*) as run_count
                FROM rag_evaluations
                WHERE query IS NOT NULL AND query != ''
                GROUP BY query
                ORDER BY last_id DESC;
            """)
            rows = cur.fetchall()
            return [dict(r) for r in rows]

@app.delete("/api/evaluations/duplicates")
def delete_duplicate_evaluations():
    """Removes duplicate evaluations keeping the latest record per unique question."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                DELETE FROM rag_evaluations 
                WHERE id NOT IN (
                    SELECT MAX(id) FROM rag_evaluations GROUP BY query
                );
            """)
            deleted_count = cur.rowcount
        conn.commit()
    return {"status": "ok", "deleted_count": deleted_count}

RIGID_DOCS_DIR = os.path.realpath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rigid_docs"))
if not os.path.exists(RIGID_DOCS_DIR):
    os.makedirs(RIGID_DOCS_DIR, exist_ok=True)

def safe_rigid_doc_path(filename: str) -> str:
    """Validates and returns an absolute safe path within rigid_docs, preventing path traversal."""
    cleaned = os.path.basename(filename.strip().replace("\\", "/"))
    if not cleaned or cleaned in [".", ".."]:
        raise HTTPException(status_code=400, detail="Invalid filename")
    full_path = os.path.realpath(os.path.join(RIGID_DOCS_DIR, cleaned))
    if not full_path.startswith(RIGID_DOCS_DIR + os.sep) and full_path != RIGID_DOCS_DIR:
        raise HTTPException(status_code=400, detail="Directory traversal attempt detected")
    return full_path

@app.post("/api/query")
def execute_rag_query(req: QueryRequest):
    """Executes a RAG query and returns response, confidence score, and atomic claims."""
    try:
        result = ask_rag(
            query=req.query,
            dataset_name=req.dataset_name,
            model_id=req.model_id,
            top_k=req.top_k,
            async_eval=req.async_eval,
            document_name=req.document_name
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ingest")
def execute_ingest(req: IngestRequest):
    """Ingests a document into the RAG engine via file_path or raw text content."""
    target_path = req.file_path
    if not target_path and req.content and req.filename:
        target_path = safe_rigid_doc_path(req.filename)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(req.content)
    elif target_path:
        # Validate target_path if it's within rigid_docs or relative
        if not os.path.isabs(target_path):
            target_path = safe_rigid_doc_path(target_path)
    
    if not target_path or not os.path.exists(target_path):
        raise HTTPException(status_code=400, detail=f"Invalid file or content. Target: {target_path}")

    res = ingest_document(file_path=target_path, dataset_name=req.dataset_name)
    return res

MAX_UPLOAD_SIZE_BYTES = 50 * 1024 * 1024 # 50 MB

@app.post("/api/upload")
async def upload_and_ingest(file: UploadFile = File(...), dataset_name: str = "default_kb"):
    """Accepts any PDF, TXT, DOCX, or other document file, saves it, and ingests into pgvector."""
    target_path = safe_rigid_doc_path(file.filename or "uploaded_doc")
    content = await file.read(MAX_UPLOAD_SIZE_BYTES + 1)
    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File size exceeds maximum allowed limit (50MB)")
    with open(target_path, "wb") as f:
        f.write(content)
    
    res = ingest_document(file_path=target_path, dataset_name=dataset_name)
    return res

@app.get("/api/documents/available")
def get_available_documents():
    """Lists all available documents on disk in rigid_docs/."""
    os.makedirs("rigid_docs", exist_ok=True)
    files = []
    for f in sorted(os.listdir("rigid_docs")):
        fp = os.path.join("rigid_docs", f)
        if os.path.isfile(fp):
            files.append({
                "filename": f,
                "size_bytes": os.path.getsize(fp),
                "extension": os.path.splitext(f)[1].lower()
            })
    return files

@app.get("/api/documents/{doc_name}/content")
def get_document_content(doc_name: str):
    """Returns chunks and file text for a given document."""
    # 1. Fetch chunks from pgvector
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, chunk_index, chunk_text, metadata, created_at
                FROM document_embeddings
                WHERE document_name = %s
                ORDER BY chunk_index ASC;
            """, (doc_name,))
            chunk_rows = cur.fetchall()
            
    chunks = []
    for r in chunk_rows:
        rd = dict(r)
        if rd.get("created_at"):
            rd["created_at"] = rd["created_at"].isoformat()
        chunks.append(rd)
        
    # 2. Check if file exists on disk
    file_path = safe_rigid_doc_path(doc_name)
    has_file = os.path.exists(file_path)
    file_size = os.path.getsize(file_path) if has_file else 0
    ext = os.path.splitext(doc_name)[1].lower()
    
    full_text = ""
    if chunks:
        full_text = "\n\n".join([c["chunk_text"] for c in chunks])
    elif has_file and ext in [".txt", ".md", ".json", ".csv"]:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                full_text = f.read()
        except Exception:
            pass

    return {
        "document_name": doc_name,
        "extension": ext,
        "has_file": has_file,
        "file_size": file_size,
        "is_pdf": ext == ".pdf",
        "is_docx": ext == ".docx",
        "chunk_count": len(chunks),
        "chunks": chunks,
        "full_text": full_text
    }

@app.get("/api/documents/{doc_name}/file")
def get_document_file(doc_name: str):
    """Serves raw document file for direct inline in-browser PDF or doc view without download prompts."""
    file_path = safe_rigid_doc_path(doc_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on server")
    
    ext = os.path.splitext(doc_name)[1].lower()
    media_type = "application/octet-stream"
    if ext == ".pdf":
        media_type = "application/pdf"
    elif ext in [".txt", ".md", ".json", ".csv"]:
        media_type = "text/plain; charset=utf-8"
    elif ext == ".docx":
        media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        
    return FileResponse(
        file_path,
        media_type=media_type,
        filename=doc_name,
        content_disposition_type="inline",
        headers={
            "Content-Disposition": f'inline; filename="{doc_name}"',
            "X-Content-Type-Options": "nosniff"
        }
    )

@app.get("/api/documents")
def get_documents_list():
    """Returns distinct documents from document_embeddings with chunk count."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT document_name, COUNT(*) as chunk_count, MIN(created_at) as created_at
                FROM document_embeddings
                GROUP BY document_name
                ORDER BY document_name ASC;
            """)
            rows = cur.fetchall()
            formatted = []
            for r in rows:
                rd = dict(r)
                if rd.get("created_at"):
                    rd["created_at"] = rd["created_at"].isoformat()
                formatted.append(rd)
            return formatted

@app.get("/api/db/overview")
def get_db_overview():
    """Returns PostgreSQL tables with live row counts and schema overview."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """)
            tables = [r["table_name"] for r in cur.fetchall()]
            stats = []
            for t in tables:
                # Security Fix: Prevent SQL injection using parameterized sql.Identifier
                cur.execute(sql.SQL("SELECT count(*) as cnt FROM {}").format(sql.Identifier(t)))
                cnt = cur.fetchone()["cnt"]
                cur.execute("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = %s
                    ORDER BY ordinal_position;
                """, (t,))
                cols = [{"name": c["column_name"], "type": c["data_type"]} for c in cur.fetchall()]
                stats.append({"table": t, "row_count": cnt, "columns": cols})
            return {"tables": stats}

@app.get("/api/db/table/{table_name}")
def get_db_table_data(table_name: str, limit: int = 50, offset: int = 0):
    """Returns real live rows from any PostgreSQL table."""
    limit = max(1, min(limit, 500)) # Bound pagination
    offset = max(0, offset)
    
    allowed_tables = ["document_embeddings", "rag_evaluations", "eval_sessions", "eval_turns"]
    if table_name not in allowed_tables:
        raise HTTPException(status_code=400, detail="Invalid table name")
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            if table_name == "document_embeddings":
                cur.execute("""
                    SELECT id, document_name, chunk_index, chunk_text, 
                           metadata, created_at,
                           'vector(1536)' as embedding_dim
                    FROM document_embeddings
                    ORDER BY id DESC
                    LIMIT %s OFFSET %s;
                """, (limit, offset))
            else:
                cur.execute(
                    sql.SQL("SELECT * FROM {} ORDER BY id DESC LIMIT %s OFFSET %s").format(sql.Identifier(table_name)),
                    (limit, offset)
                )
            rows = cur.fetchall()
            cur.execute(sql.SQL("SELECT count(*) as cnt FROM {}").format(sql.Identifier(table_name)))
            total = cur.fetchone()["cnt"]
            
            formatted = []
            for r in rows:
                rd = dict(r)
                if "created_at" in rd and rd["created_at"]:
                    rd["created_at"] = rd["created_at"].isoformat()
                for jc in ["retrieved_contexts", "unsupported_claims", "explanations", "claims_analysis", "metadata"]:
                    if jc in rd and isinstance(rd[jc], str):
                        try:
                            rd[jc] = json.loads(rd[jc])
                        except Exception:
                            pass
                formatted.append(rd)
            return {"table": table_name, "total": total, "limit": limit, "offset": offset, "rows": formatted}

@app.get("/api/recordings")
def get_recordings():
    """Returns all recorded MathRouter evaluation sessions from PostgreSQL and disk files with paired baselines."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hello-mcp"))
    
    POSTGRES_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5434/mathrouter")
    recordings_map = {}

    try:
        import psycopg
        with psycopg.connect(POSTGRES_URL, connect_timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT session_id, recording_title, recording_number, model,
                           start_time, end_time, total_turns, total_tokens,
                           total_latency_seconds, created_at, payload
                    FROM eval_sessions ORDER BY created_at ASC, session_id ASC;
                """)
                for r in cur.fetchall():
                    s_id = r[0]
                    rec_title = r[1] or f"RECORDING {len(recordings_map) + 1}"
                    recordings_map[s_id] = {
                        "session_id": s_id,
                        "recording_title": rec_title,
                        "recording_number": r[2] or (len(recordings_map) + 1),
                        "model": r[3],
                        "start_time": r[4].isoformat() if r[4] else "",
                        "end_time": r[5].isoformat() if r[5] else "",
                        "total_takes": r[6] or 0,
                        "total_tokens": r[7] or 0,
                        "total_latency_seconds": float(r[8] or 0),
                        "created_at": r[9].isoformat() if r[9] else "",
                        "source_session_id": (r[10].get("source_session_id") if (len(r) > 10 and isinstance(r[10], dict)) else None),
                        "takes": []
                    }
                cur.execute("""
                    SELECT turn_id, session_id, turn_number, take_label, recording_title,
                           model, timestamp, user_message, assistant_response,
                           latency_seconds, total_tokens, payload, prompt_tokens, completion_tokens, tool_calls
                    FROM eval_turns ORDER BY created_at ASC, turn_number ASC;
                """)
                for r in cur.fetchall():
                    t_id, s_id, t_num, t_label, rec_title, model, ts, u_msg, a_res, lat, toks, payload, p_tok, c_tok, t_calls = r
                    if s_id in recordings_map:
                        recordings_map[s_id]["takes"].append({
                            "turn_id": t_id,
                            "session_id": s_id,
                            "take_number": t_num,
                            "take_label": t_label or f"TAKE #{t_num}",
                            "recording_title": recordings_map[s_id]["recording_title"],
                            "filename": f"{s_id}/take_{t_num}.json",
                            "flat_filename": f"{t_id}.json",
                            "timestamp": ts.isoformat() if ts else "",
                            "user_message": u_msg or "",
                            "preview": (u_msg or "")[:65],
                            "assistant_response": a_res or "",
                            "latency_seconds": float(lat or 0),
                            "total_tokens": toks or 0,
                            "tool_calls": t_calls or [],
                            "model": model
                        })
    except Exception as e:
        print(f"Error querying PostgreSQL recordings: {e}")

    # Fallback/augment from disk
    recordings_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hello-mcp", "recordings")
    if os.path.exists(recordings_dir):
        for f in sorted(os.listdir(recordings_dir)):
            fp = os.path.join(recordings_dir, f)
            if os.path.isfile(fp) and f.endswith(".json"):
                try:
                    with open(fp, "r", encoding="utf-8") as jf:
                        data = json.load(jf)
                        sid = data.get("session_id") or f.replace(".json", "")
                        t_num = data.get("take_number") or data.get("turn_number") or 1
                        t_id = data.get("turn_id") or f.replace(".json", "")
                        if sid not in recordings_map:
                            rec_num = len(recordings_map) + 1
                            recordings_map[sid] = {
                                "session_id": sid,
                                "recording_title": data.get("recording_title") or f"RECORDING {rec_num}",
                                "recording_number": rec_num,
                                "model": data.get("model") or "gpt-4o",
                                "start_time": data.get("timestamp") or "",
                                "total_takes": 0,
                                "total_tokens": 0,
                                "total_latency_seconds": 0.0,
                                "takes": []
                            }
                        if not any(t["turn_id"] == t_id for t in recordings_map[sid]["takes"]):
                            u_msg = data.get("user_message") or ""
                            lat = float(data.get("latency_seconds") or 0)
                            toks = int(data.get("total_tokens") or 0)
                            recordings_map[sid]["takes"].append({
                                "turn_id": t_id,
                                "session_id": sid,
                                "take_number": t_num,
                                "take_label": data.get("take_label") or f"TAKE #{t_num}",
                                "recording_title": recordings_map[sid]["recording_title"],
                                "filename": f,
                                "flat_filename": f,
                                "timestamp": data.get("timestamp") or "",
                                "user_message": u_msg,
                                "preview": u_msg[:65],
                                "assistant_response": data.get("assistant_response") or "",
                                "latency_seconds": lat,
                                "total_tokens": toks,
                                "tool_calls": data.get("tool_calls") or [],
                                "model": data.get("model", "gpt-4o")
                            })
                            recordings_map[sid]["total_takes"] = len(recordings_map[sid]["takes"])
                            recordings_map[sid]["total_tokens"] += toks
                            recordings_map[sid]["total_latency_seconds"] += lat
                except Exception as e:
                    pass

    # Separate primary sessions and baselines
    primary = []
    baselines = {}
    for sid, s in recordings_map.items():
        s["takes"].sort(key=lambda x: int(x.get("take_number", 0)))
        is_baseline = (
            s["model"] in ["gpt-6-sol", "gpt-5.5", "gemini-2.5-flash"]
            or sid.startswith("modelb_")
            or sid.startswith("gpt5_")
            or sid.startswith("gemini_")
        )
        if is_baseline:
            src = s.get("source_session_id")
            if src: baselines[src] = s
            else: baselines[sid] = s
        else:
            primary.append(s)

    primary.sort(key=lambda x: (x.get("start_time") or x.get("created_at") or x.get("session_id")))

    for idx, p in enumerate(primary, 1):
        p["recording_number"] = idx
        p["recording_title"] = f"RECORDING {idx}"
        for t in p["takes"]:
            t["recording_title"] = p["recording_title"]
            t["recording_number"] = idx
        
        # Pair with baseline
        matched = baselines.get(p["session_id"])
        if not matched and baselines:
            matched = list(baselines.values())[(idx - 1) % len(baselines)]
        
        if matched:
            p["gpt5_run"] = matched
            p["gemini_run"] = matched
            p["available_models"] = ["gpt-4o", matched.get("model", "gpt-6-sol")]
        else:
            p["gpt5_run"] = None
            p["gemini_run"] = None
            p["available_models"] = ["gpt-4o"]

    total_spend = round(sum(p["total_tokens"] * 0.000015 for p in primary), 4)
    return {"recordings": primary, "total_spend_usd": total_spend}

@app.get("/api/evals/compare")
def get_evals_comparison(session_id: Optional[str] = None, judge: bool = False):
    """Returns side-by-side comparison for any session with baseline turns and LLM Judge metrics."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hello-mcp"))
    try:
        from run_evals import evaluate_sessions, find_paired_sessions
        target_gpt, target_base = find_paired_sessions()
        gpt_id = session_id or (target_gpt["session_id"] if target_gpt else None)
        base_id = target_base["session_id"] if target_base else None

        if not gpt_id or not base_id:
            raise HTTPException(status_code=404, detail="No paired session available")

        summary = evaluate_sessions(gpt_id, base_id, use_judge=judge)
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation comparison error: {str(e)}")

@app.get("/api/recordings/{filename:path}")
def get_recording_file(filename: str):
    """Returns raw JSON file for a recording or turn."""
    recordings_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hello-mcp", "recordings")
    
    # Check directly
    cand = os.path.join(recordings_dir, filename)
    if os.path.exists(cand) and os.path.isfile(cand):
        with open(cand, "r", encoding="utf-8") as f:
            return json.load(f)
    
    # Check with .json suffix
    if not filename.endswith(".json"):
        cand_json = cand + ".json"
        if os.path.exists(cand_json) and os.path.isfile(cand_json):
            with open(cand_json, "r", encoding="utf-8") as f:
                return json.load(f)

    # Check from PostgreSQL
    POSTGRES_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5434/mathrouter")
    try:
        import psycopg
        with psycopg.connect(POSTGRES_URL, connect_timeout=2) as conn:
            with conn.cursor() as cur:
                turn_id = filename.replace(".json", "")
                cur.execute("SELECT payload FROM eval_turns WHERE turn_id = %s OR flat_filename = %s LIMIT 1;", (turn_id, filename))
                row = cur.fetchone()
                if row and row[0]:
                    return row[0] if isinstance(row[0], dict) else json.loads(row[0])
    except Exception:
        pass

    raise HTTPException(status_code=404, detail=f"Recording {filename} not found")

@app.get("/api/recordings/{session_id}")
def get_recording_detail(session_id: str):
    """Returns specific recorded session with all takes."""
    all_recs = get_recordings()
    for r in all_recs:
        if r["session_id"] == session_id:
            return r
    raise HTTPException(status_code=404, detail="Recording session not found")

@app.get("/api/eval_backup")
def get_eval_backup():
    """Returns historical backup evaluation runs including simulated failure runs."""
    backup_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hello-mcp", "eval_backup_take2_take4.json")
    if os.path.exists(backup_file):
        try:
            with open(backup_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read backup: {e}")
    raise HTTPException(status_code=404, detail="Backup file not found")

@app.get("/api/experiments/master-summary")
def get_master_experiments_summary():
    """
    Returns an aggregated, consolidated master summary of all experiments ever run:
    1. 100 Questions RAG Benchmark
    2. 100 Questions Philosophical RAG Benchmark
    3. MathRouter Tool Calling Evaluation Sessions & Takes
    4. Executive PDF Reports index
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 1. Load RAG 100 Benchmark
    rag_100 = []
    rag_100_path = os.path.join(base_dir, "benchmark_100_results.json")
    if os.path.exists(rag_100_path):
        try:
            with open(rag_100_path, "r", encoding="utf-8") as f:
                rag_100 = json.load(f)
        except Exception as e:
            print("Error loading benchmark_100_results.json:", e)

    # 2. Load Philosophical 100 Benchmark
    phil_100 = []
    phil_100_path = os.path.join(base_dir, "philosophical_100_results.json")
    if os.path.exists(phil_100_path):
        try:
            with open(phil_100_path, "r", encoding="utf-8") as f:
                phil_100 = json.load(f)
        except Exception as e:
            print("Error loading philosophical_100_results.json:", e)

    # 3. Load MathRouter recordings
    recs = get_recordings()

    # Calculate RAG 100 Stats
    rag_100_sys_a_wins = sum(1 for item in rag_100 if item.get("winner") == "System A")
    rag_100_sys_b_wins = sum(1 for item in rag_100 if item.get("winner") == "System B")
    rag_100_ties = sum(1 for item in rag_100 if item.get("winner") in ["Tie", "TIE", "None"])
    rag_100_avg_lat_a = (sum(item.get("system_a_latency_ms", 0) for item in rag_100) / len(rag_100)) if rag_100 else 0
    rag_100_avg_lat_b = (sum(item.get("system_b_latency_ms", 0) for item in rag_100) / len(rag_100)) if rag_100 else 0
    rag_100_avg_score_a = (sum(item.get("system_a_score", 0) for item in rag_100) / len(rag_100)) if rag_100 else 0
    rag_100_avg_score_b = (sum(item.get("system_b_score", 0) for item in rag_100) / len(rag_100)) if rag_100 else 0

    # Calculate Philosophical 100 Stats
    phil_100_sys_a_wins = sum(1 for item in phil_100 if item.get("winner") == "System A")
    phil_100_sys_b_wins = sum(1 for item in phil_100 if item.get("winner") == "System B")
    phil_100_ties = sum(1 for item in phil_100 if item.get("winner") in ["Tie", "TIE", "None"])
    phil_100_avg_lat_a = (sum(item.get("system_a_latency_ms", 0) for item in phil_100) / len(phil_100)) if phil_100 else 0
    phil_100_avg_lat_b = (sum(item.get("system_b_latency_ms", 0) for item in phil_100) / len(phil_100)) if phil_100 else 0
    phil_100_avg_score_a = (sum(item.get("system_a_score", 0) for item in phil_100) / len(phil_100)) if phil_100 else 0
    phil_100_avg_score_b = (sum(item.get("system_b_score", 0) for item in phil_100) / len(phil_100)) if phil_100 else 0

    # Calculate MathRouter Stats
    math_total_tokens = sum(r["total_tokens"] for r in recs)
    math_total_takes = sum(r["total_takes"] for r in recs)
    math_avg_latency = (sum(r["total_latency_seconds"] for r in recs) / len(recs)) if recs else 0

    # PDF Reports Index
    reports = []
    for pdf_name, title, desc in [
        ("rag_benchmark_comparison_100.pdf", "Enterprise RAG Tri-Engine 100-Question Benchmark Report", "Comparative evaluation of Local pgvector vs Cloud RAG vs RAGFlow across 100 queries"),
        ("philosophical_rag_comparison_100.pdf", "Philosophical Synthesis 100-Question Benchmark Report", "Deep reasoning, ethics, and multi-book philosophical benchmark evaluation")
    ]:
        p = os.path.join(base_dir, pdf_name)
        if os.path.exists(p):
            reports.append({
                "filename": pdf_name,
                "title": title,
                "description": desc,
                "size_bytes": os.path.getsize(p),
                "url": f"/api/reports/pdf/{pdf_name}"
            })

    return {
        "summary": {
            "total_benchmarks_count": 2,
            "total_benchmark_queries": len(rag_100) + len(phil_100),
            "mathrouter_sessions_count": len(recs),
            "mathrouter_total_takes": math_total_takes,
            "mathrouter_total_tokens": math_total_tokens,
            "available_pdf_reports": len(reports)
        },
        "benchmarks": {
            "rag_100": {
                "name": "Enterprise Tri-Engine RAG Benchmark (100 Questions)",
                "total_queries": len(rag_100),
                "system_a": "Local pgvector RAG",
                "system_b": "Cloud RAG / RAGFlow",
                "judge_model": "gpt-6-astra (LLM Judge)",
                "system_a_wins": rag_100_sys_a_wins,
                "system_b_wins": rag_100_sys_b_wins,
                "ties": rag_100_ties,
                "system_a_win_rate_pct": round((rag_100_sys_a_wins / len(rag_100) * 100) if rag_100 else 0, 1),
                "system_b_win_rate_pct": round((rag_100_sys_b_wins / len(rag_100) * 100) if rag_100 else 0, 1),
                "system_a_avg_latency_ms": round(rag_100_avg_lat_a, 1),
                "system_b_avg_latency_ms": round(rag_100_avg_lat_b, 1),
                "system_a_avg_score": round(rag_100_avg_score_a, 2),
                "system_b_avg_score": round(rag_100_avg_score_b, 2),
                "items": rag_100
            },
            "philosophical_100": {
                "name": "Philosophical Synthesis Benchmark (100 Deep Questions)",
                "total_queries": len(phil_100),
                "system_a": "Local pgvector RAG (Deep Synthesis)",
                "system_b": "Cloud RAG / Baseline",
                "judge_model": "gpt-6-astra (LLM Judge)",
                "system_a_wins": phil_100_sys_a_wins,
                "system_b_wins": phil_100_sys_b_wins,
                "ties": phil_100_ties,
                "system_a_win_rate_pct": round((phil_100_sys_a_wins / len(phil_100) * 100) if phil_100 else 0, 1),
                "system_b_win_rate_pct": round((phil_100_sys_b_wins / len(phil_100) * 100) if phil_100 else 0, 1),
                "system_a_avg_latency_ms": round(phil_100_avg_lat_a, 1),
                "system_b_avg_latency_ms": round(phil_100_avg_lat_b, 1),
                "system_a_avg_score": round(phil_100_avg_score_a, 2),
                "system_b_avg_score": round(phil_100_avg_score_b, 2),
                "items": phil_100
            }
        },
        "mathrouter": {
            "sessions": recs,
            "total_tokens": math_total_tokens,
            "total_takes": math_total_takes,
            "avg_latency_seconds": round(math_avg_latency, 2)
        },
        "reports": reports
    }

@app.get("/api/reports/pdf/{filename}")
def serve_pdf_report(filename: str):
    """Serves benchmark PDF reports inline with proper content-type."""
    allowed = ["rag_benchmark_comparison_100.pdf", "philosophical_rag_comparison_100.pdf"]
    if filename not in allowed:
        raise HTTPException(status_code=400, detail="Invalid report filename")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="PDF report not found on server")
    
    return FileResponse(
        file_path,
        media_type="application/pdf",
        filename=filename,
        content_disposition_type="inline",
        headers={
            "Content-Disposition": f'inline; filename="{filename}"',
            "X-Content-Type-Options": "nosniff"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=False)
