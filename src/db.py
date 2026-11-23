import os
import json
import re
import psycopg
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    DATABASE_URL = "postgresql://postgres:postgres@localhost:5434/mathrouter"

def get_connection():
    conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)
    register_vector(conn)
    return conn

def insert_document_chunks(document_name: str, chunks: list[str], embeddings: list[list[float]], metadata: dict = None):
    """Stores chunks and their vector embeddings into document_embeddings table, replacing existing chunks for this document."""
    if metadata is None:
        metadata = {}
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM document_embeddings WHERE document_name = %s;", (document_name,))
            for idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
                cur.execute(
                    """
                    INSERT INTO document_embeddings 
                    (document_name, chunk_index, chunk_text, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (document_name, idx, chunk_text, embedding, json.dumps(metadata))
                )
        conn.commit()

def detect_target_documents(query_text: str, available_docs: list[str]) -> list[str]:
    """Detects target document names from query text using strict word boundary matching to avoid false positives."""
    if not query_text:
        return []
    q_lower = query_text.lower()
    matches = []

    alias_map = {
        "odyssey": "the-odyssey-homer.pdf",
        "homer": "the-odyssey-homer.pdf",
        "ulysses": "the-odyssey-homer.pdf",
        "frankenstein": "frankenstein-mary-shelley.pdf",
        "shelley": "frankenstein-mary-shelley.pdf",
        "prince": "Machiavelli, Niccolo - The Prince (EN, 1513, 239 p.).pdf",
        "machiavelli": "Machiavelli, Niccolo - The Prince (EN, 1513, 239 p.).pdf",
        "metamorphosis": "the-metamorphosis-franz-kafka.pdf",
        "kafka": "the-metamorphosis-franz-kafka.pdf",
        "rich dad": "epdf.pub_rich-dad-poor-dad.pdf",
        "kiyosaki": "epdf.pub_rich-dad-poor-dad.pdf",
        "animal farm": "animal-farm-george-orwell.pdf",
        "orwell": "animal-farm-george-orwell.pdf",
        "napoleon": "animal-farm-george-orwell.pdf",
        "snowball": "animal-farm-george-orwell.pdf",
        "pride and prejudice": "pride-and-prejudice-jane-austen.pdf",
        "austen": "pride-and-prejudice-jane-austen.pdf",
        "darcy": "pride-and-prejudice-jane-austen.pdf",
        "elizabeth bennet": "pride-and-prejudice-jane-austen.pdf",
        "psychology of money": "The_Psychology_of_Money.pdf",
        "housel": "The_Psychology_of_Money.pdf",
        "win friends": "How-to-win-friends-and-influence-people-Dale-Carnegie.pdf",
        "carnegie": "How-to-win-friends-and-influence-people-Dale-Carnegie.pdf",
        "republic": "republic-plato.pdf",
        "plato": "republic-plato.pdf",
        "socrates": "republic-plato.pdf",
        "romeo and juliet": "romeo-and-juliet-william-shakespeare.pdf",
        "shakespeare": "romeo-and-juliet-william-shakespeare.pdf",
        "capulet": "romeo-and-juliet-william-shakespeare.pdf",
        "montague": "romeo-and-juliet-william-shakespeare.pdf",
        "disaster recovery": "cloud_disaster_recovery.txt",
        "rpo": "cloud_disaster_recovery.txt",
        "rto": "cloud_disaster_recovery.txt",
        "replication lag": "cloud_disaster_recovery.txt",
        "canary": "continuous_deployment_policy.pdf",
        "continuous deployment": "continuous_deployment_policy.pdf",
        "continuous delivery": "continuous_deployment_policy.pdf",
        "password": "it_security_compliance.txt",
        "lockout": "it_security_compliance.txt",
        "compliance": "it_security_compliance.txt",
        "sla": "enterprise_sla_agreement.txt",
        "uptime": "enterprise_sla_agreement.txt",
        "deductible": "health_insurance_benefits.txt",
        "therapy sessions": "health_insurance_benefits.txt",
        "gdpr": "data_retention_gdpr.txt",
        "right to erasure": "data_retention_gdpr.txt",
        "retention": "data_retention_gdpr.txt",
        "code review": "code_review_security_policy.txt",
        "deployment freeze": "code_review_security_policy.txt",
        "procurement": "procurement_vendor_policy.txt",
        "vendor": "procurement_vendor_policy.txt",
        "competitive bids": "procurement_vendor_policy.txt",
        "incident": "incident_response_sev1.txt",
        "war room": "incident_response_sev1.txt",
        "sev1": "incident_response_sev1.txt",
        "parental leave": "employee_leave_parental.txt",
        "vacation": "employee_leave_parental.txt",
        "intellectual property": "intellectual_property_inventions.txt",
        "non-solicitation": "intellectual_property_inventions.txt",
        "non-compete": "intellectual_property_inventions.txt",
        "rate limit": "api_rate_limits_spec.pdf",
        "token bucket": "api_rate_limits_spec.pdf",
        "ergonomic": "remote_work_ergonomics.pdf",
        "monitor": "remote_work_ergonomics.pdf",
        "cryptographic": "cryptographic_key_management.pdf",
        "aes-256": "cryptographic_key_management.pdf",
        "per diem": "travel_per_diem_policy.pdf",
        "business class": "travel_per_diem_policy.pdf",
        "data classification": "customer_data_classification.pdf",
        "pii": "customer_data_classification.pdf",
        "spillages": "customer_data_classification.pdf",
        "contractor": "contractor_onboarding_terms.pdf",
        "laptop": "contractor_onboarding_terms.pdf",
        "bug bounty": "bug_bounty_program_rules.pdf",
        "whistleblower": "whistleblower_ethics_code.pdf"
    }

    for key, doc in alias_map.items():
        pattern = r'\b' + re.escape(key) + r'\b'
        if re.search(pattern, q_lower) and doc in available_docs:
            if doc not in matches:
                matches.append(doc)

    for d in available_docs:
        base = d.rsplit('.', 1)[0].replace('_', ' ').replace('-', ' ').lower()
        if len(base) > 4:
            pattern = r'\b' + re.escape(base) + r'\b'
            if re.search(pattern, q_lower) and d not in matches:
                matches.append(d)

    return matches

def search_similar_chunks(
    query_embedding: list[float],
    top_k: int = 8,
    document_name: str = None,
    query_text: str = None
):
    """
    Intelligent Hybrid Retrieval:
    - Supports user-specified or auto-detected target document(s)
    - Combines dense vector similarity with keyword/lexical relevance boost
    - Automatically incorporates synopsis/character chunks for broad document queries
    - Filters out advertisement and boilerplate noise
    - Defaults to top_k = 8 (configurable up to 20)
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            # 1. Fetch all distinct document names available in pgvector
            cur.execute("SELECT DISTINCT document_name FROM document_embeddings;")
            available_docs = [r["document_name"] for r in cur.fetchall()]

            # Resolve target document(s)
            target_docs = []
            if document_name:
                # Direct match or fuzzy match
                doc_clean = document_name.strip()
                if doc_clean in available_docs:
                    target_docs = [doc_clean]
                else:
                    # Fuzzy match (e.g. contractor_onboarding_hardware vs contractor_onboarding_terms)
                    base_req = doc_clean.rsplit('.', 1)[0].lower()
                    for d in available_docs:
                        d_base = d.rsplit('.', 1)[0].lower()
                        if base_req in d_base or d_base in base_req or any(part in d_base for part in base_req.split('_') if len(part) > 4):
                            target_docs = [d]
                            break

            # If no document was explicitly given or matched, detect from query text
            if not target_docs and query_text:
                target_docs = detect_target_documents(query_text, available_docs)

            # 2. Vector Candidate Search
            candidates = []
            fetch_limit = max(top_k * 3, 24)

            if target_docs:
                cur.execute(
                    """
                    SELECT id, document_name, chunk_index, chunk_text, metadata,
                           1 - (embedding <=> %s::vector) AS similarity_score
                    FROM document_embeddings
                    WHERE document_name = ANY(%s)
                    ORDER BY embedding <=> %s::vector ASC
                    LIMIT %s;
                    """,
                    (query_embedding, target_docs, query_embedding, fetch_limit)
                )
                candidates.extend(cur.fetchall())

                # If the query is multi-part and matches multiple documents (e.g. infra + canary),
                # ensure balanced representation from all target docs
                if len(target_docs) > 1:
                    per_doc_limit = max(top_k // len(target_docs), 3)
                    for t_doc in target_docs:
                        cur.execute(
                            """
                            SELECT id, document_name, chunk_index, chunk_text, metadata,
                                   1 - (embedding <=> %s::vector) AS similarity_score
                            FROM document_embeddings
                            WHERE document_name = %s
                            ORDER BY embedding <=> %s::vector ASC
                            LIMIT %s;
                            """,
                            (query_embedding, t_doc, query_embedding, per_doc_limit)
                        )
                        candidates.extend(cur.fetchall())
            else:
                cur.execute(
                    """
                    SELECT id, document_name, chunk_index, chunk_text, metadata,
                           1 - (embedding <=> %s::vector) AS similarity_score
                    FROM document_embeddings
                    ORDER BY embedding <=> %s::vector ASC
                    LIMIT %s;
                    """,
                    (query_embedding, query_embedding, fetch_limit)
                )
                candidates.extend(cur.fetchall())

            # 3. Overview / Synopsis chunk inclusion for large books
            is_broad_query = False
            if query_text:
                q_lower = query_text.lower()
                is_broad_query = any(w in q_lower for w in [
                    'character', 'characters', 'name', 'names', 'summary', 'overview',
                    'synopsis', 'plot', 'cast', 'who is', 'who are', 'who was', 'detail', 'book'
                ])

            if target_docs and is_broad_query:
                for t_doc in target_docs:
                    # Check if document has > 10 chunks (e.g. Odyssey, Frankenstein, The Prince, Metamorphosis)
                    cur.execute("SELECT COUNT(*) as cnt FROM document_embeddings WHERE document_name = %s;", (t_doc,))
                    cnt_res = cur.fetchone()
                    if cnt_res and cnt_res["cnt"] > 10:
                        # Find structural synopsis / chapter overview chunks
                        cur.execute(
                            """
                            SELECT chunk_index
                            FROM document_embeddings
                            WHERE document_name = %s 
                              AND (
                                chunk_text ILIKE '%%S Y N O P S I S%%'
                                OR chunk_text ILIKE '%%SYNOPSIS%%'
                                OR chunk_text ILIKE '%%H O W   T O   R E A D%%'
                                OR chunk_text ILIKE '%%HOW TO READ THIS BOOK%%'
                                OR chunk_text ILIKE '%%CHAPTER I.%%'
                                OR chunk_text ILIKE '%%CHAPTER 1%%'
                              )
                            ORDER BY chunk_index ASC
                            LIMIT 3;
                            """,
                            (t_doc,)
                        )
                        match_indices = [r["chunk_index"] for r in cur.fetchall()]
                        if match_indices:
                            # Expand indices to include adjacent context chunks (e.g. synopsis spans 1-4)
                            expanded_indices = set()
                            for idx in match_indices:
                                for offset in [0, 1, 2, 3]:
                                    expanded_indices.add(idx + offset)
                            cur.execute(
                                """
                                SELECT id, document_name, chunk_index, chunk_text, metadata, 0.60 AS similarity_score
                                FROM document_embeddings
                                WHERE document_name = %s AND chunk_index = ANY(%s)
                                ORDER BY chunk_index ASC;
                                """,
                                (t_doc, list(expanded_indices))
                            )
                            candidates.extend(cur.fetchall())

    # 4. Keyword & Hybrid Scoring
    stopwords = {
        'what', 'when', 'where', 'which', 'who', 'whom', 'whose', 'why', 'how',
        'that', 'this', 'these', 'those', 'with', 'from', 'have', 'more', 'some',
        'each', 'give', 'list', 'into', 'book', 'text', 'line', 'story', 'their',
        'will', 'would', 'should', 'could', 'about', 'under', 'after', 'before'
    }
    keywords = []
    if query_text:
        keywords = [w for w in re.findall(r'\b[a-zA-Z]{4,}\b', query_text.lower()) if w not in stopwords]

    # Deduplicate candidates by chunk ID
    unique_candidates = {}
    for c in candidates:
        cid = c["id"]
        if cid not in unique_candidates or float(c["similarity_score"]) > float(unique_candidates[cid]["similarity_score"]):
            unique_candidates[cid] = dict(c)

    scored_list = []
    for cid, c in unique_candidates.items():
        text_lower = c["chunk_text"].lower()
        sim = float(c["similarity_score"])

        # Filter out pure promotional / website download boilerplate if we have enough chunks
        if "scan the code or go to infobooks.org" in text_lower and len(unique_candidates) > 4:
            continue
        if "this ebook was designed and published by planet pdf" in text_lower and len(c["chunk_text"]) < 200:
            continue

        kw_hits = sum(1 for kw in keywords if kw in text_lower)
        kw_boost = min(kw_hits * 0.04, 0.25)
        total_score = sim + kw_boost

        scored_list.append((total_score, c))

    # 5. Strict Relevance Filtering (Fetch ONLY truly relevant chunks, zero extraneous noise)
    scored_list.sort(key=lambda x: x[0], reverse=True)
    if not scored_list:
        return []

    top_score = scored_list[0][0]
    # Keep only chunks with high relevance: above 0.48 baseline AND within 0.16 of top score
    min_cutoff = max(0.48, top_score - 0.16)

    filtered_results = []
    for score, chunk in scored_list:
        if score >= min_cutoff:
            filtered_results.append(chunk)
        if len(filtered_results) >= top_k:
            break

    # If no chunk met cutoff but candidates exist, return at least the single best match
    if not filtered_results and scored_list:
        filtered_results.append(scored_list[0][1])

    return filtered_results

def save_rag_evaluation(
    query: str,
    model_id: str,
    retrieved_contexts: list[str],
    response: str,
    faithfulness_score: float,
    context_relevance_score: float,
    answer_relevance_score: float,
    confidence_score: float,
    verdict: str,
    unsupported_claims: list[str],
    explanations: dict,
    latency_ms: int,
    document_name: str = "Knowledge Base",
    claims_analysis: list[dict] = None
) -> int:
    """Inserts a full evaluation record into the Elephant audit table."""
    if claims_analysis is None:
        claims_analysis = []
        
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO rag_evaluations
                (query, model_id, retrieved_contexts, response, faithfulness_score,
                 context_relevance_score, answer_relevance_score, confidence_score,
                 verdict, unsupported_claims, explanations, latency_ms, document_name, claims_analysis)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
                """,
                (
                    query,
                    model_id,
                    json.dumps(retrieved_contexts),
                    response,
                    faithfulness_score,
                    context_relevance_score,
                    answer_relevance_score,
                    confidence_score,
                    verdict,
                    json.dumps(unsupported_claims),
                    json.dumps(explanations),
                    latency_ms,
                    document_name,
                    json.dumps(claims_analysis)
                )
            )
            row = cur.fetchone()
            conn.commit()
            return row["id"]

def fetch_recent_evaluations(limit: int = 20, low_confidence_only: bool = False):
    """Retrieves recent evaluation logs for review or dashboard feeds."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            if low_confidence_only:
                cur.execute(
                    """
                    SELECT * FROM rag_evaluations 
                    WHERE confidence_score < 0.85 OR verdict = 'HALLUCINATION_DETECTED'
                    ORDER BY created_at DESC LIMIT %s
                    """,
                    (limit,)
                )
            else:
                cur.execute(
                    "SELECT * FROM rag_evaluations ORDER BY created_at DESC LIMIT %s",
                    (limit,)
                )
            return cur.fetchall()
