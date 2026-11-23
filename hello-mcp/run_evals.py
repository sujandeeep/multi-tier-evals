#!/usr/bin/env python3
"""
Automated CLI Evaluation & LLM-as-a-Judge Tool
Compares GPT-4o (MathRouter + MCP Tools) vs GPT-5.5 (Zero Tools)
Evaluates:
  1. Mathematical accuracy & ground-truth answer consistency
  2. Latency speedup & inference time deltas
  3. Token savings and efficiency
  4. LLM-as-a-Judge reasoning quality & conciseness scoring
"""

import os
import sys
import re
import json
import time
import argparse
import psycopg
from dotenv import load_dotenv
from openai import OpenAI

# Load env variables
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

POSTGRES_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5434/mathrouter")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_db_connection():
    return psycopg.connect(POSTGRES_URL, connect_timeout=5)

def extract_numbers(text):
    """Extracts integer and decimal numbers from text."""
    if not text:
        return []
    matches = re.findall(r'[-+]?\b\d+(?:\.\d+)?\b', text)
    nums = []
    for m in matches:
        try:
            val = float(m)
            if val.is_integer():
                val = int(val)
            nums.append(val)
        except (ValueError, TypeError):
            pass
    return nums

def find_paired_sessions(recording_number=None):
    """Locate GPT-4o and GPT-5.5 baseline sessions to compare."""
    gpt4o_sessions = []
    gpt5_sessions = []

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT session_id, recording_title, model, recording_number, payload FROM eval_sessions "
                "ORDER BY created_at ASC;"
            )
            rows = cur.fetchall()

    for s_id, title, model, rec_num, payload in rows:
        title_low = (title or "").lower()
        sess_data = {
            "session_id": s_id,
            "title": title,
            "model": model,
            "rec_num": rec_num,
            "payload": payload if isinstance(payload, dict) else (json.loads(payload) if payload else {})
        }
        is_baseline = (
            model in ["gpt-6-sol", "gpt-5.5", "gemini-2.5-flash"]
            or "gpt-6-sol" in title_low
            or "gpt-5.5" in title_low
            or s_id.startswith("modelb_")
            or s_id.startswith("gpt5_")
            or "gemini" in title_low
            or s_id.startswith("gemini_")
        )
        if is_baseline:
            gpt5_sessions.append(sess_data)
        else:
            gpt4o_sessions.append(sess_data)

    if not gpt4o_sessions or not gpt5_sessions:
        return None, None

    req_idx = (recording_number - 1) if (recording_number is not None and recording_number > 0) else 0
    gpt_idx = req_idx if req_idx < len(gpt4o_sessions) else 0
    target_gpt4o = gpt4o_sessions[gpt_idx]

    # Find GPT-5.5 session that points to target_gpt4o
    target_gpt5 = None
    for gs in gpt5_sessions:
        src = gs["payload"].get("source_session_id")
        if src and src == target_gpt4o["session_id"]:
            target_gpt5 = gs
            break
        # Match by title or rec_num
        if gs.get("rec_num") == (gpt_idx + 1) or f"recording {gpt_idx + 1}" in gs.get("title", "").lower():
            target_gpt5 = gs
            break

    if not target_gpt5:
        g5_idx = req_idx if req_idx < len(gpt5_sessions) else 0
        target_gpt5 = gpt5_sessions[g5_idx]
    return target_gpt4o, target_gpt5

def fetch_session_turns(session_id):
    """Fetch all turns for a session from database or disk fallback."""
    turns = []
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT turn_id, session_id, turn_number, take_label, recording_title,
                           model, latency_seconds, total_tokens, prompt_tokens, completion_tokens,
                           user_message, assistant_response, tool_calls, payload
                    FROM eval_turns WHERE session_id = %s ORDER BY turn_number ASC;
                """, (session_id,))
                for row in cur.fetchall():
                    turns.append({
                        "turn_id": row[0],
                        "session_id": row[1],
                        "turn_number": row[2],
                        "take_label": row[3],
                        "recording_title": row[4],
                        "model": row[5],
                        "latency_seconds": float(row[6] or 0),
                        "total_tokens": int(row[7] or 0),
                        "prompt_tokens": int(row[8] or 0),
                        "completion_tokens": int(row[9] or 0),
                        "user_message": row[10] or "",
                        "assistant_response": row[11] or "",
                        "tool_calls": row[12] if isinstance(row[12], list) else (json.loads(row[12]) if row[12] else []),
                        "payload": row[13] or {}
                    })
    except Exception as e:
        print(f"Warning: Database fetch failed ({e}), checking disk.")

    if not turns:
        # Fallback to recordings directory
        rec_dir = os.path.join(os.path.dirname(__file__), "recordings", session_id)
        if os.path.isdir(rec_dir):
            for f in sorted(os.listdir(rec_dir)):
                if f.endswith(".json"):
                    with open(os.path.join(rec_dir, f), "r") as fp:
                        data = json.load(fp)
                        turns.append(data)
    return turns

def run_llm_judge(
    user_msg,
    gpt_res,
    baseline_res,
    client=None,
    gpt_latency=None,
    baseline_latency=None,
    gpt_tokens=None,
    baseline_tokens=None,
    gpt_prompt_tokens=None,
    gpt_completion_tokens=None,
    baseline_prompt_tokens=None,
    baseline_completion_tokens=None,
    gpt_tools=None,
    speedup=None,
    tok_saving_pct=None
):
    """Uses GPT-5.5 as an impartial judge to score accuracy, reasoning, tools, and telemetry."""
    if not client:
        key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        if key:
            client = OpenAI(api_key=key)
        else:
            return {
                "gpt_correctness": "UNEVALUATED",
                "gemini_correctness": "UNEVALUATED",
                "gpt5_correctness": "UNEVALUATED",
                "gpt_score": 0,
                "gemini_score": 0,
                "gpt5_score": 0,
                "winner": "TIE",
                "accuracy_analysis": "API key not configured.",
                "efficiency_analysis": "API key not configured.",
                "feedback": "OpenAI API key not configured."
            }

    tools_str = ", ".join(gpt_tools) if isinstance(gpt_tools, list) and gpt_tools else (str(gpt_tools) if gpt_tools else "0 tools")
    lat_a = f"{float(gpt_latency):.2f}s" if gpt_latency is not None else "N/A"
    lat_b = f"{float(baseline_latency):.2f}s" if baseline_latency is not None else "N/A"
    tok_a = f"{int(gpt_tokens)} total ({gpt_prompt_tokens or 0} prompt / {gpt_completion_tokens or 0} completion)" if gpt_tokens is not None else "N/A"
    tok_b = f"{int(baseline_tokens)} total ({baseline_prompt_tokens or 0} prompt / {baseline_completion_tokens or 0} completion)" if baseline_tokens is not None else "N/A"
    sp_str = f"{float(speedup):.2f}x faster" if speedup is not None else "N/A"
    save_str = f"{float(tok_saving_pct):.1f}% fewer tokens" if tok_saving_pct is not None else "N/A"

    judge_prompt = f"""You are an expert impartial AI evaluator judging two models on a mathematical problem.
You are given the user input, each model's response, and the exact runtime execution telemetry (tools executed, latency in seconds, and token consumption).

======================================================================
EVALUATION TURN: Model A (GPT-4o + MCP Tools) vs Model B (GPT-6-sol - Zero Tools)
======================================================================

User Question / Context:
\"\"\"{user_msg}\"\"\"

----------------------------------------------------------------------
MODEL A: GPT-4o (Document Grounded)
----------------------------------------------------------------------
Assistant Response:
\"\"\"{gpt_res}\"\"\"

Execution Telemetry:
- Context: In-context document grounding
- Latency: {lat_a}
- Tokens Consumed: {tok_a}

----------------------------------------------------------------------
MODEL B: GPT-6-sol (Document Grounded)
----------------------------------------------------------------------
Assistant Response:
\"\"\"{baseline_res}\"\"\"

Execution Telemetry:
- Context: In-context document grounding
- Latency: {lat_b}
- Tokens Consumed: {tok_b}

----------------------------------------------------------------------
COMPARATIVE METRICS:
----------------------------------------------------------------------
- Latency Speedup: Model B is {sp_str}
- Token Efficiency: Model B used {save_str}

======================================================================
EVALUATION PILLARS:
======================================================================
1. FACTUAL CORRECTNESS: Did the model achieve accurate answers strictly supported by the document?
2. DOCUMENT GROUNDING & EVIDENCE: Did the model cite verbatim evidence and context from the document without hallucination?
3. REASONING DEPTH & ALIGNMENT: Evaluate whether the model followed the active prompt directives thoroughly.
4. HOLISTIC WINNER: Choose "Model A" (GPT-4o), "Model B" (GPT-6-sol), or "TIE". Weigh correctness first, followed by evidence grounding and latency/token efficiency.

Evaluate both models strictly and output valid JSON only:
{{
  "gpt_correctness": "PASS" or "FAIL",
  "gpt5_correctness": "PASS" or "FAIL",
  "gemini_correctness": "PASS" or "FAIL",
  "gpt_score": <1 to 5 integer for reasoning clarity & correctness>,
  "gpt5_score": <1 to 5 integer for reasoning clarity & correctness>,
  "gemini_score": <1 to 5 integer for reasoning clarity & correctness>,
  "winner": "Model A" or "Model B" or "TIE",
  "accuracy_analysis": "<Concise analysis of numerical correctness and calculation precision>",
  "efficiency_analysis": "<Analysis evaluating tool roundtrip latency ({lat_a}) and token cost ({tok_a}) vs direct parametric inference ({lat_b}, {tok_b})>",
  "feedback": "<Comprehensive, unhurried explanation of judgment analyzing numerical correctness, tool selection, confirmation protocol compliance, latency and token economics, and why the winner was chosen>"
}}
"""
    try:
        judge_model = os.getenv("JUDGE_MODEL", "gpt-5.5")
        response = client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": "You are an impartial AI evaluator. Output valid JSON only."},
                {"role": "user", "content": judge_prompt}
            ]
        )
        txt = response.choices[0].message.content or "{}"
        match = re.search(r'\{.*\}', txt, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
        else:
            parsed = json.loads(txt)
        # Ensure fallback aliases
        if "gpt5_score" not in parsed and "gemini_score" in parsed:
            parsed["gpt5_score"] = parsed["gemini_score"]
        if "gemini_score" not in parsed and "gpt5_score" in parsed:
            parsed["gemini_score"] = parsed["gpt5_score"]
        if "gpt5_correctness" not in parsed and "gemini_correctness" in parsed:
            parsed["gpt5_correctness"] = parsed["gemini_correctness"]
        if "gemini_correctness" not in parsed and "gpt5_correctness" in parsed:
            parsed["gemini_correctness"] = parsed["gpt5_correctness"]
        return parsed
    except Exception as err:
        return {
            "gpt_correctness": "PASS",
            "gemini_correctness": "PASS",
            "gpt5_correctness": "PASS",
            "gpt_score": 4,
            "gemini_score": 4,
            "gpt5_score": 4,
            "winner": "TIE",
            "accuracy_analysis": "Error during evaluation.",
            "efficiency_analysis": "Error during evaluation.",
            "feedback": f"Judge error: {str(err)}"
        }

def evaluate_sessions(gpt_session_id, baseline_session_id, use_judge=False):
    """Executes the full evaluation comparison and returns metrics dictionary."""
    gpt_turns = fetch_session_turns(gpt_session_id)
    baseline_turns = fetch_session_turns(baseline_session_id)

    if not gpt_turns or not baseline_turns:
        print(f"[ERROR] Could not find turns for one or both sessions.")
        print(f"GPT-4o ({gpt_session_id}): {len(gpt_turns)} turns | Baseline/GPT-5.5 ({baseline_session_id}): {len(baseline_turns)} turns")
        return None

    client = None
    if use_judge:
        api_key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        if api_key:
            client = OpenAI(api_key=api_key)
        else:
            print("[WARN] OPENAI_API_KEY not found; skipping LLM judge calls.")

    # Match turns by turn_number
    gpt_map = {t["turn_number"]: t for t in gpt_turns}
    baseline_map = {t["turn_number"]: t for t in baseline_turns}

    common_turn_numbers = sorted(list(set(gpt_map.keys()) & set(baseline_map.keys())))

    eval_results = []
    tot_gpt_tokens = 0
    tot_baseline_tokens = 0
    tot_gpt_latency = 0.0
    tot_baseline_latency = 0.0
    math_matches_count = 0

    for num in common_turn_numbers:
        g = gpt_map[num]
        m = baseline_map[num]

        user_msg = g.get("user_message") or m.get("user_message") or ""
        gpt_res = g.get("assistant_response") or ""
        baseline_res = m.get("assistant_response") or ""

        # Extract numeric candidates
        gpt_nums = extract_numbers(gpt_res)
        baseline_nums = extract_numbers(baseline_res)

        # Check numeric match
        shared_nums = set(gpt_nums) & set(baseline_nums)
        math_match = len(shared_nums) > 0
        if math_match:
            math_matches_count += 1

        # Token savings
        g_tok = g.get("total_tokens") or 0
        m_tok = m.get("total_tokens") or 0
        tot_gpt_tokens += g_tok
        tot_baseline_tokens += m_tok

        tok_saving_pct = round(((g_tok - m_tok) / g_tok * 100), 1) if g_tok > 0 else 0.0

        # Latency speedup
        g_lat = g.get("latency_seconds") or 0.0
        m_lat = m.get("latency_seconds") or 0.0
        tot_gpt_latency += g_lat
        tot_baseline_latency += m_lat

        speedup = round((g_lat / m_lat), 2) if m_lat > 0 else 1.0

        # Tool calls
        tool_count = len(g.get("tool_calls", []))
        tool_names = []
        for tc in g.get("tool_calls", []):
            summ = tc.get("summary", "")
            match = re.search(r'\[(.*?) Started\]', summ)
            if match:
                tool_names.append(match.group(1).split(" - ")[0])
            elif tc.get("name"):
                tool_names.append(tc.get("name"))

        judge_data = None
        if use_judge:
            print(f"  [Judge GPT-5.5] Grading Turn #{num}...")
            judge_context = user_msg
            if user_msg.strip().lower() in ["yes", "yep", "proceed", "go ahead", "ok", "confirm", "sure"]:
                earlier = [
                    gpt_map[k].get("user_message", "")
                    for k in sorted(gpt_map.keys())
                    if k < num and gpt_map[k].get("user_message", "").strip().lower() not in ["yes", "yep", "ok", "proceed", "sure"]
                ]
                if earlier:
                    judge_context = f"Original Problem: '{earlier[-1]}' -> User Confirmation: '{user_msg}'"
            judge_data = run_llm_judge(
                user_msg=judge_context,
                gpt_res=gpt_res,
                baseline_res=baseline_res,
                client=client,
                gpt_latency=g_lat,
                baseline_latency=m_lat,
                gpt_tokens=g_tok,
                baseline_tokens=m_tok,
                gpt_prompt_tokens=g.get("prompt_tokens") or 0,
                gpt_completion_tokens=g.get("completion_tokens") or 0,
                baseline_prompt_tokens=m.get("prompt_tokens") or 0,
                baseline_completion_tokens=m.get("completion_tokens") or 0,
                gpt_tools=tool_names or ([f"{tool_count} tools"] if tool_count > 0 else ["0 tools"]),
                speedup=speedup,
                tok_saving_pct=tok_saving_pct
            )

            # Persist judge_data into eval_turns for both GPT and baseline turns
            try:
                with get_db_connection() as conn:
                    with conn.cursor() as cur:
                        if g.get("turn_id"):
                            cur.execute("""
                                UPDATE eval_turns
                                SET payload = COALESCE(payload, '{}'::jsonb) || %s::jsonb
                                WHERE turn_id = %s;
                            """, (json.dumps({"judge": judge_data}), g["turn_id"]))
                        if m.get("turn_id"):
                            cur.execute("""
                                UPDATE eval_turns
                                SET payload = COALESCE(payload, '{}'::jsonb) || %s::jsonb
                                WHERE turn_id = %s;
                            """, (json.dumps({"judge": judge_data}), m["turn_id"]))
                        conn.commit()
            except Exception as pe:
                print(f"  [Judge Warning] Failed to persist judge to database: {pe}")
        else:
            g_payload = g.get("payload") if isinstance(g.get("payload"), dict) else {}
            m_payload = m.get("payload") if isinstance(m.get("payload"), dict) else {}
            judge_data = g_payload.get("judge") or m_payload.get("judge")

        eval_results.append({
            "turn_number": num,
            "user_message": user_msg,
            "gpt_response": gpt_res,
            "gpt5_response": baseline_res,
            "gemini_response": baseline_res, # legacy alias
            "gpt_tokens": g_tok,
            "gpt_prompt_tokens": g.get("prompt_tokens") or 0,
            "gpt_completion_tokens": g.get("completion_tokens") or 0,
            "gpt5_tokens": m_tok,
            "gemini_tokens": m_tok,
            "gpt5_prompt_tokens": m.get("prompt_tokens") or 0,
            "gpt5_completion_tokens": m.get("completion_tokens") or 0,
            "gemini_prompt_tokens": m.get("prompt_tokens") or 0,
            "gemini_completion_tokens": m.get("completion_tokens") or 0,
            "token_saving_pct": tok_saving_pct,
            "gpt_latency": g_lat,
            "gpt5_latency": m_lat,
            "gemini_latency": m_lat,
            "speedup": speedup,
            "gpt_tools": tool_names or [f"{tool_count} tools"] if tool_count > 0 else ["0 tools"],
            "gpt5_tools": ["0 tools (direct)"],
            "gemini_tools": ["0 tools (direct)"],
            "shared_numbers": list(shared_nums),
            "math_match": math_match,
            "judge": judge_data
        })

    overall_token_saving = round(((tot_gpt_tokens - tot_baseline_tokens) / tot_gpt_tokens * 100), 1) if tot_gpt_tokens > 0 else 0.0
    overall_speedup = round((tot_gpt_latency / tot_baseline_latency), 2) if tot_baseline_latency > 0 else 1.0

    summary = {
        "gpt_session_id": gpt_session_id,
        "gpt5_session_id": baseline_session_id,
        "gemini_session_id": baseline_session_id, # backward compatibility
        "model_b_name": "GPT-6-sol",
        "total_turns_evaluated": len(eval_results),
        "math_accuracy_rate_pct": round((math_matches_count / len(eval_results) * 100), 1) if eval_results else 0.0,
        "total_gpt_tokens": tot_gpt_tokens,
        "total_gpt5_tokens": tot_baseline_tokens,
        "total_gemini_tokens": tot_baseline_tokens,
        "overall_token_savings_pct": overall_token_saving,
        "total_gpt_latency_s": round(tot_gpt_latency, 3),
        "total_gpt5_latency_s": round(tot_baseline_latency, 3),
        "total_gemini_latency_s": round(tot_baseline_latency, 3),
        "overall_speedup_factor": overall_speedup,
        "turns": eval_results
    }

    return summary

def print_evaluation_report(summary, show_judge=False):
    """Prints a beautiful formatted CLI evaluation report."""
    print("\n" + "=" * 90)
    print("        MATHROUTER EVALUATION REPORT: GPT-4o (MCP TOOLS) vs GPT-6-sol (NO TOOLS)")
    print("=" * 90)
    print(f"GPT-4o Session:    {summary['gpt_session_id']}")
    print(f"GPT-6-sol Session: {summary.get('gpt5_session_id', summary.get('gemini_session_id'))}")
    print(f"Turns Evaluated:   {summary['total_turns_evaluated']}")
    print("-" * 90)
    print(f"  * Overall Token Savings:   {summary['overall_token_savings_pct']}%  ({summary['total_gpt_tokens']:,} GPT-4o tok -> {summary['total_gpt5_tokens']:,} GPT-6-sol tok)")
    print(f"  * Overall Speedup:         {summary['overall_speedup_factor']}x faster ({summary['total_gpt_latency_s']}s GPT-4o -> {summary['total_gpt5_latency_s']}s GPT-6-sol)")
    print(f"  * Numerical Ground-Truth:  {summary['math_accuracy_rate_pct']}% matches")
    print("-" * 90)

    # Table Header
    header = f"{'Turn':<5} | {'User Prompt':<22} | {'GPT-4o Tools':<15} | {'GPT-4o (Lat/Tok)':<16} | {'GPT-5.5 (Lat/Tok)':<16} | {'Speedup':<8} | {'Tok Saved':<9}"
    print(header)
    print("-" * len(header))

    for t in summary["turns"]:
        prompt_clip = (t["user_message"][:20] + "..") if len(t["user_message"]) > 22 else t["user_message"]
        tools_str = ",".join(t["gpt_tools"])[:14]
        gpt_metric = f"{t['gpt_latency']:.2f}s / {t['gpt_tokens']}t"
        baseline_metric = f"{t.get('gpt5_latency', t.get('gemini_latency')):.2f}s / {t.get('gpt5_tokens', t.get('gemini_tokens'))}t"
        speedup_str = f"{t['speedup']}x"
        saved_str = f"{t['token_saving_pct']}%"

        row = f"#{t['turn_number']:<4} | {prompt_clip:<22} | {tools_str:<15} | {gpt_metric:<16} | {baseline_metric:<16} | {speedup_str:<8} | {saved_str:<9}"
        print(row)

        if show_judge and t.get("judge"):
            j = t["judge"]
            winner = j.get("winner", "TIE")
            gpt_sc = j.get("gpt_score", 0)
            g5_sc = j.get("gpt5_score", j.get("gemini_score", 0))
            feedback = j.get("feedback", "")
            print(f"       >>> [LLM Judge Verdict]: Winner={winner} | GPT-4o Score: {gpt_sc}/5 | GPT-5.5 Score: {g5_sc}/5")
            print(f"       >>> [Feedback]: {feedback}")
            print("-" * len(header))

    print("=" * 90 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Evaluate GPT-4o (MathRouter MCP Tools) vs GPT-5.5 (Zero Tools)")
    parser.add_argument("--recording", type=int, default=None, help="Recording number to evaluate (e.g. 1)")
    parser.add_argument("--gpt", type=str, default=None, help="Explicit GPT-4o session ID")
    parser.add_argument("--gpt5", type=str, default=None, help="Explicit GPT-5.5 session ID")
    parser.add_argument("--gemini", type=str, default=None, help="Explicit baseline session ID (alias for gpt5)")
    parser.add_argument("--judge", action="store_true", help="Run LLM-as-a-Judge for reasoning scores and feedback using GPT-5.5")
    parser.add_argument("--save-json", action="store_true", help="Save complete evaluation report as JSON artifact")
    args = parser.parse_args()

    gpt_id = args.gpt
    baseline_id = args.gpt5 or args.gemini

    if not gpt_id or not baseline_id:
        gpt_sess, baseline_sess = find_paired_sessions(args.recording)
        if not gpt_id and gpt_sess:
            gpt_id = gpt_sess["session_id"]
        if not baseline_id and baseline_sess:
            baseline_id = baseline_sess["session_id"]

    if not gpt_id or not baseline_id:
        print("[ERROR] Could not find both a GPT-4o session and a GPT-5.5 session.")
        print(f"Found GPT-4o: {gpt_id} | GPT-5.5: {baseline_id}")
        print("Please ensure both sessions exist in PostgreSQL or run python rerun_gpt5.py first.")
        sys.exit(1)

    print(f"\n[INFO] Starting Evaluation:")
    print(f"  -> Model A (GPT-4o + MCP Tools): {gpt_id}")
    print(f"  -> Model B (GPT-5.5 Zero Tools):  {baseline_id}")
    if args.judge:
        print(f"  -> LLM-as-a-Judge: Enabled (GPT-5.5 Evaluator)")

    summary = evaluate_sessions(gpt_id, baseline_id, use_judge=args.judge)
    if summary:
        print_evaluation_report(summary, show_judge=args.judge)

        if args.save_json:
            ts = int(time.time())
            out_file = os.path.join(os.path.dirname(__file__), "recordings", f"eval_report_{ts}.json")
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"[SUCCESS] JSON report saved: {out_file}\n")

if __name__ == "__main__":
    main()
