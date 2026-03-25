"""
Module 3 – LLM Agent Module
============================
Mirrors the ESGReveal LLM Agent Module:

  1. SearchTerm Generator  → uses metadata.search_term list
  2. Vector Retrieval      → top-N via FAISS (preprocessing_module)
  3. Semantic Re-ranking   → cosine re-rank with the same embed model
  4. Prompt Generator      → fills ESGReveal-style prompt template
  5. LLM Answering         → calls Qwen (DashScope OpenAI-compatible API)
  6. Output                → JSON per indicator: {KPI, Topic, Disclosure, Value, Unit, Action}
"""

import json, re
import numpy as np
from typing import List, Dict, Any

from openai import OpenAI

from config import (
    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_URL,
    LLM_MODEL,
    TOP_N,
    TOP_K,
    REQUEST_INTERVAL_SECONDS,
)
from metadata_module import MetadataEntry
from preprocessing_module import ReportPreprocessor


_client = None
_DASHSCOPE_CN_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_DASHSCOPE_INTL_BASE = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


def _get_client():
    global _client
    if _client is None:
        if (not DASHSCOPE_API_KEY) or ("YOUR_DASHSCOPE_API_KEY_HERE" in DASHSCOPE_API_KEY):
            raise RuntimeError(
                "DashScope API key is missing. Please set DASHSCOPE_API_KEY "
                "(or OPENAI_API_KEY for backward compatibility) in .env."
            )
        _client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)
    return _client


def _normalize_base_url(url: str) -> str:
    return (url or "").strip().rstrip("/")


def _alternate_dashscope_base_url(url: str):
    base = _normalize_base_url(url)
    if base == _DASHSCOPE_CN_BASE:
        return _DASHSCOPE_INTL_BASE
    if base == _DASHSCOPE_INTL_BASE:
        return _DASHSCOPE_CN_BASE
    return None


def _create_completion(client, prompt: str):
    return client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise data extraction assistant. "
                    "Always respond with valid JSON only."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=512,
    )


# ── Prompt template ───────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """You are an expert analyst tasked with extracting structured environmental and biodiversity data from official reports.

===== REFERENCE CONTENT (Retrieved from the report) =====
{reference_content}

===== EXPERT KNOWLEDGE =====
{expert_knowledge}

===== EXTRACTION TASK =====
Aspect  : {aspect}
KPI     : {kpi}
Topic   : {topic}
Quantity: {quantity}

Please use ONLY the Reference Content above. Do NOT deviate from the provided material.

In terms of [{aspect}], extract the [{kpi}] related to [{topic}] and give [{quantity}].

Return your answer in the following JSON format:
{{
  "KPI"        : "<KPI name>",
  "Topic"      : "<specific topic extracted>",
  "Disclosure" : "<yes | no | partial>",
  "Value"      : "<numerical value or N/A>",
  "Unit"       : "<unit of measurement or N/A>",
  "Action"     : "<key action described, or N/A>",
  "EvidenceQuote": "<copy exact words from REFERENCE CONTENT, or N/A>"
}}

If the information is not found in the reference content, set Disclosure to "no" and all other fields to "N/A".
Rules for EvidenceQuote:
- Must be copied verbatim from REFERENCE CONTENT (no paraphrasing).
- Keep it short (about 1-2 sentences).
- If no reliable evidence exists, output "N/A".
"""


# ── Semantic re-ranking ────────────────────────────────────────────────────────
def rerank(query: str, candidates: List[tuple], model, top_k: int = TOP_K) -> List[str]:
    """
    Re-rank candidate (score, text) pairs by semantic similarity.
    Mirrors coROM re-ranking in ESGReveal §3.4.1.
    """
    if not candidates:
        return []
    texts  = [c[1] for c in candidates]
    q_emb  = model.encode([query], normalize_embeddings=True)
    c_embs = model.encode(texts,   normalize_embeddings=True)
    scores = (c_embs @ q_emb.T).squeeze()
    if scores.ndim == 0:
        scores = np.array([float(scores)])
    ranked_idx = np.argsort(scores)[::-1][:top_k]
    return [texts[i] for i in ranked_idx]


# ── Qwen (OpenAI-compatible) LLM call ─────────────────────────────────────────
def call_llm(prompt: str, max_retries: int = 5) -> str:
    import time
    global _client
    client = _get_client()
    for attempt in range(max_retries):
        try:
            response = _create_completion(client, prompt)
            return (response.choices[0].message.content or "").strip()

        except Exception as e:
            err = str(e)
            # Common DashScope pitfall: key is created on intl site, but CN endpoint is used (or vice versa).
            if "invalid_api_key" in err:
                current_base = _normalize_base_url(DASHSCOPE_BASE_URL)
                alt_base = _alternate_dashscope_base_url(current_base)
                if alt_base:
                    try:
                        print(
                            "  [AuthHint] 当前端点鉴权失败，尝试切换到另一站点："
                            f"{alt_base}"
                        )
                        alt_client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=alt_base)
                        alt_resp = _create_completion(alt_client, prompt)
                        _client = alt_client
                        return (alt_resp.choices[0].message.content or "").strip()
                    except Exception:
                        pass
                raise RuntimeError(
                    "DashScope 鉴权失败（invalid_api_key）。请检查 API Key 是否有效，"
                    "并确认 DASHSCOPE_BASE_URL 与 Key 所在站点一致："
                    f"中国站={_DASHSCOPE_CN_BASE}，国际站={_DASHSCOPE_INTL_BASE}。"
                ) from e
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                # 从错误信息里提取建议等待秒数，默认等 15s
                import re as _re
                match = _re.search(r"retry in (\d+)", err)
                wait  = int(match.group(1)) + 3 if match else 15
                print(f"  [RateLimit] 触发限速，等待 {wait}s 后重试 "
                      f"(attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise   # 非限速错误直接抛出

    raise RuntimeError(f"LLM 调用在 {max_retries} 次重试后仍然失败。")


# ── JSON parsing helper ───────────────────────────────────────────────────────
def parse_json_response(raw: str) -> Dict[str, Any]:
    """Extract JSON from LLM response, handling markdown code blocks."""
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"raw_response": raw, "parse_error": True}


# ── Main agent class ──────────────────────────────────────────────────────────
# class LLMAgent:
#     def __init__(self, preprocessor: ReportPreprocessor):
#         self.preprocessor = preprocessor
#         self.embed_model  = preprocessor.model

#     def process_entry(self, entry: MetadataEntry) -> Dict[str, Any]:
#         """
#         Full pipeline for a single MetadataEntry:
#           generate query → retrieve → rerank → build prompt → LLM → parse
#         """
#         # 1. SearchTerm → query string
#         query = " ".join(entry.search_term[:3])

#         # 2. TopN vector retrieval
#         candidates = self.preprocessor.retrieve(query, top_n=TOP_N)

#         # 3. TopK semantic re-ranking
#         top_texts = rerank(query, candidates, self.embed_model, top_k=TOP_K)

#         # 4. Build prompt
#         reference = "\n---\n".join(top_texts) if top_texts else "No relevant content found."
#         prompt = PROMPT_TEMPLATE.format(
#             reference_content = reference,
#             expert_knowledge  = entry.knowledge or "No additional expert knowledge provided.",
#             aspect            = entry.aspect,
#             kpi               = entry.kpi,
#             topic             = entry.topic,
#             quantity          = entry.quantity,
#         )

#         # 5. LLM answering
#         raw = call_llm(prompt)

#         # 6. Parse & enrich
#         result = parse_json_response(raw)
#         result["_aspect"]      = entry.aspect
#         result["_search_term"] = entry.search_term
#         return result

#     def run(self, entries: List[MetadataEntry], max_entries: int = None) -> List[Dict]:
#         """Process all (or up to max_entries) metadata entries."""
#         subset  = entries[:max_entries] if max_entries else entries
#         results = []
#         for i, entry in enumerate(subset):
#             print(f"[LLMAgent] Processing {i+1}/{len(subset)}: {entry.aspect} – {entry.kpi[:60]}...")
#             r = self.process_entry(entry)
#             results.append(r)
#         return results
class LLMAgent:
    def __init__(self, preprocessor: ReportPreprocessor,
                 request_interval: float = REQUEST_INTERVAL_SECONDS):
        """
        request_interval: 每次 API 调用之间的最小间隔（秒）
                          免费版 5 RPM → 至少 12s，建议设 13s 留余量
        """
        self.preprocessor      = preprocessor
        self.embed_model       = preprocessor.model
        self.request_interval  = request_interval

    def process_entry(self, entry: MetadataEntry) -> Dict[str, Any]:
        # Use curated metadata terms (aspect + topic + element terms + KPI element text)
        # so retrieval reflects the known constraints from metadata.
        query_parts = [entry.aspect, entry.topic] + entry.search_term + [entry.kpi]
        query = " ".join(list(dict.fromkeys([p for p in query_parts if p]))[:8])
        candidates = self.preprocessor.retrieve(query, top_n=TOP_N)
        top_texts  = rerank(query, candidates, self.embed_model, top_k=TOP_K)

        reference = "\n---\n".join(top_texts) if top_texts else "No relevant content found."
        prompt = PROMPT_TEMPLATE.format(
            reference_content = reference,
            expert_knowledge  = entry.knowledge or "No additional expert knowledge provided.",
            aspect            = entry.aspect,
            kpi               = entry.kpi,
            topic             = entry.topic,
            quantity          = entry.quantity,
        )

        raw    = call_llm(prompt)
        result = parse_json_response(raw)
        result["_aspect"]      = entry.aspect
        result["_search_term"] = entry.search_term
        # Keep raw retrieved snippets for traceability and manual PDF cross-check.
        result["_retrieved_reference_1"] = top_texts[0] if len(top_texts) > 0 else "N/A"
        result["_retrieved_reference_2"] = top_texts[1] if len(top_texts) > 1 else "N/A"
        return result

    def run(self, entries: List[MetadataEntry], max_entries: int = None) -> List[Dict]:
        import time
        subset  = entries[:max_entries] if max_entries else entries
        results = []
        total   = len(subset)

        for i, entry in enumerate(subset):
            print(f"[LLMAgent] ({i+1}/{total}) {entry.aspect} – {entry.kpi[:55]}...")
            r = self.process_entry(entry)
            results.append(r)

            # 限速控制：最后一条不需要等待
            if i < total - 1:
                print(f"  [RateLimit] 等待 {self.request_interval}s（5 RPM 限制）...")
                time.sleep(self.request_interval)

        return results


if __name__ == "__main__":
    from metadata_module import load_metadata

    entries = load_metadata("data/GBF_Monitoring_Framework_Supplementary_Tables.xlsx")
    proc    = ReportPreprocessor(
        "data/EPA-Report-East-Rockingham-Waste-to-Energy-Revised-Proposal.pdf",
        report_name="epa_report"
    )
    proc.load()
    agent   = LLMAgent(proc)

    results = agent.run(entries, max_entries=5)
    print(json.dumps(results, indent=2, ensure_ascii=False))
