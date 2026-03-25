"""
Module 1 – Metadata Module
===========================
Mirrors the ESG Metadata Module in ESGReveal, but adapted for
the GBF (Global Biodiversity Framework) xlsx file.

Structure: <Aspect, KPI, Topic, Quantity, SearchTerm, Knowledge>
  - Aspect      = Goal / Target group (e.g. "Goal A", "Target 3")
  - KPI         = Element description (the specific indicator)
  - Topic       = Derived sub-topic
  - Quantity    = extraction type: "Numerical" | "Key Actions" | "Status"
  - SearchTerm  = retrieval keywords
  - Knowledge   = short domain context injected into the LLM prompt

Notes
-----
For the "preliminary trial" sheet, this loader uses curated fields:
  - KPI (overview text)
  - Goal/Target
  - Target search term/topic
  - Element Search Term
  - Element (KPI or Topic?)
instead of only auto-generating keywords.
"""

import re
import pandas as pd
from dataclasses import dataclass, field
from typing import List


@dataclass
class MetadataEntry:
    aspect:      str
    kpi:         str
    topic:       str
    quantity:    str
    search_term: List[str] = field(default_factory=list)
    knowledge:   str = ""


# ── Knowledge snippets keyed on Aspect ────────────────────────────────────────
ASPECT_KNOWLEDGE = {
    "Goal A": (
        "Goal A of the Kunming-Montreal GBF focuses on maintaining and restoring "
        "ecosystem integrity, connectivity and resilience, and on halting species extinction."
    ),
    "Goal B": (
        "Goal B concerns the sustainable use of biodiversity and valuing nature's "
        "contributions to people (NCP / ecosystem services)."
    ),
    "Goal C": (
        "Goal C addresses fair and equitable sharing of benefits from genetic resources, "
        "digital sequence information (DSI), and traditional knowledge."
    ),
    "Goal D": (
        "Goal D ensures adequate means of implementation: finance, capacity, technology, "
        "and cooperation for all Parties."
    ),
}

# Generic knowledge for numbered Targets
TARGET_KNOWLEDGE_TEMPLATES = {
    "1":  "Target 1 – spatial planning to halt biodiversity loss; key metrics: % area under effective management.",
    "2":  "Target 2 – 30% of degraded ecosystems under effective restoration by 2030.",
    "3":  "Target 3 – 30x30: at least 30% of terrestrial, inland water, marine and coastal areas protected by 2030.",
    "4":  "Target 4 – prevent human-induced extinction; maintain genetic diversity of wild/domestic species.",
    "5":  "Target 5 – sustainable, safe and legal harvesting and trade of wild species.",
    "6":  "Target 6 – reduce invasive alien species introduction rates by 50%; eradicate on priority sites.",
    "7":  "Target 7 – halve nutrient loss and pesticide risk; eliminate plastic pollution.",
    "8":  "Target 8 – minimise climate change and ocean acidification impacts on biodiversity.",
    "9":  "Target 9 – sustainable management of wild species providing social and economic benefits.",
    "10": "Target 10 – sustainable management of agriculture, aquaculture, fisheries and forestry.",
    "11": "Target 11 – restore and maintain nature-based solutions and ecosystem functions.",
    "12": "Target 12 – increase area, quality and connectivity of green/blue urban spaces.",
    "13": "Target 13 – ABS measures for genetic resources and traditional knowledge.",
    "14": "Target 14 – integrate biodiversity into policies, planning, EIAs and national accounts.",
    "15": "Target 15 – business disclosure of biodiversity risks, dependencies and impacts.",
    "16": "Target 16 – sustainable consumption; halve food waste by 2030.",
    "17": "Target 17 – biosafety measures and biotechnology handling.",
    "18": "Target 18 – eliminate, phase out or reform harmful incentives ≥$500B/yr.",
    "19": "Target 19 – mobilise ≥$200B/yr for biodiversity; international flows ≥$20-30B.",
    "20": "Target 20 – capacity building, technology transfer and scientific cooperation.",
    "21": "Target 21 – ensure access to best available data, information and knowledge.",
    "22": "Target 22 – ensure participation and access to justice for IPLCs, women, youth.",
    "23": "Target 23 – gender equality in biodiversity governance.",
}


def _clean_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).replace("\r", "\n").strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalize_aspect_label(raw_aspect: str) -> str:
    """Normalize labels like 'GOAL A ...' or 'TARGET 7 ...' to 'Goal A' / 'Target 7'."""
    txt = _clean_text(raw_aspect)
    if not txt:
        return ""

    m = re.search(r"\b(Goal)\s*([A-D])\b", txt, flags=re.IGNORECASE)
    if m:
        return f"Goal {m.group(2).upper()}"

    m = re.search(r"\b(Target)\s*(\d{1,2})\b", txt, flags=re.IGNORECASE)
    if m:
        return f"Target {int(m.group(2))}"

    first_line = txt.split("\n", 1)[0].strip()
    return first_line


def _split_terms(raw: str) -> List[str]:
    text = _clean_text(raw)
    if not text:
        return []
    parts = re.split(r"[,;/|\n]| and | AND ", text)
    terms = [p.strip() for p in parts if p and p.strip()]
    return terms


def _derive_search_terms(kpi: str, aspect: str) -> List[str]:
    """Auto-generate search terms from the KPI text."""
    # Remove parenthetical suffixes
    clean = re.sub(r"\(.*?\)", "", kpi).strip()
    # Split on common stop words / punctuation to get noun phrases
    tokens = re.split(r"[,;./]| and | or | to | of | the | by | for | in ", clean)
    terms = [t.strip() for t in tokens if len(t.strip()) > 4][:4]
    # Always include the aspect label
    terms.insert(0, aspect)
    return list(dict.fromkeys(terms))   # deduplicate preserving order


def _infer_quantity(kpi: str) -> str:
    """Classify whether we expect numerical data, key actions, or a status."""
    kpi_lower = kpi.lower()
    if any(w in kpi_lower for w in ["ensure", "take", "encourage", "apply", "foster",
                                     "establish", "strengthen", "implement", "promote",
                                     "mainstream", "identify", "protect", "manage"]):
        return "Key Actions"
    if any(w in kpi_lower for w in ["%", "per cent", "halve", "tenfold", "at least",
                                     "reduce", "increase", "billion", "area", "rate"]):
        return "Numerical"
    return "Status"


def _get_knowledge(aspect: str) -> str:
    if aspect in ASPECT_KNOWLEDGE:
        return ASPECT_KNOWLEDGE[aspect]
    match = re.match(r"Target (\d+)", aspect)
    if match:
        return TARGET_KNOWLEDGE_TEMPLATES.get(match.group(1), "")
    return ""


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if not item:
            continue
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _find_column(df: pd.DataFrame, candidates: List[str]) -> str:
    cols = list(df.columns)
    lowered = {str(c).strip().lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lowered:
            return lowered[c.lower()]
    return ""


def _load_preliminary_trial(df: pd.DataFrame) -> List[MetadataEntry]:
    """
    Parse curated metadata from the 'preliminary trial' sheet.
    The sheet uses merged-like rows, so we forward-fill group-level fields.
    """
    col_kpi_overview = _find_column(df, ["KPI"])
    col_aspect = _find_column(df, ["Goal/Target"])
    col_target_term = _find_column(df, ["Target search term/topic"])
    col_element_term = _find_column(df, ["Element Search Term"])
    col_element = _find_column(df, ["Element (KPI or Topic?)"])
    col_secondary = _find_column(df, ["二级检索"])
    col_topic_hint = next(
        (
            c
            for c in df.columns
            if str(c).strip().startswith("Topic")
        ),
        "",
    )
    col_sheet_knowledge = next((c for c in df.columns if "Knowledge" in str(c)), "")

    required = [col_kpi_overview, col_aspect, col_target_term, col_element_term, col_element]
    if not all(required):
        raise ValueError(
            "preliminary trial sheet is missing required columns. "
            "Expected KPI / Goal-Target / Target search term-topic / "
            "Element Search Term / Element (KPI or Topic?)."
        )

    filled = df.copy()
    # Only fill group-level fields. Do NOT forward-fill term columns across rows,
    # otherwise Goal/Target blocks can inherit irrelevant terms from previous blocks.
    for c in [col_kpi_overview, col_aspect]:
        filled[c] = filled[c].ffill()
    if col_topic_hint:
        filled[col_topic_hint] = filled[col_topic_hint].ffill()
    if col_sheet_knowledge:
        filled[col_sheet_knowledge] = filled[col_sheet_knowledge].ffill()

    entries: List[MetadataEntry] = []
    for _, row in filled.iterrows():
        element_text = _clean_text(row[col_element])
        if not element_text:
            continue

        raw_aspect = _clean_text(row[col_aspect])
        aspect = _normalize_aspect_label(raw_aspect)

        kpi_overview = _clean_text(row[col_kpi_overview])
        target_term_raw = _clean_text(row[col_target_term])
        element_term_raw = _clean_text(row[col_element_term])
        secondary_term = _clean_text(row[col_secondary]) if col_secondary else ""
        topic_hint = _clean_text(row[col_topic_hint]) if col_topic_hint else ""
        sheet_knowledge = _clean_text(row[col_sheet_knowledge]) if col_sheet_knowledge else ""

        # Fallbacks when curated term cells are blank on a row.
        # Goal C/D and some Target rows intentionally leave these empty.
        target_term = target_term_raw or aspect
        element_term = element_term_raw or element_text

        # Known terms from the curated sheet are preferred over fully generated ones.
        terms = _dedupe_keep_order(
            [aspect]
            + _split_terms(target_term)
            + _split_terms(element_term)
            + _split_terms(secondary_term)
            + _derive_search_terms(element_text, aspect)
            + [element_text]
        )

        topic = topic_hint or target_term or aspect
        qty = _infer_quantity(element_text)

        knowledge_parts = []
        if kpi_overview:
            knowledge_parts.append(f"GBF KPI overview: {kpi_overview}")
        if sheet_knowledge:
            knowledge_parts.append(f"Sheet guidance: {sheet_knowledge}")
        aspect_knowledge = _get_knowledge(aspect)
        if aspect_knowledge:
            knowledge_parts.append(aspect_knowledge)
        knowledge = "\n".join(knowledge_parts)

        entries.append(
            MetadataEntry(
                aspect=aspect,
                kpi=element_text,
                topic=topic,
                quantity=qty,
                search_term=terms,
                knowledge=knowledge,
            )
        )
    return entries


def _load_generic_sheet(df: pd.DataFrame) -> List[MetadataEntry]:
    """Fallback parser for sheets like 'Supplementary Table 2'."""
    if len(df.columns) < 2:
        raise ValueError("Metadata sheet requires at least 2 columns.")

    col_aspect = df.columns[0]
    col_elem = df.columns[1]
    entries: List[MetadataEntry] = []
    for _, row in df.iterrows():
        aspect = _clean_text(row[col_aspect])
        kpi = _clean_text(row[col_elem])
        if not aspect or not kpi:
            continue
        aspect = _normalize_aspect_label(aspect)
        topic = kpi.split(".")[0].strip() if "." in kpi else aspect
        qty = _infer_quantity(kpi)
        terms = _derive_search_terms(kpi, aspect)
        know = _get_knowledge(aspect)
        entries.append(
            MetadataEntry(
                aspect=aspect,
                kpi=kpi,
                topic=topic,
                quantity=qty,
                search_term=terms,
                knowledge=know,
            )
        )
    return entries


def load_metadata(xlsx_path: str, sheet_name: str = "preliminary trial") -> List[MetadataEntry]:
    """
    Read the GBF xlsx and return a list of MetadataEntry objects.

    Parameters
    ----------
    xlsx_path  : path to GBF_Monitoring_Framework_Supplementary_Tables.xlsx
    sheet_name : which sheet to read (default = 'preliminary trial')
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    df.columns = [str(c).strip() for c in df.columns]

    if _find_column(df, ["Element (KPI or Topic?)"]) and _find_column(df, ["Target search term/topic"]):
        entries = _load_preliminary_trial(df)
    else:
        entries = _load_generic_sheet(df)

    print(f"[MetadataModule] Loaded {len(entries)} indicator entries from '{sheet_name}'.")
    return entries


if __name__ == "__main__":
    import json
    entries = load_metadata("data/GBF_Monitoring_Framework_Supplementary_Tables.xlsx", sheet_name="preliminary trial")
    sample = entries[:3]
    for e in sample:
        print(json.dumps(e.__dict__, indent=2))
