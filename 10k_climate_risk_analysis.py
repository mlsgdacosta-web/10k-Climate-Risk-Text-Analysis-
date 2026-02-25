#!/usr/bin/env python3
""" Form 10-K text preprocessing & climate-risk analysis

Process:
- Within each 10-K, find passages that discuss climate risks.
- Provide sentiment analysis on these passages.
- Aggregate to company level and measure each company’s exposure on climate risks.
- Separate physical and transition risks.
- Try to clean and extract each Item within each 10-K.

How to run:
  python 10k_climate_risk_analysis.py --input-dir 10ks --output-dir outputs --write-items

Outputs (saved in --output-dir):
  - passages.csv         one row per extracted climate-risk passage
  - company_summary.csv  company-level exposure measures
  - items.jsonl          (optional) newline-delimited JSON of extracted Item text per filing

Notes:
- Climate passage extraction and physical/transition classification are keyword-based.
- Sentiment uses VADER (NLTK if available, otherwise vaderSentiment).
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# HTML parsing (preferred)
try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    BeautifulSoup = None

# Sentiment (VADER)
VADER_MODE = None
try:
    from nltk.sentiment import SentimentIntensityAnalyzer  # type: ignore
    VADER_MODE = "nltk"
except Exception:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
        VADER_MODE = "vaderSentiment"
    except Exception:
        SentimentIntensityAnalyzer = None


# -----------------------------
# Keyword dictionaries
# -----------------------------
CLIMATE_KEYWORDS = [
    "climate change", "global warming", "greenhouse", "ghg", "emissions",
    "decarbon", "carbon", "energy transition", "sustainable",
    "tcfd", "sasb", "scope 1", "scope 2", "scope 3",
    "physical risk", "transition risk", "climate risk", "climate-related",
    "extreme weather", "carbon price", "carbon tax", "cap and trade", "renewable",
    "clean energy", "fossil fuel", "climate policy",
]

PHYSICAL_RISK_KEYWORDS = [
    "hurricane", "storm", "flood", "flooding", "wildfire", "fire", "drought",
    "heat", "heatwave", "extreme weather", "sea level", "sea-level",
    "rising sea", "coastal", "temperature", "precipitation", "tornado",
    "cyclone", "rainfall", "snowpack", "water scarcity", "water shortage",
    "climate-driven", "natural catastrophe",
]

TRANSITION_RISK_KEYWORDS = [
    "carbon tax", "cap and trade", "emissions trading", "reporting requirement",
    "net zero", "decarbon", "electrification", "renewable", "clean energy",
    "stranded asset", "energy transition", "technological change",
    "mitigation", "carbon price", "transition plan",
]


# -----------------------------
# Helpers
# -----------------------------

def read_text_file(path: Path) -> str:
    """Read a file with encoding fallbacks."""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return path.read_text(encoding="latin-1", errors="ignore")


def html_to_text(raw: str) -> str:
    """Convert HTML-ish text to plain text."""
    if BeautifulSoup is not None:
        # lxml is faster if installed; html.parser works otherwise
        try:
            soup = BeautifulSoup(raw, "lxml")
        except Exception:
            soup = BeautifulSoup(raw, "html.parser")

        for tag in soup(["script", "style", "noscript"]):
            tag.extract()

        txt = soup.get_text(separator="\n")
    else:
        # Fallback: strip tags (less reliable)
        txt = re.sub(r"<[^>]+>", " ", raw)

    txt = txt.replace("\r", "\n")
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()


def extract_edgar_header_fields(raw: str) -> Dict[str, Optional[str]]:
    """Extract common EDGAR header fields when present."""

    def grab(pattern: str) -> Optional[str]:
        m = re.search(pattern, raw, flags=re.IGNORECASE)
        return m.group(1).strip() if m else None

    return {
        "company_name": grab(r"COMPANY CONFORMED NAME:\s*(.+)"),
        "cik": grab(r"CENTRAL INDEX KEY:\s*([0-9]+)"),
        "period": grab(r"CONFORMED PERIOD OF REPORT:\s*([0-9]{8})"),
        "filed": grab(r"FILED AS OF DATE:\s*([0-9]{8})"),
    }


ITEM_PATTERNS = [
    ("1A", r"\bITEM\s*1A[\.:\-\s]"),
    ("1B", r"\bITEM\s*1B[\.:\-\s]"),
    ("1C", r"\bITEM\s*1C[\.:\-\s]"),
    ("1", r"\bITEM\s*1[\.:\-\s]"),
    ("2", r"\bITEM\s*2[\.:\-\s]"),
    ("3", r"\bITEM\s*3[\.:\-\s]"),
    ("4", r"\bITEM\s*4[\.:\-\s]"),
    ("5", r"\bITEM\s*5[\.:\-\s]"),
    ("6", r"\bITEM\s*6[\.:\-\s]"),
    ("7A", r"\bITEM\s*7A[\.:\-\s]"),
    ("7", r"\bITEM\s*7[\.:\-\s]"),
    ("8", r"\bITEM\s*8[\.:\-\s]"),
    ("9A", r"\bITEM\s*9A[\.:\-\s]"),
    ("9B", r"\bITEM\s*9B[\.:\-\s]"),
    ("9C", r"\bITEM\s*9C[\.:\-\s]"),
    ("9", r"\bITEM\s*9[\.:\-\s]"),
    ("10", r"\bITEM\s*10[\.:\-\s]"),
    ("11", r"\bITEM\s*11[\.:\-\s]"),
    ("12", r"\bITEM\s*12[\.:\-\s]"),
    ("13", r"\bITEM\s*13[\.:\-\s]"),
    ("14", r"\bITEM\s*14[\.:\-\s]"),
    ("15", r"\bITEM\s*15[\.:\-\s]"),
    ("16", r"\bITEM\s*16[\.:\-\s]"),
]


def split_into_items(text: str) -> Dict[str, str]:
    """Heuristic segmentation by 'Item X' headers."""
    hits: List[Tuple[int, str]] = []
    for code, pat in ITEM_PATTERNS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            hits.append((m.start(), code))

    if not hits:
        return {"FULL_TEXT": text}

    hits.sort(key=lambda x: x[0])

    # Deduplicate nearby hits (table of contents)
    deduped: List[Tuple[int, str]] = []
    last_pos = -10**9
    for pos, code in hits:
        if pos - last_pos < 200:
            continue
        deduped.append((pos, code))
        last_pos = pos

    items: Dict[str, str] = {}
    for i, (pos, code) in enumerate(deduped):
        end = deduped[i + 1][0] if i + 1 < len(deduped) else len(text)
        seg = text[pos:end].strip()

        # Filter out very short segments (TOC lines)
        if len(seg) < 500:
            continue

        # Keep the most substantial chunk seen for each item code
        if code not in items or len(seg) > len(items[code]):
            items[code] = seg

    return items if items else {"FULL_TEXT": text}


def tokenize_paragraphs(text: str) -> List[str]:
    """Split text into paragraph-like chunks."""
    parts = re.split(r"\n\s*\n", text)
    paras: List[str] = []
    for p in parts:
        p = re.sub(r"\s+", " ", p.strip())
        if len(p) >= 80:
            paras.append(p)
    return paras


def keyword_hits(text: str, keywords: List[str]) -> int:
    t = text.lower()
    return sum(1 for k in keywords if k in t)


def classify_risk_type(passage: str) -> str:
    """Return PHYSICAL, TRANSITION, BOTH, or UNSPECIFIED."""
    ph = keyword_hits(passage, PHYSICAL_RISK_KEYWORDS)
    tr = keyword_hits(passage, TRANSITION_RISK_KEYWORDS)
    if ph > 0 and tr > 0:
        return "BOTH"
    if ph > 0:
        return "PHYSICAL"
    if tr > 0:
        return "TRANSITION"
    return "UNSPECIFIED"


def ensure_vader() -> Optional["SentimentIntensityAnalyzer"]:
    """Create a VADER analyzer if available."""
    if SentimentIntensityAnalyzer is None:
        return None

    if VADER_MODE == "nltk":
        try:
            import nltk  # type: ignore

            try:
                nltk.data.find("sentiment/vader_lexicon.zip")
            except LookupError:
                nltk.download("vader_lexicon", quiet=True)
        except Exception:
            pass

    try:
        return SentimentIntensityAnalyzer()
    except Exception:
        return None


@dataclass
class FilingResult:
    file: str
    company_name: Optional[str]
    cik: Optional[str]
    period: Optional[str]
    filed: Optional[str]
    items: Dict[str, str]


def process_filing(path: Path) -> FilingResult:
    raw = read_text_file(path)
    header = extract_edgar_header_fields(raw)
    text = html_to_text(raw)
    items = split_into_items(text)
    return FilingResult(
        file=path.name,
        company_name=header.get("company_name"),
        cik=header.get("cik"),
        period=header.get("period"),
        filed=header.get("filed"),
        items=items,
    )


def extract_climate_passages(items: Dict[str, str], prefer_items: Optional[List[str]] = None) -> List[Tuple[str, str]]:
    """Return (item_code, paragraph) for paragraphs containing climate keywords."""
    out: List[Tuple[str, str]] = []

    search_order: List[str] = []
    if prefer_items:
        for it in prefer_items:
            if it in items:
                search_order.append(it)

    for it in items.keys():
        if it not in search_order:
            search_order.append(it)

    for it in search_order:
        for p in tokenize_paragraphs(items[it]):
            if keyword_hits(p, CLIMATE_KEYWORDS) > 0:
                out.append((it, p))

    return out


def compute_company_exposure(passages_df: pd.DataFrame) -> pd.DataFrame:
    """Company-level aggregation / exposure index."""
    if passages_df.empty:
        return pd.DataFrame(
            columns=[
                "cik",
                "company_name",
                "n_files",
                "n_passages",
                "keyword_hits_total",
                "avg_sentiment_compound",
                "share_physical",
                "share_transition",
                "share_both",
                "share_unspecified",
                "exposure_index",
            ]
        )

    # Binary indicators for shares
    df = passages_df.copy()
    df["is_physical"] = (df["risk_type"] == "PHYSICAL").astype(float)
    df["is_transition"] = (df["risk_type"] == "TRANSITION").astype(float)
    df["is_both"] = (df["risk_type"] == "BOTH").astype(float)
    df["is_unspecified"] = (df["risk_type"] == "UNSPECIFIED").astype(float)

    grp_cols = ["cik", "company_name"]
    g = df.groupby(grp_cols, dropna=False)

    summary = g.agg(
        n_files=("file", "nunique"),
        n_passages=("passage", "size"),
        keyword_hits_total=("keyword_hits", "sum"),
        avg_sentiment_compound=("sent_compound", "mean"),
        share_physical=("is_physical", "mean"),
        share_transition=("is_transition", "mean"),
        share_both=("is_both", "mean"),
        share_unspecified=("is_unspecified", "mean"),
    ).reset_index()

    # Exposure index:
    # log(1 + keyword hits) * (1 + passages/10)
    import numpy as np

    summary["exposure_index"] = np.log1p(summary["keyword_hits_total"]) * (1.0 + summary["n_passages"] / 10.0)

    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="Folder containing 10-K files (HTML/text).")
    ap.add_argument("--output-dir", required=True, help="Folder to save outputs.")
    ap.add_argument(
        "--write-items",
        action="store_true",
        help="If set, write extracted item text per filing to items.jsonl (can be large).",
    )
    ap.add_argument(
        "--prefer-items",
        default="1A,7,7A,1,2",
        help="Comma-separated list of items to prioritize when searching for climate passages.",
    )
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefer_items = [x.strip() for x in args.prefer_items.split(",") if x.strip()]

    vader = ensure_vader()
    if vader is None:
        print(
            "WARNING: VADER sentiment analyzer not available. Install nltk or vaderSentiment. "
            "Sentiment columns will be NaN."
        )

    # Robust file discovery (all files, extensionless included); ignore tiny artifacts
    files = sorted([p for p in in_dir.rglob("*") if p.is_file() and p.stat().st_size > 50_000])
    print(f"DEBUG: Found {len(files)} candidate filing files under {in_dir.resolve()}", flush=True)
    print("DEBUG: First 5 files:", [f.name for f in files[:5]], flush=True)
    if not files:
        raise FileNotFoundError(f"No filing-like files found in {in_dir.resolve()}")

    rows = []
    hit_files = 0

    items_fh = (out_dir / "items.jsonl").open("w", encoding="utf-8") if args.write_items else None

    import time
    t0 = time.time()

    for i, fp in enumerate(files, start=1):
        if i == 1 or i % 10 == 0:
            elapsed = time.time() - t0
            print(f"[{i}/{len(files)}] Processing: {fp.name} (elapsed {elapsed/60:.1f} min)", flush=True)

        res = process_filing(fp)

        if items_fh is not None:
            items_fh.write(
                json.dumps(
                    {
                        "file": res.file,
                        "company_name": res.company_name,
                        "cik": res.cik,
                        "period_of_report": res.period,
                        "filed_as_of_date": res.filed,
                        "items": res.items,
                    }
                )
                + "\n"
            )

        passages = extract_climate_passages(res.items, prefer_items=prefer_items)
        if passages:
            hit_files += 1

        for item_code, passage in passages:
            kh = keyword_hits(passage, CLIMATE_KEYWORDS)
            risk_type = classify_risk_type(passage)

            sent = {"neg": None, "neu": None, "pos": None, "compound": None}
            if vader is not None:
                try:
                    sent = vader.polarity_scores(passage)
                except Exception:
                    pass

            rows.append(
                {
                    "file": res.file,
                    "company_name": res.company_name,
                    "cik": res.cik,
                    "period_of_report": res.period,
                    "filed_as_of_date": res.filed,
                    "item": item_code,
                    "risk_type": risk_type,
                    "keyword_hits": kh,
                    "sent_neg": sent.get("neg"),
                    "sent_neu": sent.get("neu"),
                    "sent_pos": sent.get("pos"),
                    "sent_compound": sent.get("compound"),
                    "passage": passage,
                }
            )

    print(f"DEBUG: Filings with >=1 extracted climate passage: {hit_files} / {len(files)}", flush=True)

    if items_fh is not None:
        items_fh.close()

    passages_df = pd.DataFrame(rows)
    passages_df.to_csv(out_dir / "passages.csv", index=False)

    company_summary = compute_company_exposure(passages_df)
    company_summary.to_csv(out_dir / "company_summary.csv", index=False)

    print("Done.")
    print(f"Wrote: {(out_dir / 'passages.csv').resolve()}")
    print(f"Wrote: {(out_dir / 'company_summary.csv').resolve()}")
    if args.write_items:
        print(f"Wrote: {(out_dir / 'items.jsonl').resolve()}")
if __name__ == "__main__":
    main()