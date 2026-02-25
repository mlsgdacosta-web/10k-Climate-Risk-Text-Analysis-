# 10k-Climate-Risk-Text-Analysis-
NLP-based analysis of climate risk disclosures in SEC 10-K filing, including sentiment analysis and firm-level exposure measurement.

# What this script does:
1) Reads a folder of EDGAR 10-K filings (.txt).
2) Converts each filing to plain text (strip tags / normalize whitespace) to make downstream parsing more reliable.
3) Attempts to split filings into major 10-K sections/items (e.g., Item 1A, Item 7) so that extracted climate passages can be linked to the part of the filing (risk factors vs. MD&A, etc.). If item splitting is imperfect, the code still searches broadly in the available text.
4) Scans the text for climate-related passages using a keyword-based filter and records each matching passage along with metadata (file, CIK, company name, filing dates, item label).
5) Classifies each passage into a “risk type”:
       - PHYSICAL: language about physical climate impacts (storms, wildfire, flooding, heat, sea level, etc.)
       - TRANSITION: language about transition/regulatory/market changes (policy, carbon taxes, disclosure rules, net-zero commitments, etc.)
       - BOTH: signals of both physical + transition risk in the same passage
       - UNSPECIFIED: climate language without enough context to classify reliably
6) Computes sentiment using VADER (lexicon-based sentiment). This produces a compound sentiment score per passage.

# Choosing the climate keywords:
 - The keyword list is intentionally broad and includes common climate terms (e.g. "climate change", "emissions", "carbon", "greenhouse gas", "net zero") plus related corporate disclosure terms (e.g., "ESG", "sustainability") so we capture typical 10-K wording across many industries.
 - The goal is to identify candidate climate disclosures reliably across a large set of filings, not to perfectly isolate only climate-risk text. False positives could be inspected and filtered later if needed.

# What the outputs mean:
 1) outputs/passages.csv (passage-level dataset)
    Each row is one extracted passage containing ≥1 climate keyword.
    Key columns:
      - file: source filing filename
      - cik, company_name: identifiers pulled from the filing header
      - period_of_report, filed_as_of_date: reporting/filing dates when available
      - item: section label (e.g., "1A", "7", or "FULL_TEXT" if not cleanly split)
      - risk_type: PHYSICAL / TRANSITION / BOTH / UNSPECIFIED (rule-based)
      - keyword_hits: count of climate keywords matched in the passage
      - sent_*: VADER sentiment components; sent_compound is the main summary score
      - passage: the extracted text snippet

 2) outputs/company_summary.csv (company-level exposure summary)
    Aggregates passages.csv by company (CIK), producing:
      - n_files: number of unique filings for the company that produced ≥1 passage
      - n_passages: number of climate passages extracted for the company
      - keyword_hits_total: total keyword matches across all passages
      - avg_sentiment_compound: average sentiment across passages
      - share_physical / share_transition / share_both / share_unspecified: composition of passage types
      - exposure_index: a simple exposure proxy (constructed from passage volume / keyword intensity). Higher values indicate more climate-related discussion in the filing text, not necessarily higher real-world risk.

# Why this design:
 - Keyword screening is transparent and fast for large-scale text (hundreds of 10-Ks).
 - Splitting into 10-K items improves interpretability (risk vs. discussion sections).
 - A simple rule-based risk classifier is easy to audit and adjust.
 - VADER sentiment is lightweight and reproducible. It provides a directional tone measure for disclosures, though it is not a domain-specific finance sentiment model.

# Practical validation / checks I used:
 - Confirm the script finds all filings in the directory and reports progress.
 - Check outputs contain many unique files and many unique CIKs (not collapsing to 1).
 - Sample a handful of passages manually to confirm they appear in the raw filings.
 - Compare company_summary counts to passage-level counts (aggregation consistency).

# Limitations:
 - Keyword methods may capture non-risk climate references (e.g., generic ESG language).
 - Item splitting can be imperfect due to EDGAR formatting differences across firms.
 - Sentiment can be noisy on legal text; results are best used comparatively.
