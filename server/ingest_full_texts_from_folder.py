"""
Ingest previously downloaded PMC OA plain-text files into a metadata CSV.

Assumes you already used fetch_full_texts_s3.py to populate a folder of files named:
  pmcid<digits>.txt   (preferred) or PMC<digits>.txt

This script:
  1. Reads the input CSV (must contain 'pmcid' column).
  2. Scans the provided text directory for matching files.
  3. Loads file contents and inserts into/creates the 'full_text' column.
  4. Writes out <basename>_with_fulltext_ingested.csv and .xlsx
  5. Optionally enforces min length; shorter texts can be skipped unless --include-short.

Usage (PowerShell examples):
  # Basic (default locations)
  python -m server.ingest_full_texts_from_folder \
      --csv server/pubmed_data/pubmed_plastic_surgery_PMCID.csv \
      --text-dir server/pubmed_data/full_texts -v

  # Skip very short texts (<120 chars) unless --include-short
  python -m server.ingest_full_texts_from_folder --min-chars 120 -v

  # Force overwrite existing populated full_text entries
  python -m server.ingest_full_texts_from_folder --overwrite -v

  # Only process first 200 rows (debug)
  python -m server.ingest_full_texts_from_folder --max 200 -v
"""
from __future__ import annotations
import argparse
import logging
import os
import sys
from typing import Dict, Tuple, List
import re
import requests
from dotenv import load_dotenv

load_dotenv()


import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def normalize_pmcid(raw: str) -> str:
    if not raw:
        return ""
    raw = raw.strip().upper()
    if raw.startswith("PMC"):
        return raw
    return f"PMC{raw}"


def derive_pmcid_from_filename(filename: str) -> str | None:
    """Handle pmcid<digits>.txt or PMC<digits>.txt variants."""
    low = filename.lower()
    if not low.endswith('.txt'):
        return None
    stem = low[:-4]  # remove .txt
    # pmcid1234567
    if stem.startswith('pmcid'):
        digits = stem[len('pmcid'):]
        if digits.isdigit():
            return f"PMC{digits}"
    # PMC1234567
    if stem.startswith('pmc'):
        digits = stem[len('pmc'):]
        if digits.isdigit():
            return f"PMC{digits}"
    return None


ID_CONVERTER_API = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"


def _chunked(lst: List[str], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def map_pmids_to_pmcids(pmids: List[str], chunk_size: int = 200, timeout: int = 40) -> Dict[str, str]:
    """Use NCBI ID Converter to map PMID -> PMCID."""
    session = requests.Session()
    out: Dict[str, str] = {}
    clean = [p.strip() for p in pmids if p and str(p).strip().isdigit()]
    if not clean:
        return out
    for chunk in _chunked(clean, chunk_size):
        try:
            r = session.get(ID_CONVERTER_API, params={"format": "json", "ids": ",".join(chunk)}, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            for rec in data.get("records", []):
                pmid = str(rec.get("pmid", "")).strip()
                pmcid = str(rec.get("pmcid", "")).strip()
                if pmid and pmcid:
                    out[pmid] = pmcid
        except Exception as e:
            logger.warning(f"PMID mapping chunk failed: {e}")
            continue
    return out


def infer_pmcid_from_url_field(val: str) -> List[str]:
    if not val or not isinstance(val, str):
        return []
    # collect PMC digits from patterns like /PMC1234567/
    found = re.findall(r"PMC(\d+)", val, flags=re.IGNORECASE)
    return [f"PMC{d}" for d in found]


def sanitize_text(s: str, keep_newlines: bool) -> str:
    if keep_newlines:
        return s
    # collapse consecutive whitespace/newlines into single space
    return re.sub(r"\s+", " ", s).strip()


def ingest(csv_path: str, text_dir: str, out_dir: str, min_chars: int, include_short: bool, overwrite: bool, max_rows: int | None,
           autofill_pmcid: bool, infer_from_links: bool, sanitize: bool, keep_newlines: bool, append_missing: bool) -> Tuple[str, str]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not os.path.isdir(text_dir):
        raise FileNotFoundError(f"Text directory not found: {text_dir}")
        

    logger.info(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    if 'pmcid' not in df.columns:
        if autofill_pmcid or infer_from_links:
            # create empty column we will try to populate
            df['pmcid'] = ''
        else:
            raise ValueError("CSV must contain a 'pmcid' column (or use --autofill-pmcid-from-idconv / --infer-pmcid-from-links)")

    # Attempt inference from link columns first (may fill many at once)
    inferred_link = 0
    if infer_from_links:
        link_cols = [c for c in ['candidate_pdf_urls', 'preferred_pdf_url'] if c in df.columns]
        if link_cols:
            for i, row in df.iterrows():
                if str(row.get('pmcid', '')).strip():
                    continue
                for lc in link_cols:
                    vals = infer_pmcid_from_url_field(str(row.get(lc, '')))
                    if vals:
                        # take first
                        df.at[i, 'pmcid'] = vals[0]
                        inferred_link += 1
                        break
            logger.info(f"Inferred {inferred_link} PMCIDs from link columns")
        else:
            logger.info("No link columns present for inference")

    # Autofill via PMID mapping for remaining blanks
    mapped = 0
    if autofill_pmcid:
        if 'pmid' in df.columns:
            remaining_pmids = [str(p) for p, pmc in zip(df['pmid'], df['pmcid']) if not str(pmc).strip()]
            mapping = map_pmids_to_pmcids(remaining_pmids)
            for i, row in df.iterrows():
                if not str(row.get('pmcid', '')).strip():
                    pmid_val = str(row.get('pmid', '')).strip()
                    pmcid_val = mapping.get(pmid_val)
                    if pmcid_val:
                        df.at[i, 'pmcid'] = pmcid_val
                        mapped += 1
            logger.info(f"Mapped {mapped} PMCIDs via ID converter API")
        else:
            logger.warning("Cannot autofill PMCIDs: no 'pmid' column present")
    if 'full_text' not in df.columns:
        df['full_text'] = ''

    # Build map pmcid -> row mask for faster assignment
    df['pmcid_norm'] = df['pmcid'].astype(str).map(normalize_pmcid)
    pmcid_index: Dict[str, pd.Series] = {}
    for pmcid_val, grp in df.groupby('pmcid_norm'):
        pmcid_index[pmcid_val] = grp.index

    files = [f for f in os.listdir(text_dir) if f.lower().endswith('.txt')]
    logger.info(f"Scanning {len(files)} text files in {text_dir}")

    # Optional row limit
    if max_rows is not None and max_rows < len(df):
        df = df.iloc[:max_rows].copy()
        # rebuild index for truncated df
        df['pmcid_norm'] = df['pmcid'].astype(str).map(normalize_pmcid)
        pmcid_index = {pmc: df.index[df['pmcid_norm'] == pmc] for pmc in df['pmcid_norm'].unique()}

    ingested = 0
    skipped_short = 0
    skipped_existing = 0
    missing_match = 0
    appended = 0

    for fname in tqdm(files, desc='ingest'):
        pmcid = derive_pmcid_from_filename(fname)
        if not pmcid:
            continue
        if pmcid not in pmcid_index:
            if append_missing:
                # create new row with pmcid only
                new_row = {col: '' for col in df.columns}
                new_row['pmcid'] = pmcid
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                # update index mapping
                df['pmcid_norm'] = df['pmcid'].astype(str).map(normalize_pmcid)
                pmcid_index[pmcid] = df.index[df['pmcid_norm'] == pmcid]
                appended += 1
                idxs = pmcid_index[pmcid]
            else:
                missing_match += 1
                continue
        idxs = pmcid_index[pmcid]
        # skip if already populated and not overwriting
        if not overwrite:
            if any(df.loc[idxs, 'full_text'].str.len() >= min_chars):
                skipped_existing += 1
                continue
        path = os.path.join(text_dir, fname)
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
                text = fh.read().strip()
        except Exception as e:
            logger.warning(f"Failed to read {fname}: {e}")
            continue
        if len(text) < min_chars and not include_short:
            skipped_short += 1
            continue
        if sanitize:
            text_out = sanitize_text(text, keep_newlines)
        else:
            text_out = text if keep_newlines else text
        df.loc[idxs, 'full_text'] = text_out
        ingested += 1

    # Drop helper column
    df = df.drop(columns=['pmcid_norm'])

    base = os.path.splitext(os.path.basename(csv_path))[0]
    out_csv = os.path.join(out_dir, base + '_with_fulltext_ingested.csv')
    out_xlsx = os.path.join(out_dir, base + '_with_fulltext_ingested.xlsx')

    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding='utf-8')
    try:
        df.to_excel(out_xlsx, index=False)
    except Exception:
        logger.warning('Failed to write Excel output.')

    logger.info(f"Ingestion complete: ingested={ingested} appended_new_rows={appended} skipped_existing={skipped_existing} skipped_short={skipped_short} missing_match_files={missing_match}")
    logger.info(f"Output CSV: {out_csv}")
    return out_csv, out_xlsx


def main():
    p = argparse.ArgumentParser(description='Ingest folder full_text .txt files into CSV full_text column')
    default_csv = os.path.join(os.path.dirname(__file__), 'pubmed_data', 'pubmed_plastic_surgery_PMCID.csv')
    default_txt = os.path.join(os.path.dirname(__file__), 'pubmed_data', 'full_texts')
    p.add_argument('--csv', default=default_csv, help='Input CSV with pmcid column')
    p.add_argument('--text-dir', default=default_txt, help='Directory containing pmcid*.txt files')
    p.add_argument('--out-dir', default=os.path.join(os.path.dirname(__file__), 'pubmed_data'), help='Directory to place updated CSV/XLSX')
    p.add_argument('--min-chars', type=int, default=50, help='Minimum length to accept a text (else skipped unless --include-short)')
    p.add_argument('--include-short', action='store_true', help='Store texts even if below min-chars')
    p.add_argument('--overwrite', action='store_true', help='Overwrite existing full_text values (length >= min-chars)')
    p.add_argument('--max', type=int, default=None, help='Limit processing to first N rows of CSV (debug)')
    p.add_argument('--autofill-pmcid-from-idconv', action='store_true', help='Call NCBI ID converter to fill missing pmcid values from pmid')
    p.add_argument('--infer-pmcid-from-links', action='store_true', help='Extract PMCIDs from candidate_pdf_urls/preferred_pdf_url columns when pmcid blank')
    p.add_argument('--sanitize-newlines', action='store_true', help='Collapse whitespace/newlines in full_text before writing (default ON unless --keep-newlines)')
    p.add_argument('--keep-newlines', action='store_true', help='Preserve original newlines (disables sanitize)')
    p.add_argument('--append-missing', action='store_true', help='Append new rows for text files whose PMCIDs are not present in CSV')
    p.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    args = p.parse_args()

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                            format='%(asctime)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S')
    else:
        logging.getLogger().setLevel(logging.DEBUG if args.verbose else logging.INFO)

    try:
        ingest(
            csv_path=args.csv,
            text_dir=args.text_dir,
            out_dir=args.out_dir,
            min_chars=args.min_chars,
            include_short=args.include_short,
            overwrite=args.overwrite,
            max_rows=args.max,
            autofill_pmcid=args.autofill_pmcid_from_idconv,
            infer_from_links=args.infer_pmcid_from_links,
            sanitize= (not args.keep_newlines) if not args.sanitize_newlines else True and not args.keep_newlines,
            keep_newlines=args.keep_newlines,
            append_missing=args.append_missing
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
