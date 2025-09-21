"""
Download PDFs for PMC articles (PMCID-based; will auto-convert from PMID if needed),
extract text, add `full_text` to a CSV/Excel produced by generate_pubmed_data.py,
and delete downloaded tar.gz files after extraction.

Usage:
    # CSV can contain either a 'pmcid' column or a 'pmid' column
    python -m server.extract_full_texts \
        --csv server/pubmed_data/pubmed_plastic_surgery_PMID.csv \
        --out-dir server/pubmed_data

Dependencies:
    pip install requests pandas PyPDF2 beautifulsoup4 lxml tqdm
"""
from __future__ import annotations
import argparse
import io
import os
import re
import tarfile
import tempfile
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from tqdm import tqdm
import logging

OA_FILE_LIST_URL = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_file_list.csv"
OA_BULK_BASE = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk"
ID_CONVERTER_API = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"

logger = logging.getLogger(__name__)


def _extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for p in reader.pages:
            try:
                text = p.extract_text() or ""
            except Exception:
                text = ""
            pages.append(text)
        return "\n\n".join(pages).strip()
    except Exception:
        return ""


def _try_direct_pdf_download(pmcid: str, session: requests.Session) -> Optional[bytes]:
    # normalize pmcid to form 'PMC12345' or 'PMC12345' input accepted
    pmcid_norm = pmcid if pmcid.upper().startswith("PMC") else f"PMC{pmcid}"
    candidates = [
        f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid_norm}/pdf",
        f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid_norm}/pdf/",
        f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid_norm}/pdf/{pmcid_norm}.pdf",
        f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid_norm}/pdf/{pmcid_norm}.pdf?report=object"
    ]
    # also try fetching the article page and parsing for a .pdf link
    try:
        page = session.get(f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid_norm}/", timeout=30)
        if page.ok:
            soup = BeautifulSoup(page.content, "lxml")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.lower().endswith(".pdf"):
                    if href.startswith("http"):
                        candidates.insert(0, href)
                    else:
                        candidates.insert(0, requests.compat.urljoin("https://www.ncbi.nlm.nih.gov", href))
    except Exception:
        pass

    for url in candidates:
        try:
            r = session.get(url, timeout=60)
            if r.status_code == 200 and r.headers.get("content-type", "").lower().startswith("application/pdf"):
                return r.content
            # some endpoints redirect to PDF with different content-type; try to accept octet-stream
            if r.status_code == 200 and len(r.content) > 1000:
                return r.content
        except Exception:
            continue
    return None


def _chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _map_pmids_to_pmcids(pmids: list[str], session: requests.Session, chunk_size: int = 200) -> dict[str, str]:
    """Use NCBI's ID Converter API to map PMID -> PMCID.
    Returns a dict of pmid (string) -> pmcid (string). Missing mappings won't be included.
    Docs: https://www.ncbi.nlm.nih.gov/pmc/tools/id-converter-api/
    """
    result: dict[str, str] = {}
    clean_pmids = [str(p).strip() for p in pmids if str(p).strip()]
    if not clean_pmids:
        return result
    for chunk in _chunked(clean_pmids, chunk_size):
        try:
            r = session.get(ID_CONVERTER_API, params={"format": "json", "ids": ",".join(chunk)}, timeout=60)
            r.raise_for_status()
            data = r.json()
            for rec in data.get("records", []):
                pmid = str(rec.get("pmid", "")).strip()
                pmcid = str(rec.get("pmcid", "")).strip()
                if pmid and pmcid:
                    result[pmid] = pmcid
        except Exception:
            # best-effort mapping; skip on error for this chunk
            continue
    return result


def _resolve_input_csv(csv_path: str) -> str:
    """If the given csv_path doesn't exist, try sensible fallbacks.
    Prefer swapping PMCID/PMID in the filename and checking the repo's pubmed_data.
    Returns a valid path or raises FileNotFoundError with guidance.
    """
    if os.path.exists(csv_path):
        return csv_path

    # Determine candidate directory to search
    given_dir = os.path.dirname(csv_path)
    default_dir = os.path.join(os.path.dirname(__file__), "pubmed_data")
    search_dir = given_dir if given_dir and os.path.isdir(given_dir) else default_dir

    base = os.path.basename(csv_path)
    candidates = []
    # Try swapping PMCID <-> PMID in the base filename
    if base:
        if "PMCID" in base:
            candidates.append(base.replace("PMCID", "PMID"))
        if "PMID" in base:
            candidates.append(base.replace("PMID", "PMCID"))

    # Known filenames in this repo
    candidates.extend([
        "pubmed_plastic_surgery_PMID.csv",
        "pubmed_plastic_surgery_PMCID.csv",
    ])

    for cand in candidates:
        alt = os.path.join(search_dir, cand)
        if os.path.exists(alt):
            logger.info(f"Input CSV not found at '{csv_path}'. Using '{alt}' instead.")
            return alt

    # Nothing found: build a helpful message listing available CSVs
    try:
        files = [f for f in os.listdir(search_dir) if f.lower().endswith(".csv")]
    except Exception:
        files = []
    available = ", ".join(files) if files else "<none>"
    raise FileNotFoundError(
        "Input CSV not found: "
        + f"'{csv_path}'. Checked directory '{search_dir}'. "
        + f"Available CSVs here: {available}. "
        + "If your input file lives elsewhere, pass --csv <path-to-file>."
    )


def _download_oa_file_list(session: requests.Session, cache_path: str) -> str:
    if os.path.exists(cache_path):
        return cache_path
    r = session.get(OA_FILE_LIST_URL, timeout=60)
    r.raise_for_status()
    with open(cache_path, "wb") as f:
        f.write(r.content)
    return cache_path


def _find_tar_for_pmcid(pmcid: str, oa_list_text: str) -> Optional[str]:
    # pmcid may be with or without 'PMC'
    pmcid_pattern = pmcid if pmcid.upper().startswith("PMC") else f"PMC{pmcid}"
    # find lines containing the pmcid
    for line in oa_list_text.splitlines():
        if pmcid_pattern in line:
            m = re.search(r"(oa_file_[0-9]+\.tar\.gz)", line)
            if m:
                return m.group(1)
    return None


def _extract_pdf_from_tar_for_pmcid(tar_path: str, pmcid: str) -> Optional[bytes]:
    pmcid_digits = pmcid.upper().replace("PMC", "")
    with tarfile.open(tar_path, "r:gz") as tf:
        # try to find members that include the pmcid digits and end with .pdf
        for member in tf.getmembers():
            name = member.name.lower()
            if not name.endswith(".pdf"):
                continue
            if pmcid_digits in name:
                f = tf.extractfile(member)
                if f:
                    return f.read()
    return None


def main(csv_path: str, out_dir: str, max_items: Optional[int] = None, keep_tar: bool = False):
    session = requests.Session()
    csv_path = _resolve_input_csv(csv_path)
    logger.info(f"Reading input CSV: {csv_path}")
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    logger.info(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
    if "full_text" not in df.columns:
        df["full_text"] = ""

    # Ensure we have a PMCID column; if only PMID provided, try to map
    if ("pmcid" not in df.columns) or df["pmcid"].fillna("").eq("").all():
        if "pmid" in df.columns:
            try:
                pmid_list = df["pmid"].astype(str).tolist()
                logger.info("Mapping PMID to PMCID via NCBI ID Converter API...")
                mapping = _map_pmids_to_pmcids(pmid_list, session)
                df["pmcid"] = df["pmid"].astype(str).map(lambda x: mapping.get(str(x).strip(), ""))
                mapped = int(df["pmcid"].astype(str).str.len().gt(0).sum())
                logger.info(f"Mapped {mapped} of {len(df)} PMIDs to PMCIDs")
            except Exception:
                # if mapping fails, create empty column and proceed (will skip those rows)
                df["pmcid"] = df.get("pmcid", "")
                logger.exception("PMID→PMCID mapping failed; continuing without mappings")
        else:
            raise ValueError("CSV must contain either a 'pmcid' or a 'pmid' column.")

    rows = df.to_dict(orient="records")
    to_process = [r for r in rows if r.get("pmcid")]
    if max_items:
        to_process = to_process[:max_items]
    logger.info(f"PMCID rows to process: {len(to_process)}" + (f" (capped to max={max_items})" if max_items else ""))

    # If nothing to do, skip heavy network ops
    oa_text = ""
    if to_process:
        cache_dir = os.path.join(out_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        oa_list_path = os.path.join(cache_dir, "oa_file_list.csv")

        # download file list once (may be large)
        try:
            logger.info("Preparing PMC OA file list (this can be large; cached after first time)...")
            _download_oa_file_list(session, oa_list_path)
            with open(oa_list_path, "r", encoding="utf-8", errors="ignore") as f:
                oa_text = f.read()
        except Exception:
            oa_text = ""
            logger.warning("Could not load PMC OA file list; tar fallback may be limited")

    # group pmcids by tarname to minimize downloads
    tar_needed = {}
    pmcid_to_tar = {}

    for r in to_process:
        pmcid = r.get("pmcid", "").strip()
        if not pmcid:
            continue
        tarname = _find_tar_for_pmcid(pmcid, oa_text) if oa_text else None
        if tarname:
            pmcid_to_tar[pmcid] = tarname
            tar_needed.setdefault(tarname, []).append(pmcid)
    if pmcid_to_tar:
        logger.info(f"Identified tar archives for {len(pmcid_to_tar)} PMCIDs across {len(tar_needed)} archives")

    # process rows; try direct download first, then tar extraction
    for idx, r in enumerate(tqdm(to_process, desc="PMCID"), start=1):
        pmcid = r.get("pmcid", "").strip()
        if not pmcid:
            continue
        # skip if already have full_text
        if r.get("full_text"):
            continue

        text = ""
        # try direct PDF download
        logger.info(f"[{idx}/{len(to_process)}] {pmcid}: trying direct PDF download...")
        pdf_bytes = _try_direct_pdf_download(pmcid, session)
        if pdf_bytes:
            logger.info(f"{pmcid}: direct PDF download succeeded")
            text = _extract_text_from_pdf_bytes(pdf_bytes)
        else:
            logger.info(f"{pmcid}: direct PDF download not found")

        # if direct failed, try tar approach
        if not text and pmcid in pmcid_to_tar:
            tarname = pmcid_to_tar[pmcid]
            tar_url = f"{OA_BULK_BASE}/{tarname}"
            local_tar = os.path.join(cache_dir, tarname)
            try:
                logger.info(f"{pmcid}: attempting tar fallback via {tarname}")
                if not os.path.exists(local_tar):
                    # stream download to file
                    logger.info(f"{pmcid}: downloading tar archive {tar_url} ...")
                    with session.get(tar_url, stream=True, timeout=120) as rtar:
                        rtar.raise_for_status()
                        with open(local_tar, "wb") as fh:
                            for chunk in rtar.iter_content(1024 * 1024):
                                if chunk:
                                    fh.write(chunk)
                    logger.info(f"{pmcid}: tar archive downloaded to {local_tar}")
                pdf_bytes = _extract_pdf_from_tar_for_pmcid(local_tar, pmcid)
                if pdf_bytes:
                    logger.info(f"{pmcid}: PDF extracted from tar")
                    text = _extract_text_from_pdf_bytes(pdf_bytes)
            except Exception:
                text = ""
                logger.exception(f"{pmcid}: tar fallback failed")
            finally:
                if not keep_tar and os.path.exists(local_tar):
                    try:
                        os.remove(local_tar)
                        logger.info(f"{pmcid}: removed cached tar {local_tar}")
                    except Exception:
                        pass
        elif not text:
            logger.info(f"{pmcid}: no tar fallback available; skipping")

        # write back extracted text to dataframe
        if text:
            # find matching row(s) by pmid or pmcid and set full_text
            mask = (df["pmcid"].fillna("") == r.get("pmcid", ""))
            df.loc[mask, "full_text"] = text
            logger.info(f"{pmcid}: text extracted and saved to dataframe")
        else:
            # optionally leave as empty if not found
            logger.info(f"{pmcid}: full text not found")

    # save outputs
    out_csv = os.path.join(out_dir, os.path.basename(csv_path).replace(".csv", "_with_fulltext.csv"))
    out_xlsx = os.path.join(out_dir, os.path.basename(csv_path).replace(".csv", "_with_fulltext.xlsx"))
    logger.info(f"Writing updated CSV to: {out_csv}")
    df.to_csv(out_csv, index=False, encoding="utf-8")
    try:
        logger.info(f"Writing updated Excel to: {out_xlsx}")
        df.to_excel(out_xlsx, index=False)
    except Exception:
        logger.warning("Failed to write Excel output; continuing")

    logger.info(f"Saved updated CSV: {out_csv}")
    logger.info(f"Saved updated Excel: {out_xlsx}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=os.path.join(os.path.dirname(__file__), "pubmed_data", "pubmed_plastic_surgery_PMID.csv"))
    p.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "pubmed_data"))
    p.add_argument("--max", type=int, default=None, help="max number of pmcids to process (for testing)")
    p.add_argument("--keep-tar", action="store_true", help="do not delete downloaded tar.gz files")
    p.add_argument("-v", "--verbose", action="store_true", help="enable verbose/DEBUG logging")
    args = p.parse_args()
    # configure logging
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                            format="%(asctime)s %(levelname)s %(message)s",
                            datefmt="%H:%M:%S")
    else:
        logging.getLogger().setLevel(logging.DEBUG if args.verbose else logging.INFO)
    main(csv_path=args.csv, out_dir=args.out_dir, max_items=args.max, keep_tar=args.keep_tar)