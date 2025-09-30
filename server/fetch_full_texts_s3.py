"""
Fetch plain-text full articles from the PMC OA public S3 bucket ONLY and save each
as an individual file under server/pubmed_data/full_texts (default). No PDF scraping,
no tar downloads, no OA file list usage, and by default no modification of the input CSV.

File naming convention:
    pmcid<digits>.txt   (e.g. PMC1234567 -> pmcid1234567.txt)

S3 object pattern (public, no creds):
    https://pmc-oa-opendata.s3.amazonaws.com/<subset>/txt/all/PMC#########.txt
Where <subset> ∈ {oa_comm, oa_noncomm, phe_timebound, author_manuscript}

If you ALSO want to update the CSV with a 'full_text' column like before, add --update-csv.

Usage examples (PowerShell):
    # Basic (fetch from oa_comm only; saves to default folder)
    python -m server.fetch_full_texts_s3 -v

    # Multiple subsets + higher parallelism
    python -m server.fetch_full_texts_s3 --s3-sets oa_comm,oa_noncomm,author_manuscript --parallel 16 -v

    # Limit first 40 PMCIDs for a dry run
    python -m server.fetch_full_texts_s3 --max 40 -v

    # Force re-download even if file already exists
    python -m server.fetch_full_texts_s3 --overwrite -v

    # Also update CSV with full_text column
    python -m server.fetch_full_texts_s3 --update-csv -v

    # Emit aws s3 cp script for misses
    python -m server.fetch_full_texts_s3 --emit-aws-script missed.ps1 -v
"""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import os
import sys
import time
from dataclasses import dataclass
from typing import List

import pandas as pd
import requests
from tqdm import tqdm
import logging

S3_BASE = "https://pmc-oa-opendata.s3.amazonaws.com"
VALID_SUBSETS = {"oa_comm", "oa_noncomm", "phe_timebound", "author_manuscript"}

logger = logging.getLogger(__name__)

@dataclass
class FetchResult:
    pmcid: str
    subset: str | None
    ok: bool
    text: str
    status: str  # 'ok' | 'missing' | 'short' | 'error' | 'skipped'
    length: int
    url: str | None
    error: str | None = None


def build_url(pmcid: str, subset: str) -> str:
    pmcid_norm = pmcid if pmcid.upper().startswith("PMC") else f"PMC{pmcid}"
    return f"{S3_BASE}/{subset}/txt/all/{pmcid_norm}.txt"


def fetch_one(pmcid: str, subsets: List[str], session: requests.Session, timeout: float, min_chars: int) -> FetchResult:
    for subset in subsets:
        url = build_url(pmcid, subset)
        try:
            r = session.get(url, timeout=timeout)
        except Exception as e:
            last_err = str(e)
            continue
        if r.status_code == 200:
            txt = r.text.strip()
            if len(txt) >= min_chars:
                return FetchResult(pmcid=pmcid, subset=subset, ok=True, text=txt, status="ok", length=len(txt), url=url)
            else:
                return FetchResult(pmcid=pmcid, subset=subset, ok=False, text=txt, status="short", length=len(txt), url=url)
    return FetchResult(pmcid=pmcid, subset=None, ok=False, text="", status="missing", length=0, url=None, error=None)


def main(args: argparse.Namespace) -> int:
    csv_path = args.csv
    if not os.path.exists(csv_path):
        logger.error(f"CSV not found: {csv_path}")
        return 2

    df = pd.read_csv(csv_path, dtype=str).fillna("")
    if "pmcid" not in df.columns:
        logger.error("Input CSV must have a 'pmcid' column (already mapped). Use your PMCID version.")
        return 2
    if args.update_csv and "full_text" not in df.columns:
        df["full_text"] = ""

    # Prepare work list
    rows = df.to_dict(orient="records")
    all_pmcids = [r.get("pmcid", "").strip() for r in rows if r.get("pmcid", "").strip()]
    unique_pmcids = []
    seen = set()
    for p in all_pmcids:
        if p and p not in seen:
            seen.add(p)
            unique_pmcids.append(p)

    if args.max:
        unique_pmcids = unique_pmcids[: args.max]

    if not unique_pmcids:
        logger.warning("No PMCIDs to process.")
        return 0

    subsets = [s.strip() for s in args.s3_sets.split(",") if s.strip()]
    invalid = [s for s in subsets if s not in VALID_SUBSETS]
    if invalid:
        logger.error(f"Invalid subset(s): {invalid}. Valid: {sorted(VALID_SUBSETS)}")
        return 2

    session = requests.Session()

    # Filter out already present unless overwrite
    if not args.overwrite:
        # Skip if file already exists in output dir and size >= threshold
        to_check_dir = args.save_txt_dir or args.default_txt_dir
        existing = 0
        remaining = []
        for p in unique_pmcids:
            pmcid_norm = p if p.upper().startswith("PMC") else f"PMC{p}"
            digits = pmcid_norm.upper().replace("PMC", "")
            out_name = f"pmcid{digits}.txt"
            out_path = os.path.join(to_check_dir, out_name)
            if os.path.exists(out_path) and os.path.getsize(out_path) > args.min_chars:
                existing += 1
                continue
            remaining.append(p)
        if existing:
            logger.info(f"Skipping {existing} PMCIDs already saved (size > min-chars). Use --overwrite to refetch.")
        unique_pmcids = remaining

    logger.info(f"Fetching texts for {len(unique_pmcids)} PMCIDs using subsets: {subsets} (parallel={args.parallel})")

    started = time.time()
    results: List[FetchResult] = []

    with cf.ThreadPoolExecutor(max_workers=args.parallel) as executor:
        fut_to_pmcid = {executor.submit(fetch_one, pmcid, subsets, session, args.timeout, args.min_chars): pmcid for pmcid in unique_pmcids}
        for fut in tqdm(cf.as_completed(fut_to_pmcid), total=len(fut_to_pmcid), desc="fetch"):
            pmcid = fut_to_pmcid[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = FetchResult(pmcid=pmcid, subset=None, ok=False, text="", status="error", length=0, url=None, error=str(e))
            results.append(res)

    # Apply results
    os.makedirs(args.out_dir, exist_ok=True)
    txt_dir = args.save_txt_dir or args.default_txt_dir
    os.makedirs(txt_dir, exist_ok=True)

    success = 0
    short = 0
    missing = 0
    errors = 0

    for res in results:
        if res.status == "ok":
            success += 1
            pmcid_norm = res.pmcid if res.pmcid.upper().startswith("PMC") else f"PMC{res.pmcid}"
            digits = pmcid_norm.upper().replace("PMC", "")
            out_file = os.path.join(txt_dir, f"pmcid{digits}.txt")
            with open(out_file, "w", encoding="utf-8") as fh:
                fh.write(res.text)
            if args.update_csv:
                mask = df["pmcid"].astype(str).str.strip().eq(res.pmcid)
                df.loc[mask, "full_text"] = res.text
        elif res.status == "short":
            short += 1
            if args.store_short:
                pmcid_norm = res.pmcid if res.pmcid.upper().startswith("PMC") else f"PMC{res.pmcid}"
                digits = pmcid_norm.upper().replace("PMC", "")
                out_file = os.path.join(txt_dir, f"pmcid{digits}.txt")
                with open(out_file, "w", encoding="utf-8") as fh:
                    fh.write(res.text)
                if args.update_csv:
                    mask = df["pmcid"].astype(str).str.strip().eq(res.pmcid)
                    df.loc[mask, "full_text"] = res.text
        elif res.status == "missing":
            missing += 1
        else:
            errors += 1

    duration = time.time() - started

    if args.update_csv:
        base = os.path.splitext(os.path.basename(csv_path))[0]
        out_csv = os.path.join(args.out_dir, base + "_with_fulltext_s3.csv")
        out_xlsx = os.path.join(args.out_dir, base + "_with_fulltext_s3.xlsx")
        df.to_csv(out_csv, index=False, encoding="utf-8")
        try:
            df.to_excel(out_xlsx, index=False)
        except Exception:
            logger.warning("Failed to write Excel output (xlsx). Continuing.")
        logger.info(f"Updated CSV written: {out_csv}")
    else:
        logger.info("CSV not modified (enable with --update-csv)")

    # Optionally generate AWS CLI script for misses
    if args.emit_aws_script:
        missed = [r for r in results if r.status in {"missing", "short"}]
        if missed:
            with open(args.emit_aws_script, "w", encoding="utf-8") as fh:
                fh.write("# PowerShell script: attempt aws s3 cp for missed PMCIDs\n")
                fh.write("# Requires: AWS CLI installed (no credentials needed).\n")
                for r in missed:
                    for subset in subsets:
                        url_path = build_url(r.pmcid, subset).replace(S3_BASE + "/", "")
                        pmcid_norm = r.pmcid if r.pmcid.upper().startswith("PMC") else f"PMC{r.pmcid}"
                        out_txt = os.path.join(args.save_txt_dir or ".", f"{pmcid_norm}.txt")
                        # Ensure Windows path escapes backslashes by using raw literal style
                        out_txt_norm = out_txt.replace("\\", "/")
                        fh.write(f"aws s3 cp s3://pmc-oa-opendata/{url_path} {out_txt_norm}\n")
            logger.info(f"Wrote AWS CLI miss script: {args.emit_aws_script}")
        else:
            logger.info("No misses; skipping aws script generation.")

    logger.info(f"Done in {duration:.1f}s | success={success} short={short} missing={missing} errors={errors}")
    logger.info(f"Text directory: {txt_dir}")
    return 0 if success > 0 else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fetch PMC OA plain texts from S3 only (no PDFs/tars)")
    default_csv = os.path.join(os.path.dirname(__file__), "pubmed_data", "pubmed_plastic_surgery_PMCID.csv")
    p.add_argument("--csv", default=default_csv, help="Input CSV containing pmcid column")
    p.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "pubmed_data"), help="Directory for augmented CSV/Excel")
    p.add_argument("--s3-sets", default="oa_comm", help="Comma-separated S3 subsets order to try")
    p.add_argument("--parallel", type=int, default=8, help="Thread pool size")
    p.add_argument("--timeout", type=float, default=25, help="Per-request timeout (seconds)")
    p.add_argument("--max", type=int, default=None, help="Optional limit of PMCIDs for testing")
    p.add_argument("--min-chars", type=int, default=50, help="Minimum characters for success")
    p.add_argument("--overwrite", action="store_true", help="Re-fetch even if full_text already populated (length >= min-chars)")
    default_txt_dir = os.path.join(os.path.dirname(__file__), "pubmed_data", "full_texts")
    p.add_argument("--save-txt-dir", help="Optional override directory for text outputs (default=pubmed_data/full_texts)")
    p.add_argument("--update-csv", action="store_true", help="Also add/overwrite full_text column in CSV and write *_with_fulltext_s3 outputs")
    p.add_argument("--emit-aws-script", help="Write a PowerShell script with aws s3 cp commands for misses")
    p.add_argument("--store-short", action="store_true", help="Store texts even if below min-chars threshold (status=short)")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = p.parse_args()

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                            format="%(asctime)s %(levelname)s %(message)s",
                            datefmt="%H:%M:%S")
    else:
        logging.getLogger().setLevel(logging.DEBUG if args.verbose else logging.INFO)

    args.default_txt_dir = default_txt_dir
    rc = main(args)
    sys.exit(rc)
