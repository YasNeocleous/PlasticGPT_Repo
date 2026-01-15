import requests
import xml.etree.ElementTree as ET
import os
import re
import pandas as pd

OA_FILE_LIST_URL = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_file_list.csv"
OA_BULK_BASE = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk"
OA_BASE = "https://ftp.ncbi.nlm.nih.gov/pub/pmc"

def fetch_pubmed_articles(query, retstart=0, count=200):
    search_url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        "?db=pubmed&term={query}+AND+hasabstract[text]&retmax={count}&retstart={retstart}&retmode=json"
    ).format(query=query.replace(' ', '+'), count=count, retstart=retstart)
    search_resp = requests.get(search_url)
    id_list = search_resp.json()['esearchresult']['idlist']

    if not id_list:
        return []

    fetch_url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        "?db=pubmed&id={ids}&retmode=xml"
    ).format(ids=','.join(id_list))
    fetch_resp = requests.get(fetch_url)
    root = ET.fromstring(fetch_resp.content)

    articles = []
    for article in root.findall('.//PubmedArticle'):
        pmid = article.findtext('.//PMID', default='N/A')
        title = article.findtext('.//ArticleTitle', default='N/A')
        authors = []
        for author in article.findall('.//Author'):
            last = author.findtext('LastName')
            first = author.findtext('ForeName')
            if last and first:
                authors.append(f"{first} {last}")
        # Extract PMC ID and DOI if present in ArticleId elements
        pmcid = ''
        doi = ''
        for aid in article.findall('.//ArticleId'):
            idtype = aid.get('IdType')
            if idtype == 'pmc' and aid.text:
                pmcid = aid.text
            if idtype == 'doi' and aid.text:
                doi = aid.text

        # Extract abstract text (may have multiple AbstractText elements)
        abstract_parts = [elem.text for elem in article.findall('.//AbstractText') if elem.text]
        abstract = '\n'.join(abstract_parts) if abstract_parts else ''

        articles.append({
            'pmid': pmid,
            'pmcid': pmcid,
            'doi': doi,
            'title': title,
            'authors': ', '.join(authors),
            'abstract': abstract
        })

    return articles

if __name__ == "__main__":
    query = "plastic surgery OR (plastic and reconstructive surgery) OR facial plastic surgery OR aesthetic surgery OR cosmetic surgery"
    all_results = []

    for i in range(5):
        print(f"Fetching batch {i+1}...")
        batch = fetch_pubmed_articles(query, retstart=i*200, count=200)
        all_results.extend(batch)
        # Remove duplicates by PMID
        df = pd.DataFrame(all_results)
        df = df.drop_duplicates(subset="pmid", keep="first")
        all_results = df.to_dict(orient="records")
        print(f"Rows after batch {i+1}: {len(df)}")

    output_dir = os.path.join(os.path.dirname(__file__), "pubmed_data")
    os.makedirs(output_dir, exist_ok=True)

    # Build candidate PDF URLs and OA tar URLs when possible
    def _candidate_pdf_urls_for_pmcid(pmcid: str):
        if not pmcid:
            return []
        pmcid_norm = pmcid if pmcid.upper().startswith("PMC") else f"PMC{pmcid}"
        return [
            f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid_norm}/pdf",
            f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid_norm}/pdf/{pmcid_norm}.pdf",
        ]

    def _download_oa_file_list(cache_path: str) -> str:
        if os.path.exists(cache_path):
            return cache_path
        resp = requests.get(OA_FILE_LIST_URL, timeout=60)
        resp.raise_for_status()
        with open(cache_path, "wb") as f:
            f.write(resp.content)
        return cache_path

    def _find_tar_for_pmcid(pmcid: str, oa_list_text: str):
        if not pmcid or not oa_list_text:
            return None
        pmcid_pattern = pmcid if pmcid.upper().startswith("PMC") else f"PMC{pmcid}"
        for line in oa_list_text.splitlines():
            if pmcid_pattern in line:
                m = re.search(r"(oa_file_[0-9]+\.tar\.gz)", line)
                if m:
                    return m.group(1)
        return None

    def _find_oa_pdf_for_pmcid(pmcid: str, oa_list_text: str):
        if not pmcid or not oa_list_text:
            return None
        pmcid_pattern = pmcid if pmcid.upper().startswith("PMC") else f"PMC{pmcid}"
        for line in oa_list_text.splitlines():
            if pmcid_pattern in line and 'oa_pdf/' in line:
                m = re.search(r"(oa_pdf/[^,\s]+?\.pdf)", line)
                if m:
                    return m.group(1)
        return None

    def _find_oa_bulk_for_pmcid(pmcid: str, oa_list_text: str):
        if not pmcid or not oa_list_text:
            return None
        pmcid_pattern = pmcid if pmcid.upper().startswith("PMC") else f"PMC{pmcid}"
        for line in oa_list_text.splitlines():
            if pmcid_pattern in line and 'oa_bulk/' in line:
                m = re.search(r"(oa_bulk/[^,\s]+?\.tar\.gz)", line)
                if m:
                    return m.group(1)
        return None

    # Prepare OA manifest cache
    cache_dir = os.path.join(output_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    oa_list_path = os.path.join(cache_dir, "oa_file_list.csv")
    try:
        _download_oa_file_list(oa_list_path)
        with open(oa_list_path, "r", encoding="utf-8", errors="ignore") as f:
            oa_text = f.read()
    except Exception:
        oa_text = ""

    csv_path = os.path.join(output_dir, "pubmed_plastic_surgery_PMCID.csv")
    excel_path = os.path.join(output_dir, "pubmed_plastic_surgery_PMCID.xlsx")

    # Enrich with paths
    if not df.empty:
        # candidate direct PDF URLs
        df["candidate_pdf_urls"] = df["pmcid"].astype(str).apply(
            lambda x: ";".join(_candidate_pdf_urls_for_pmcid(x)) if x else ""
        )
        df["preferred_pdf_url"] = df["candidate_pdf_urls"].apply(lambda s: s.split(";")[0] if s else "")
        # OA direct pdf and tar (if present in OA manifest)
        if oa_text:
            df["oa_pdf_relpath"] = df["pmcid"].astype(str).apply(lambda x: _find_oa_pdf_for_pmcid(x, oa_text) or "")
            df["oa_pdf_url"] = df["oa_pdf_relpath"].apply(lambda p: f"{OA_BASE}/{p}" if p else "")
            # Try both explicit tar path and legacy tar name
            df["oa_tar_relpath"] = df["pmcid"].astype(str).apply(lambda x: _find_oa_bulk_for_pmcid(x, oa_text) or "")
            df["oa_tar_name"] = df.apply(
                lambda row: (row["oa_tar_relpath"].split("/")[-1] if row["oa_tar_relpath"] else _find_tar_for_pmcid(row["pmcid"], oa_text) or ""),
                axis=1,
            )
            df["oa_tar_url"] = df.apply(
                lambda row: (f"{OA_BASE}/{row['oa_tar_relpath']}" if row['oa_tar_relpath'] else (f"{OA_BULK_BASE}/{row['oa_tar_name']}" if row['oa_tar_name'] else "")),
                axis=1,
            )
        else:
            df["oa_pdf_relpath"] = ""
            df["oa_pdf_url"] = ""
            df["oa_tar_relpath"] = ""
            df["oa_tar_name"] = ""
            df["oa_tar_url"] = ""

    df.to_csv(csv_path, index=False, encoding="utf-8")
    df.to_excel(excel_path, index=False)

    print(f"Combined data saved to {csv_path} and {excel_path}")