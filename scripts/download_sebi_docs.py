"""
SEBI Master Circular Downloader (FINAL EXPANDED VERSION)
"""

import os
import json
import time
import hashlib
import requests
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin
from bs4 import BeautifulSoup

from logger import setup_logger
logger = setup_logger("sebi_downloader")

RAW_DIR = Path("data/raw")
MANIFEST_FILE = Path("data/manifest.json")
DELAY_BETWEEN_REQUESTS = 1.5
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,text/html,*/*",
}

SEBI_LISTING_PAGES = [
    "https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=1&ssid=6&smid=0",
    "https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=1&ssid=7&smid=0",
]


def load_manifest():
    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE) as f:
            return json.load(f)
    return {}


def save_manifest(manifest):
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)


def file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def is_already_downloaded(filename, manifest):
    filepath = RAW_DIR / filename
    if filepath.exists() and filename in manifest:
        if manifest[filename].get("status") == "success":
            logger.info(f"[SKIP] {filename}")
            return True
    return False


def slugify(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text)
    return text[:80]


def is_relevant(title):
    title = title.lower()
    keywords = [
        "mutual", "fund", "icdr", "broker", "kyc",
        "compliance", "surveillance", "investor",
        "portfolio", "guidelines", "circular", "framework"
    ]
    return any(k in title for k in keywords)


def download_pdf(url, filename, manifest):
    filepath = RAW_DIR / filename

    if filepath.exists():
        logger.info(f"[SKIP EXISTING] {filename}")
        return True

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"[TRY {attempt}] {filename}")

            resp = requests.get(
                url,
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT,
                stream=True,
                allow_redirects=True,
            )

            if resp.status_code != 200:
                logger.warning(f"{filename} HTTP {resp.status_code}")
                return False

            content_type = resp.headers.get("Content-Type", "")

            if "pdf" not in content_type:
                logger.warning(f"{filename} suspicious content-type: {content_type}")

                if "text/html" in content_type:
                    match = re.search(
                        r"https://www\.sebi\.gov\.in/sebi_data/[^\s'\"]+\.pdf",
                        resp.text,
                    )
                    if match:
                        logger.info("Found embedded PDF → retrying")
                        return download_pdf(match.group(0), filename, manifest)

            RAW_DIR.mkdir(parents=True, exist_ok=True)

            with open(filepath, "wb") as f:
                for chunk in resp.iter_content(8192):
                    if chunk:
                        f.write(chunk)

            size_kb = filepath.stat().st_size // 1024

            if size_kb < 10:
                logger.warning(f"{filename} too small → retry")
                filepath.unlink()
                continue

            manifest[filename] = {
                "url": url,
                "status": "success",
                "size_kb": size_kb,
                "md5": file_hash(filepath),
                "timestamp": datetime.now().isoformat(),
            }

            logger.info(f"[OK] {filename} ({size_kb} KB)")
            return True

        except Exception:
            logger.error(f"{filename} failed", exc_info=True)

        time.sleep(DELAY_BETWEEN_REQUESTS * (attempt + 1))

    manifest[filename] = {
        "url": url,
        "status": "failed",
        "timestamp": datetime.now().isoformat(),
    }
    return False


def scrape_sebi_listing(listing_url):
    found = []
    try:
        logger.info(f"Scraping: {listing_url}")
        resp = requests.get(listing_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        soup = BeautifulSoup(resp.text, "html.parser")

        for a in soup.find_all("a", href=True):
            href = a["href"]
            title = a.get_text(strip=True)

            if (
                "/legal/master-circulars/" in href
                or "/legal/circulars/" in href
                or "/legal/regulations/" in href
                or "/legal/guidelines/" in href
            ):
                if title and is_relevant(title):
                    full_url = (
                        href if href.startswith("http")
                        else f"https://www.sebi.gov.in{href}"
                    )
                    fname = slugify(title) + ".pdf"
                    found.append((fname, full_url))

        logger.info(f"Found {len(found)} relevant links")

    except Exception:
        logger.error("Scraping failed", exc_info=True)

    return found


def find_pdf_on_circular_page(page_url):
    try:
        resp = requests.get(page_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        html = resp.text

        match = re.search(
            r"https://www\.sebi\.gov\.in/sebi_data/[^\s'\"]+\.pdf",
            html,
        )
        if match:
            return match.group(0)

        soup = BeautifulSoup(html, "html.parser")

        for a in soup.find_all("a", href=True):
            if ".pdf" in a["href"]:
                return urljoin("https://www.sebi.gov.in", a["href"])

        for tag in soup.find_all(["iframe", "embed", "object"]):
            src = tag.get("src") or tag.get("data") or ""
            if ".pdf" in src:
                return urljoin("https://www.sebi.gov.in", src)

    except Exception:
        logger.error("PDF extraction failed", exc_info=True)

    return None


def main():
    logger.info("=" * 60)
    logger.info("SEBI DATASET DOWNLOADER STARTED")
    logger.info("=" * 60)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    success = skip = fail = 0

    scraped_links = []
    for url in SEBI_LISTING_PAGES:
        scraped_links.extend(scrape_sebi_listing(url))
        time.sleep(DELAY_BETWEEN_REQUESTS)

    for filename, page_url in scraped_links[:80]:
        if is_already_downloaded(filename, manifest):
            skip += 1
            continue

        logger.info(f"Processing: {page_url}")
        pdf_url = find_pdf_on_circular_page(page_url)

        if not pdf_url:
            logger.warning(f"No PDF found: {page_url}")
            fail += 1
            continue

        if download_pdf(pdf_url, filename, manifest):
            success += 1
        else:
            fail += 1

        save_manifest(manifest)
        time.sleep(DELAY_BETWEEN_REQUESTS)

    total = success + skip + fail

    logger.info("=" * 60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info(f"Total   : {total}")
    logger.info(f"Success : {success}")
    logger.info(f"Skipped : {skip}")
    logger.info(f"Failed  : {fail}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()