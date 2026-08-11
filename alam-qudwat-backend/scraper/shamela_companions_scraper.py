#!/usr/bin/env python3
"""
Scrape the first-layer companions section from:
    Siyar A'lam al-Nubala' (Shamela book_id=10906)

Default verified Shamela internal page range:
    1440 .. 3085

The next section, "كبار التابعين", starts at internal page 3086.

Outputs (default: data/raw/companions_tier1):
    pages/<page_id>.json                  per-page cache / resumable download
    companions_tier1_pages.jsonl         page-level RAG source
    companions_tier1_raw.txt             combined text for human review
    companions_tier1_manifest.json       scrape metadata and summary

Run:
    python shamela_companions_scraper.py

Re-run safely to resume from cache:
    python shamela_companions_scraper.py

Force fresh download:
    python shamela_companions_scraper.py --refresh
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BOOK_ID = 10906
BOOK_TITLE = "سير أعلام النبلاء - ط الرسالة"
AUTHOR = "شمس الدين الذهبي"
BASE_URL = f"https://read.shamela.ws/book/{BOOK_ID}/{{page_id}}"
# BASE_URL = f"https://read.shamela.ws/book/10906/3084"

# Verified against the current Shamela index on 2026-08-11.
START_PAGE_ID = 1440  # أبو عبيدة بن الجراح
END_PAGE_ID = 3083    # final page before "كبار التابعين"
NEXT_SECTION_PAGE_ID = 3084

DATASET_ID = "siyar_companions_tier1"
DATASET_NAME = "الطبقة الأولى - قسم الصحابة"
DEFAULT_OUTPUT = "data/raw/companions_tier1"

USER_AGENT = "Mozilla/5.0 (compatible; AlamQudwatRAGScraper/2.0)"

ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
ARABIC_DIACRITICS_RE = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]"
)

# We intentionally do NOT label every indexed biography as "صحابي" at the
# person level. The printed section contains a small number of contextual
# biographies as well. The source section is preserved verbatim so downstream
# curation can make an explicit scholarly classification.


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept-Language": "ar,en;q=0.8",
            "Accept": "text/html,application/xhtml+xml",
        }
    )
    return session


def clean_line(line: str) -> str:
    line = line.replace("\u200f", "").replace("\u200e", "")
    return re.sub(r"[ \t\u00a0]+", " ", line).strip()


def strip_diacritics(text: str) -> str:
    return ARABIC_DIACRITICS_RE.sub("", text)


def normalize_for_match(text: str) -> str:
    text = strip_diacritics(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def contains_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text))


def arabic_number_to_int(value: str | None) -> int | None:
    if not value:
        return None
    ascii_value = value.translate(ARABIC_DIGITS)
    match = re.search(r"\d+", ascii_value)
    return int(match.group()) if match else None


def stable_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def remove_non_content_tags(soup: BeautifulSoup) -> None:
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()


def soup_lines(soup: BeautifulSoup) -> list[str]:
    raw_text = soup.get_text("\n")
    return [
        cleaned
        for raw in raw_text.splitlines()
        if (cleaned := clean_line(raw))
    ]


def extract_printed_meta(soup: BeautifulSoup) -> tuple[int | None, int | None]:
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    volume_match = re.search(r"ج\s*([0-9٠-٩]+)", title)
    page_match = re.search(r"ص\s*([0-9٠-٩]+)", title)
    return (
        arabic_number_to_int(volume_match.group(1)) if volume_match else None,
        arabic_number_to_int(page_match.group(1)) if page_match else None,
    )


def extract_title_entry(soup: BeautifulSoup) -> str | None:
    """Best-effort current entry name from the HTML title."""
    if not soup.title:
        return None

    title = clean_line(soup.title.get_text(" ", strip=True))
    title = re.sub(r"\s*-\s*المكتبة الشاملة\s*$", "", title)

    # Everything after the book title is normally the current biography/section.
    patterns = [
        r"كتاب سير أعلام النبلاء ط الرسالة\s*-\s*(.+)$",
        r"كتاب سير أعلام النبلاء - ط الرسالة\s*-\s*(.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, title)
        if match:
            value = clean_line(match.group(1))
            return value or None
    return None


def extract_breadcrumb(lines: list[str]) -> list[str]:
    markers = ("مسار الصفحة الحالية:", "مسار الصفحة الحالية")
    start = None
    for index, line in enumerate(lines):
        if line in markers or line.startswith("مسار الصفحة الحالية"):
            start = index + 1
            break

    if start is None:
        return []

    result: list[str] = []
    stop_markers = {
        "+ - التشكيل",
        "+- التشكيل",
        "التشكيل",
        "تحميل الصفحة السابقة",
    }

    for line in lines[start:]:
        if line in stop_markers or line.startswith("تحميل الصفحة السابقة"):
            break
        if line in {"فهرس الكتاب", BOOK_TITLE, "بحــث", "بحث"}:
            continue
        result.append(line)

    return result


def remove_leading_number(title: str) -> str:
    # Examples: "١ - أبو عبيدة ...", "102 - مروان ..."
    return re.sub(r"^\s*[0-9٠-٩]+\s*-\s*", "", title).strip()


def infer_entry_and_section(
    breadcrumb: list[str], title_entry: str | None
) -> tuple[str | None, str | None]:
    """Return (entry_name, source_section)."""
    clean_crumbs = [c for c in breadcrumb if c and not c.startswith("الجزء ")]

    entry_name = None
    section = None

    if clean_crumbs:
        last = clean_crumbs[-1]
        if re.match(r"^[0-9٠-٩]+\s*-\s*", last):
            entry_name = remove_leading_number(last)
            if len(clean_crumbs) >= 2:
                section = clean_crumbs[-2]
        else:
            section = last

    if not entry_name and title_entry:
        # Section-title pages exist, so only use the title as an entry when it
        # is not one of the known section labels.
        normalized = normalize_for_match(title_entry)
        section_labels = {
            normalize_for_match(value)
            for value in (
                "الصحابة",
                "السابقون الأولون",
                "تابع: الطبقة الأولى - الصحابة",
                "ومن بقايا صغار الصحابة",
                "ومن صغار الصحابة",
                "كبار التابعين",
            )
        }
        if normalized not in section_labels:
            entry_name = remove_leading_number(title_entry)

    return entry_name, section


def trim_before_next_section(text: str) -> tuple[str, bool]:
    """Trim any accidental start of the next section from the final source page."""
    lines = text.splitlines()
    boundary_index = None

    for i, line in enumerate(lines):
        normalized = normalize_for_match(line)
        if "كبار التابعين" in normalized:
            boundary_index = i
            break

    if boundary_index is None:
        return text.strip(), False

    trimmed = "\n".join(lines[:boundary_index]).strip()
    return trimmed, True


def extract_page(html: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    printed_volume, printed_page = extract_printed_meta(soup)
    title_entry = extract_title_entry(soup)

    remove_non_content_tags(soup)
    lines = soup_lines(soup)
    breadcrumb = extract_breadcrumb(lines)
    entry_name, source_section = infer_entry_and_section(breadcrumb, title_entry)

    start_marker = "تحميل الصفحة السابقة"
    end_marker = "تحميل الصفحة التالية"

    try:
        start_index = lines.index(start_marker) + 1
    except ValueError as exc:
        raise ValueError(
            "لم أجد علامة بداية نص الصفحة. ربما تغيّر تصميم موقع الشاملة."
        ) from exc

    try:
        end_index = lines.index(end_marker, start_index)
    except ValueError as exc:
        raise ValueError(
            "لم أجد علامة نهاية نص الصفحة. ربما تغيّر تصميم موقع الشاملة."
        ) from exc

    text = "\n".join(lines[start_index:end_index]).strip()
    text, boundary_trimmed = trim_before_next_section(text)

    if len(text) < 20 or not contains_arabic(text):
        raise ValueError("النص المستخرج قصير جدًا أو لا يبدو نصًا عربيًا صحيحًا.")

    return {
        "printed_volume": printed_volume,
        "printed_page": printed_page,
        "entry_name": entry_name,
        "source_section": source_section,
        "breadcrumb": breadcrumb,
        "text": text,
        "content_hash": stable_text_hash(text),
        "next_section_trimmed": boundary_trimmed,
    }


def fetch_page(
    session: requests.Session,
    page_id: int,
    timeout: int = 30,
) -> dict[str, Any]:
    url = BASE_URL.format(page_id=page_id)
    response = session.get(url, timeout=timeout)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    if "text/html" not in content_type.lower():
        raise ValueError(f"استجابة غير متوقعة من {url}: {content_type}")

    parsed = extract_page(response.text)

    character_name = parsed.get("entry_name")
    character_id = None
    if character_name:
        character_id = "siyar_" + hashlib.sha256(
            normalize_for_match(character_name).encode("utf-8")
        ).hexdigest()[:16]

    return {
        "book_id": BOOK_ID,
        "book_title": BOOK_TITLE,
        "author": AUTHOR,
        "dataset_id": DATASET_ID,
        "dataset_name": DATASET_NAME,
        "layer": 1,
        "source_group": "الصحابة",
        # Important: group != scholarly person-level classification.
        "person_category": None,
        "character_id": character_id,
        "character_name": character_name,
        "page_id": page_id,
        "url": url,
        "scraped_at": utc_now_iso(),
        **parsed,
    }


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(path)


def load_cached_record(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    required = {"page_id", "url", "text", "content_hash"}
    missing = required.difference(record)
    if missing:
        raise ValueError(f"Cache غير صالح في {path}; حقول ناقصة: {sorted(missing)}")
    return record


def write_outputs(output_root: Path, records: list[dict[str, Any]]) -> None:
    jsonl_path = output_root / "companions_tier1_pages.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    raw_path = output_root / "companions_tier1_raw.txt"
    with raw_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                f"\n\n===== page_id={record['page_id']}"
                f" | volume={record.get('printed_volume')}"
                f" | page={record.get('printed_page')}"
                f" | entry={record.get('entry_name') or '-'} =====\n\n"
            )
            f.write(record["text"].strip())
            f.write("\n")

    entry_counts = Counter(
        record.get("entry_name") for record in records if record.get("entry_name")
    )
    section_counts = Counter(
        record.get("source_section")
        for record in records
        if record.get("source_section")
    )

    manifest = {
        "dataset_id": DATASET_ID,
        "dataset_name": DATASET_NAME,
        "book_id": BOOK_ID,
        "book_title": BOOK_TITLE,
        "author": AUTHOR,
        "internal_page_start": START_PAGE_ID,
        "internal_page_end": END_PAGE_ID,
        "next_section_page_id": NEXT_SECTION_PAGE_ID,
        "total_pages": len(records),
        "unique_detected_entries": len(entry_counts),
        "entries": [
            {"entry_name": name, "page_count": count}
            for name, count in entry_counts.items()
        ],
        "source_sections": dict(section_counts),
        "generated_at": utc_now_iso(),
        "notes": [
            "This dataset follows the printed/indexed companions block in Siyar A'lam al-Nubala'.",
            "person_category is intentionally null: the section may include contextual biographies; curate person-level scholarly classification separately.",
            "The Rashidun caliphs are not duplicated here because they are in the separate Rashidun collection already scraped before this range.",
        ],
    }
    save_json(output_root / "companions_tier1_manifest.json", manifest)

    print(f"\nSaved JSONL : {jsonl_path}")
    print(f"Saved raw   : {raw_path}")
    print(f"Saved manifest: {output_root / 'companions_tier1_manifest.json'}")
    print(f"Pages: {len(records)}")
    print(f"Detected entries: {len(entry_counts)}")


def scrape(
    output_root: Path,
    delay: float,
    refresh: bool,
    start_page: int,
    end_page: int,
) -> list[dict[str, Any]]:
    pages_dir = output_root / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    session = build_session()
    records: list[dict[str, Any]] = []
    total = end_page - start_page + 1

    print(f"Dataset: {DATASET_NAME}")
    print(f"Shamela range: {start_page}..{end_page} ({total} pages)")
    print(f"Output: {output_root}")
    print("Existing page cache will be reused automatically.\n")

    try:
        for index, page_id in enumerate(range(start_page, end_page + 1), start=1):
            cache_path = pages_dir / f"{page_id}.json"

            if cache_path.exists() and not refresh:
                record = load_cached_record(cache_path)
                print(f"[{index:04}/{total}] cache page_id={page_id}")
            else:
                print(
                    f"[{index:04}/{total}] fetch page_id={page_id} ...",
                    end=" ",
                    flush=True,
                )
                record = fetch_page(session, page_id)

                # Strong guard: if Shamela's structure/range changes and the
                # current page clearly belongs to the next section, abort rather
                # than silently poisoning the companions dataset.
                section_norm = normalize_for_match(record.get("source_section") or "")
                breadcrumb_norm = normalize_for_match(
                    " ".join(record.get("breadcrumb") or [])
                )
                if "كبار التابعين" in section_norm or "كبار التابعين" in breadcrumb_norm:
                    raise RuntimeError(
                        f"page_id={page_id} أصبح ضمن قسم كبار التابعين. "
                        "أوقف السكربر لحماية الـdataset؛ راجع الحدود قبل المتابعة."
                    )

                save_json(cache_path, record)
                print("OK")

                if page_id != end_page:
                    time.sleep(delay)

            records.append(record)

    finally:
        session.close()

    return records


def smoke_test(delay: float) -> None:
    """Fetch only start page + known next-section page to validate live boundaries."""
    session = build_session()
    try:
        print(f"Checking start page {START_PAGE_ID} ...")
        start = fetch_page(session, START_PAGE_ID)
        print(
            "  OK:",
            start.get("entry_name"),
            f"(vol={start.get('printed_volume')}, page={start.get('printed_page')})",
        )
        time.sleep(delay)

        print(f"Checking next-section page {NEXT_SECTION_PAGE_ID} ...")
        nxt = fetch_page(session, NEXT_SECTION_PAGE_ID)
        combined = normalize_for_match(
            " ".join(nxt.get("breadcrumb") or [])
            + " "
            + (nxt.get("source_section") or "")
        )
        if "كبار التابعين" not in combined:
            raise RuntimeError(
                "فشل التحقق: الصفحة المتوقعة لبداية كبار التابعين لم تعد مصنفة كذلك. "
                "لا تشغّل الـfull scrape قبل مراجعة الحدود."
            )
        print("  OK: boundary still points to كبار التابعين")
        print("Smoke test passed.")
    finally:
        session.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape the first-layer companions block from Siyar A'lam al-Nubala' on Shamela."
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"مجلد الحفظ (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="ثوانٍ بين الطلبات الجديدة (default: 1.5; minimum: 1.0)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="إعادة تنزيل الصفحات حتى لو كانت موجودة في cache",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="تحقق سريع من أول صفحة وحد بداية كبار التابعين بدون تنزيل المجموعة كاملة",
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=START_PAGE_ID,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--end-page",
        type=int,
        default=END_PAGE_ID,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if args.delay < 0.1:
        print("رفض التشغيل: استخدم --delay بقيمة 0.1 ثانية أو أكثر.", file=sys.stderr)
        sys.exit(2)

    if args.start_page > args.end_page:
        print("خطأ: start-page أكبر من end-page.", file=sys.stderr)
        sys.exit(2)

    # Never silently cross the verified next-section boundary.
    if args.end_page >= NEXT_SECTION_PAGE_ID:
        print(
            f"رفض التشغيل: end-page يجب أن يكون أقل من {NEXT_SECTION_PAGE_ID} "
            "حتى لا يدخل كبار التابعين في Dataset الصحابة.",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        if args.smoke_test:
            smoke_test(args.delay)
            return

        output_root = Path(args.output)
        output_root.mkdir(parents=True, exist_ok=True)

        records = scrape(
            output_root=output_root,
            delay=0.1,
            refresh=args.refresh,
            start_page=args.start_page,
            end_page=args.end_page,
        )
        write_outputs(output_root, records)
        print("\nDone.")

    except KeyboardInterrupt:
        print(
            "\nتم إيقاف السكربت. الصفحات المنزلة محفوظة ويمكن إعادة التشغيل للمتابعة."
        )
        sys.exit(130)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        print(
            "ما تم تنزيله بقي محفوظًا في cache. أصلح السبب ثم أعد نفس الأمر للمتابعة.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
