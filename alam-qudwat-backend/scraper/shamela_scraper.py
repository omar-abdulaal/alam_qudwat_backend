#!/usr/bin/env python3
import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BOOK_ID = 10906
BASE_URL = f"https://read.shamela.ws/book/{BOOK_ID}/{{page_id}}"

CALIPHS = {
    "abu_bakr": {
        "name": "أبو بكر الصديق",
        "start": 1155,
        "end": 1215,
    },
    "umar": {
        "name": "عمر بن الخطاب",
        "start": 1216,
        "end": 1290,
    },
    "uthman": {
        "name": "عثمان بن عفان",
        "start": 1291,
        "end": 1364,
    },
    "ali": {
        "name": "علي بن أبي طالب",
        "start": 1365,
        "end": 1430,
    },
}

USER_AGENT = (
    "Mozilla/5.0 (compatible; AlamQudwatRAGScraper/1.0)"
)


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
    # نحافظ على النص العربي والتشكيل، وننظف فقط المسافات الأفقية الزائدة.
    return re.sub(r"[ \t\u00a0]+", " ", line).strip()


def extract_printed_page(soup: BeautifulSoup):
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    match = re.search(r"ص\s*([0-9٠-٩]+)", title)
    return match.group(1) if match else None


def contains_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text))


def extract_book_text(html: str) -> tuple[str, str | None]:
    soup = BeautifulSoup(html, "html.parser")
    printed_page = extract_printed_page(soup)

    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    raw_text = soup.get_text("\n")
    lines = [clean_line(line) for line in raw_text.splitlines()]
    lines = [line for line in lines if line]

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

    content_lines = lines[start_index:end_index]
    text = "\n".join(content_lines).strip()

    if len(text) < 40 or not contains_arabic(text):
        raise ValueError("النص المستخرج قصير جدًا أو لا يبدو نصًا عربيًا صحيحًا.")

    return text, printed_page


def fetch_page(session: requests.Session, page_id: int, timeout: int = 30) -> dict:
    url = BASE_URL.format(page_id=page_id)
    response = session.get(url, timeout=timeout)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    if "text/html" not in content_type.lower():
        raise ValueError(f"استجابة غير متوقعة من {url}: {content_type}")

    text, printed_page = extract_book_text(response.text)
    return {
        "page_id": page_id,
        "printed_page": printed_page,
        "url": url,
        "text": text,
    }


def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(path)


def scrape_caliph(
    session: requests.Session,
    key: str,
    config: dict,
    output_root: Path,
    delay: float,
    refresh: bool,
) -> list[dict]:
    caliph_dir = output_root / "pages" / key
    caliph_dir.mkdir(parents=True, exist_ok=True)

    records = []
    total = config["end"] - config["start"] + 1

    print(f"\n=== {config['name']} | {total} صفحات ===")

    for index, page_id in enumerate(range(config["start"], config["end"] + 1), start=1):
        cache_path = caliph_dir / f"{page_id}.json"

        if cache_path.exists() and not refresh:
            record = json.loads(cache_path.read_text(encoding="utf-8"))
            print(f"[{index:03}/{total}] cache page_id={page_id}")
        else:
            print(f"[{index:03}/{total}] fetch page_id={page_id} ...", end=" ", flush=True)
            page = fetch_page(session, page_id)
            record = {
                "book_id": BOOK_ID,
                "book_title": "سير أعلام النبلاء - ط الرسالة",
                "author": "شمس الدين الذهبي",
                "collection": "سير الخلفاء الراشدين",
                "caliph_id": key,
                "caliph_name": config["name"],
                **page,
            }
            save_json(cache_path, record)
            print("OK")

            if page_id != config["end"]:
                time.sleep(delay)

        records.append(record)

    # ملف نصي كامل للخليفة للمراجعة البشرية.
    txt_path = output_root / f"{key}_raw.txt"
    txt_content = "\n\n".join(record["text"] for record in records)
    txt_path.write_text(txt_content + "\n", encoding="utf-8")

    # JSONL: كل صفحة سجل مستقل مع metadata، وهو الأنسب للمرحلة التالية.
    jsonl_path = output_root / f"{key}_pages.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved: {txt_path}")
    print(f"Saved: {jsonl_path}")
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Scrape the Rashidun caliphs section from Shamela for RAG preparation."
    )
    parser.add_argument(
        "--caliph",
        choices=["all", *CALIPHS.keys()],
        default="all",
        help="أي خليفة تريد استخراجه؟ الافتراضي: all",
    )
    parser.add_argument(
        "--output",
        default="data/raw/rashidun",
        help="مجلد حفظ النتائج",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="عدد الثواني بين الطلبات الجديدة. الافتراضي: 1.5",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="إعادة تنزيل الصفحات حتى لو كانت موجودة في cache",
    )
    args = parser.parse_args()

    if args.delay < 1.0:
        print("رفض التشغيل: استخدم --delay بقيمة 1 ثانية أو أكثر.", file=sys.stderr)
        sys.exit(2)

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    selected = CALIPHS.items() if args.caliph == "all" else [(args.caliph, CALIPHS[args.caliph])]

    session = build_session()
    all_records = []

    try:
        for key, config in selected:
            records = scrape_caliph(
                session=session,
                key=key,
                config=config,
                output_root=output_root,
                delay=args.delay,
                refresh=args.refresh,
            )
            all_records.extend(records)
    except KeyboardInterrupt:
        print("\nتم إيقاف السكربت. الصفحات التي نزلت محفوظة ويمكنك إعادة التشغيل للمتابعة.")
        sys.exit(130)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        print("الصفحات التي تم تنزيلها قبل الخطأ بقيت محفوظة في cache.", file=sys.stderr)
        sys.exit(1)
    finally:
        session.close()

    combined_path = output_root / "rashidun_pages.jsonl"
    with combined_path.open("w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\nDone.")
    print(f"Combined JSONL: {combined_path}")
    print(f"Total pages: {len(all_records)}")


if __name__ == "__main__":
    main()
