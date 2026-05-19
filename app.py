import os
import re
import json
import threading
import tempfile
import zipfile
from collections import Counter
from dotenv import load_dotenv
from flask import Flask, request, render_template, send_file, abort
from azure.storage.blob import BlobServiceClient
import fitz  # PyMuPDF

# Import two-stage AI screener and anonymizer
from ai_screener import run_two_stage_screening
from anonymizer import anonymize_resume_text

load_dotenv()

app = Flask(__name__)
app.secret_key = "resume-search-secret-key"

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_BLOB_CONTAINER = os.getenv("AZURE_BLOB_CONTAINER", "resumes")

DEFAULT_TOP_N = 10
MAX_RESUMES_TO_CACHE = None
CACHE_FILE = "resume_cache.json"

# Default counts for each AI stage — overridable from the HTML form
DEFAULT_DEEPSEEK_TOP_N = 10
DEFAULT_GPT_TOP_N = 5

RESUME_CACHE = []

# Replaces Flask session for storing large search state server-side
SEARCH_STATE = {}

CACHE_LOADING = False
CACHE_LOADED = False
CACHE_PROGRESS = 0
CACHE_TOTAL = 0

KEYWORD_COLOR = (1, 0.65, 0.65)       # light red
REQUIREMENT_COLOR = (1, 0.95, 0.45)   # light yellow

REFERENCE_PATTERNS = [
    r"\bprofessional\s+referees\b",
    r"\bprofessional\s+references\b",
    r"\breferees\s*:",
    r"\breferee\s*:",
    r"\breferences\s*:",
    r"\breference\s*:",
    r"\breferees\b",
    r"\breferences\b",
    r"\breferee\b",
    r"\breference\b",
    r"r\s*e\s*f\s*e\s*r\s*e\s*e\s*s",
    r"r\s*e\s*f\s*e\s*r\s*e\s*n\s*c\s*e\s*s",
]

STOPWORDS = {
    "the", "and", "with", "for", "from", "that", "this", "need", "needs",
    "must", "have", "has", "had", "who", "will", "can", "are", "was",
    "were", "been", "being", "into", "onto", "about", "your", "their",
    "our", "you", "they", "his", "her", "she", "him", "job", "role",
    "candidate", "applicant", "person", "work", "working", "resume",
    "experience", "background", "ability", "skills", "skill", "required",
    "requirement", "requirements", "looking", "seek", "seeking", "able",
    "strong", "good", "excellent", "preferred", "essential"
}


def get_container_client():
    if not AZURE_STORAGE_CONNECTION_STRING:
        raise RuntimeError("Missing AZURE_STORAGE_CONNECTION_STRING")

    blob_service = BlobServiceClient.from_connection_string(
        AZURE_STORAGE_CONNECTION_STRING
    )
    return blob_service.get_container_client(AZURE_BLOB_CONTAINER)


def normalize_search_text(text):
    text = text or ""
    text = text.lower()
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_reference_section(text):
    text = text or ""
    cut_position = len(text)

    for pattern in REFERENCE_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            cut_position = min(cut_position, match.start())

    return text[:cut_position]


def parse_keywords(raw_keywords):
    parts = re.split(r"[\n,;]+", raw_keywords or "")
    return [x.strip() for x in parts if x.strip()]


def keyword_to_pattern(keyword):
    keyword = keyword.strip()

    if not keyword:
        return None

    section_match = re.fullmatch(
        r"s\.?\s*(\d+)",
        keyword,
        flags=re.IGNORECASE
    )

    if section_match:
        number = section_match.group(1)
        return rf"\b(s\s*\.?\s*{number}|section\s*{number})\b"

    escaped = re.escape(keyword)
    escaped = escaped.replace(r"\ ", r"\s+")

    return r"(?<![a-zA-Z0-9])" + escaped + r"(?![a-zA-Z0-9])"


def find_keyword_matches(text, keywords):
    text = normalize_search_text(text)
    matches = []

    for keyword in keywords:
        pattern = keyword_to_pattern(keyword)

        if not pattern:
            continue

        found = re.findall(pattern, text, flags=re.IGNORECASE)

        if found:
            matches.append({
                "keyword": keyword,
                "count": len(found),
            })

    return matches


def extract_requirement_terms(job_requirement, top_n=30):
    words = re.findall(
        r"\b[a-zA-Z][a-zA-Z0-9\-]{2,}\b",
        (job_requirement or "").lower()
    )

    clean_words = [
        word for word in words
        if word not in STOPWORDS
    ]

    counter = Counter(clean_words)

    return [
        {
            "term": term,
            "requirement_count": count,
        }
        for term, count in counter.most_common(top_n)
    ]


def count_requirement_terms(text, requirement_terms):
    text = normalize_search_text(text)
    matches = []

    for item in requirement_terms:
        term = item["term"]
        pattern = r"(?<![a-zA-Z0-9])" + re.escape(term) + r"(?![a-zA-Z0-9])"

        count = len(re.findall(pattern, text, flags=re.IGNORECASE))

        if count > 0:
            matches.append({
                "term": term,
                "count": count,
            })

    return matches


def parse_employee_document_from_text_blob(blob_name):
    match = re.match(
        r"^text/([^/]+)/resume_([^_]+)_(\d+)\.txt$",
        blob_name
    )

    if not match:
        return None

    return {
        "employee_id": match.group(1),
        "document_id": match.group(3),
        "text_blob": blob_name,
    }


def read_blob_text(container_client, blob_name):
    data = container_client.get_blob_client(blob_name).download_blob().readall()
    return data.decode("utf-8", errors="ignore")


def read_blob_bytes(container_client, blob_name):
    return container_client.get_blob_client(blob_name).download_blob().readall()


def read_latest_index(container_client, employee_id):
    blob_name = f"index/latest/{employee_id}.json"
    blob_client = container_client.get_blob_client(blob_name)

    if not blob_client.exists():
        return {}

    try:
        data = blob_client.download_blob().readall()
        return json.loads(data.decode("utf-8"))
    except Exception:
        return {}


def save_cache_to_file():
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(RESUME_CACHE, f, ensure_ascii=False)

    print("💾 Resume cache saved:", CACHE_FILE)


def load_cache_from_file():
    global RESUME_CACHE, CACHE_LOADED, CACHE_LOADING, CACHE_PROGRESS, CACHE_TOTAL

    if not os.path.exists(CACHE_FILE):
        return False

    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            RESUME_CACHE = json.load(f)

        CACHE_LOADING = False
        CACHE_LOADED = True
        CACHE_PROGRESS = len(RESUME_CACHE)
        CACHE_TOTAL = len(RESUME_CACHE)

        print("✅ Resume cache loaded from local file:", len(RESUME_CACHE))
        return True

    except Exception as e:
        print("⚠️ Failed to load local cache:", e)
        return False


def load_resume_cache_from_azure():
    global RESUME_CACHE
    global CACHE_LOADING
    global CACHE_LOADED
    global CACHE_PROGRESS
    global CACHE_TOTAL

    CACHE_LOADING = True
    CACHE_LOADED = False
    CACHE_PROGRESS = 0
    CACHE_TOTAL = 0

    print("📥 Loading resume cache from Azure Blob...")

    container_client = get_container_client()
    all_blobs = list(container_client.list_blobs(name_starts_with="text/"))

    if MAX_RESUMES_TO_CACHE is not None:
        all_blobs = all_blobs[:MAX_RESUMES_TO_CACHE]

    CACHE_TOTAL = len(all_blobs)
    cache = []

    for index, blob in enumerate(all_blobs, start=1):
        parsed = parse_employee_document_from_text_blob(blob.name)

        if not parsed:
            CACHE_PROGRESS = index
            continue

        employee_id = parsed["employee_id"]
        document_id = parsed["document_id"]

        try:
            raw_text = read_blob_text(container_client, blob.name)

            if not raw_text.strip():
                CACHE_PROGRESS = index
                continue

            cleaned_text = remove_reference_section(raw_text)

            latest_index = read_latest_index(
                container_client,
                employee_id
            )

            cache.append({
                "employee_id": employee_id,
                "document_id": document_id,
                "employee_name": latest_index.get("employee_name", ""),
                "original_blob": latest_index.get("original_blob", ""),
                "text_blob": blob.name,
                "cleaned_text": cleaned_text,
                "text_length": len(cleaned_text),
            })

        except Exception as e:
            print("❌ Failed loading:", blob.name, e)

        CACHE_PROGRESS = index

        if index % 50 == 0:
            print(f"📦 Loaded {index}/{CACHE_TOTAL}")

    RESUME_CACHE = cache

    CACHE_LOADING = False
    CACHE_LOADED = True
    CACHE_PROGRESS = CACHE_TOTAL

    print("✅ Resume cache loaded from Azure:", len(RESUME_CACHE))
    save_cache_to_file()


def start_background_cache_loading():
    global CACHE_LOADING

    if CACHE_LOADING:
        return

    thread = threading.Thread(
        target=load_resume_cache_from_azure,
        daemon=True
    )
    thread.start()


def ensure_cache_loaded_for_search():
    if RESUME_CACHE:
        return True

    if CACHE_LOADING:
        return False

    if load_cache_from_file():
        return True

    start_background_cache_loading()
    return False


def search_resumes(keywords_text, job_requirement, top_n):
    keywords = parse_keywords(keywords_text)
    requirement_terms = extract_requirement_terms(job_requirement)

    results = []

    for resume in RESUME_CACHE:
        cleaned_text = resume["cleaned_text"]

        keyword_matches = find_keyword_matches(cleaned_text, keywords)

        if len(keyword_matches) < len(keywords):
            continue

        requirement_matches = count_requirement_terms(
            cleaned_text,
            requirement_terms
        )

        keyword_score = sum(x["count"] for x in keyword_matches) * 10
        requirement_score = sum(x["count"] for x in requirement_matches) * 3
        total_score = keyword_score + requirement_score

        results.append({
            "employee_id": resume["employee_id"],
            "document_id": resume["document_id"],
            "employee_name": resume["employee_name"],
            "score": total_score,
            "keyword_score": keyword_score,
            "requirement_score": requirement_score,
            "matched_keywords": keyword_matches,
            "matched_requirement_terms": requirement_matches,
            "original_blob": resume["original_blob"],
            "text_blob": resume["text_blob"],
            "resume_text": cleaned_text[:3000],
            "text_length": resume["text_length"],
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return {
        "results": results if top_n == 0 else results[:top_n],
        "all_matched_count": len(results),
        "scanned_count": len(RESUME_CACHE),
        "requirement_terms": requirement_terms,
        "keywords": keywords,
    }


def find_reference_y(page):
    words = page.get_text("words")

    if not words:
        return None

    words = sorted(words, key=lambda w: (round(w[1], 1), w[0]))
    lines = {}

    for w in words:
        x0, y0, x1, y1, word, *_ = w
        line_key = round(y0, 1)
        lines.setdefault(line_key, []).append(word)

    for y, line_words in lines.items():
        line_text = " ".join(line_words).lower()
        compact = re.sub(r"[^a-z]", "", line_text)

        if (
            "professionalreferees" in compact
            or "professionalreferences" in compact
            or "referees" in compact
            or "references" in compact
            or "referee" in compact
            or "reference" in compact
        ):
            return y

    return None


def delete_reference_area(page, ref_y):
    if ref_y is None:
        return

    delete_rect = fitz.Rect(
        0,
        ref_y,
        page.rect.width,
        page.rect.height
    )

    page.draw_rect(
        delete_rect,
        color=(1, 1, 1),
        fill=(1, 1, 1)
    )


def highlight_terms_before_reference(page, terms, ref_y, color):
    for term in terms:
        if not term:
            continue

        instances = page.search_for(term)

        for inst in instances:
            if ref_y is not None and inst.y0 >= ref_y:
                continue

            highlight = page.add_highlight_annot(inst)
            highlight.set_colors(stroke=color)
            highlight.update()


def build_highlight_terms_from_keywords(keywords):
    terms = []

    for keyword in keywords:
        keyword = keyword.strip()

        if not keyword:
            continue

        section_match = re.fullmatch(
            r"s\.?\s*(\d+)",
            keyword,
            flags=re.IGNORECASE
        )

        if section_match:
            number = section_match.group(1)
            terms.extend([
                f"S{number}",
                f"S.{number}",
                f"Section {number}",
            ])
        else:
            terms.append(keyword)

    return list(dict.fromkeys(terms))


def build_highlighted_pdf_bytes(employee_id, document_id, keywords_text, requirement_terms):
    container_client = get_container_client()

    blob_name = f"original/{employee_id}/resume_{employee_id}_{document_id}.pdf"
    blob_client = container_client.get_blob_client(blob_name)

    if not blob_client.exists():
        abort(404, description="Original resume PDF not found.")

    pdf_bytes = blob_client.download_blob().readall()

    keywords = parse_keywords(keywords_text)
    keyword_terms = build_highlight_terms_from_keywords(keywords)
    requirement_highlight_terms = [
        item["term"]
        for item in requirement_terms
        if item.get("term")
    ]

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    reference_started = False

    for page in doc:
        if reference_started:
            page.draw_rect(
                page.rect,
                color=(1, 1, 1),
                fill=(1, 1, 1)
            )
            continue

        ref_y = find_reference_y(page)

        highlight_terms_before_reference(
            page=page,
            terms=keyword_terms,
            ref_y=ref_y,
            color=KEYWORD_COLOR
        )

        highlight_terms_before_reference(
            page=page,
            terms=requirement_highlight_terms,
            ref_y=ref_y,
            color=REQUIREMENT_COLOR
        )

        if ref_y is not None:
            delete_reference_area(page, ref_y)
            reference_started = True

    output_bytes = doc.tobytes(
        garbage=4,
        deflate=True
    )

    doc.close()
    return output_bytes


def render_page(
    keywords="",
    job_requirement="",
    results=None,
    total_results=0,
    requirement_terms=None,
    top_n=DEFAULT_TOP_N,
    deepseek_results=None,
    gpt_results=None,
    deepseek_top_n=DEFAULT_DEEPSEEK_TOP_N,
    gpt_top_n=DEFAULT_GPT_TOP_N,
):
    return render_template(
        "resumesearch1.html",
        keywords=keywords,
        job_requirement=job_requirement,
        results=results or [],
        total_resumes=len(RESUME_CACHE),
        total_results=total_results,
        requirement_terms=requirement_terms or [],
        top_n=top_n,
        cache_loading=CACHE_LOADING,
        cache_loaded=CACHE_LOADED,
        cache_progress=CACHE_PROGRESS,
        cache_total=CACHE_TOTAL,
        deepseek_results=deepseek_results or [],
        gpt_results=gpt_results or [],
        deepseek_top_n=deepseek_top_n,
        gpt_top_n=gpt_top_n,
    )


@app.route("/", methods=["GET"])
def home():
    if not RESUME_CACHE and not CACHE_LOADING:
        if not load_cache_from_file():
            start_background_cache_loading()

    return render_page()


@app.route("/search", methods=["POST"])
def search():
    keywords_text   = request.form.get("keywords", "")
    job_requirement = request.form.get("job_requirement", "")
    top_n_raw       = request.form.get("top_n", str(DEFAULT_TOP_N))
    deepseek_top_n  = int(request.form.get("deepseek_top_n", DEFAULT_DEEPSEEK_TOP_N))
    gpt_top_n       = int(request.form.get("gpt_top_n", DEFAULT_GPT_TOP_N))

    try:
        top_n = int(top_n_raw)
    except Exception:
        top_n = DEFAULT_TOP_N

    if not ensure_cache_loaded_for_search():
        return render_page(
            keywords=keywords_text,
            job_requirement=job_requirement,
            top_n=top_n,
        )

    data = search_resumes(
        keywords_text=keywords_text,
        job_requirement=job_requirement,
        top_n=top_n,
    )

    # Store large state server-side in memory — avoids cookie size limit
    SEARCH_STATE["keywords"]          = keywords_text
    SEARCH_STATE["job_requirement"]   = job_requirement
    SEARCH_STATE["requirement_terms"] = data["requirement_terms"]
    SEARCH_STATE["results"]           = data["results"]
    SEARCH_STATE["keywords_list"]     = data["keywords"]

    return render_page(
        keywords=keywords_text,
        job_requirement=job_requirement,
        results=data["results"],
        total_results=data["all_matched_count"],
        requirement_terms=data["requirement_terms"],
        top_n=top_n,
        deepseek_top_n=deepseek_top_n,
        gpt_top_n=gpt_top_n,
    )


@app.route("/run-ai-screening", methods=["POST"])
def run_ai_screening():
    # Pull everything from server-side state — no session needed
    keywords_text     = SEARCH_STATE.get("keywords", "")
    job_requirement   = SEARCH_STATE.get("job_requirement", "")
    requirement_terms = SEARCH_STATE.get("requirement_terms", [])
    results           = SEARCH_STATE.get("results", [])
    keywords          = SEARCH_STATE.get("keywords_list", [])

    deepseek_top_n = int(request.form.get("deepseek_top_n", DEFAULT_DEEPSEEK_TOP_N))
    gpt_top_n      = int(request.form.get("gpt_top_n", DEFAULT_GPT_TOP_N))

    if not results:
        print("⚠️ No keyword results found in SEARCH_STATE — run a search first.")
        return render_page()

    print(f"🤖 Running AI screening: {len(results)} candidates → Deepseek top {deepseek_top_n} → GPT top {gpt_top_n}")

    ai_data = run_two_stage_screening(
        job_requirement=job_requirement,
        keywords=keywords,
        keyword_results=results,
        deepseek_top_n=deepseek_top_n,
        gpt_top_n=gpt_top_n,
    )

    return render_page(
        keywords=keywords_text,
        job_requirement=job_requirement,
        results=results,
        total_results=len(results),
        requirement_terms=requirement_terms,
        deepseek_results=ai_data["deepseek_results"],
        gpt_results=ai_data["gpt_results"],
        deepseek_top_n=deepseek_top_n,
        gpt_top_n=gpt_top_n,
    )


@app.route("/download-highlighted/<employee_id>/<document_id>", methods=["GET"])
def download_highlighted_resume(employee_id, document_id):
    keywords_text     = SEARCH_STATE.get("keywords", "")
    requirement_terms = SEARCH_STATE.get("requirement_terms", [])

    highlighted_bytes = build_highlighted_pdf_bytes(
        employee_id=employee_id,
        document_id=document_id,
        keywords_text=keywords_text,
        requirement_terms=requirement_terms,
    )

    temp_path = os.path.join(
        tempfile.gettempdir(),
        f"highlighted_resume_{employee_id}_{document_id}.pdf"
    )

    with open(temp_path, "wb") as f:
        f.write(highlighted_bytes)

    return send_file(
        temp_path,
        as_attachment=True,
        download_name=f"highlighted_resume_{employee_id}_{document_id}.pdf",
        mimetype="application/pdf"
    )


@app.route("/download-all-highlighted-results", methods=["GET"])
def download_all_highlighted_results():
    results           = SEARCH_STATE.get("results", [])
    keywords_text     = SEARCH_STATE.get("keywords", "")
    requirement_terms = SEARCH_STATE.get("requirement_terms", [])

    if not results:
        abort(404, description="No search results to download.")

    zip_path = os.path.join(
        tempfile.gettempdir(),
        "highlighted_resume_search_results.zip"
    )

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for row in results:
            employee_id   = str(row.get("employee_id", ""))
            document_id   = str(row.get("document_id", ""))
            employee_name = str(row.get("employee_name", ""))
            safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", employee_name).strip("_")

            try:
                pdf_bytes = build_highlighted_pdf_bytes(
                    employee_id=employee_id,
                    document_id=document_id,
                    keywords_text=keywords_text,
                    requirement_terms=requirement_terms,
                )

                filename = f"{employee_id}_{safe_name}_highlighted_resume_{document_id}.pdf"
                zip_file.writestr(filename, pdf_bytes)

            except Exception as e:
                print("❌ Failed adding highlighted PDF:", employee_id, document_id, e)

    return send_file(
        zip_path,
        as_attachment=True,
        download_name="highlighted_resume_search_results.zip",
        mimetype="application/zip"
    )


@app.route("/refresh-cache", methods=["GET"])
def refresh_cache():
    start_background_cache_loading()
    return render_page()


@app.route("/cache-status", methods=["GET"])
def cache_status():
    return {
        "cache_loading":  CACHE_LOADING,
        "cache_loaded":   CACHE_LOADED,
        "cache_progress": CACHE_PROGRESS,
        "cache_total":    CACHE_TOTAL,
        "loaded_resumes": len(RESUME_CACHE),
    }


if __name__ == "__main__":
    if not load_cache_from_file():
        start_background_cache_loading()

    app.run(debug=True, port=5000, use_reloader=False)