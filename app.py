
"""
Text Summarization App
Summarizes long text, .txt or .pdf files using BART models.
Runs locally. No API key needed.
"""

import streamlit as st
from pypdf import PdfReader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

st.set_page_config(
    page_title="Text Summarizer",
    page_icon="📝",
    layout="wide"
)

# =========================================================
# PASTEL CSS
# =========================================================

st.markdown("""
<style>

/* ========================================================
   GLOBAL APP
   ======================================================== */

.stApp {
    background:
        radial-gradient(
            circle at 5% 5%,
            rgba(221, 214, 254, 0.55),
            transparent 25%
        ),
        radial-gradient(
            circle at 95% 10%,
            rgba(191, 219, 254, 0.55),
            transparent 25%
        ),
        radial-gradient(
            circle at 50% 100%,
            rgba(254, 215, 170, 0.35),
            transparent 30%
        ),
        #faf9ff;
    color: #374151;
}

/* ========================================================
   MAIN CONTAINER
   ======================================================== */

.block-container {
    max-width: 1250px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}

/* ========================================================
   TITLE
   ======================================================== */

h1 {
    text-align: center;
    font-size: 3.1rem !important;
    font-weight: 800 !important;
    letter-spacing: -1.5px;

    background: linear-gradient(
        90deg,
        #8b5cf6,
        #6366f1,
        #ec4899
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    margin-bottom: 0.2rem !important;
}

/* ========================================================
   SUBTITLE
   ======================================================== */

.hero-text {
    text-align: center;
    color: #6b7280;
    font-size: 17px;
    margin-bottom: 30px;
}

/* ========================================================
   SIDEBAR
   ======================================================== */

section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #f1efff 0%,
            #f8f5ff 50%,
            #eef7ff 100%
        );

    border-right: 1px solid #e5e1f5;
}

section[data-testid="stSidebar"] h2 {
    color: #5b4b8a;
    font-weight: 800;
}

section[data-testid="stSidebar"] label {
    color: #514b62 !important;
    font-weight: 600;
}

.sidebar-card {
    background: rgba(255, 255, 255, 0.65);
    border: 1px solid #e5def8;
    border-radius: 18px;
    padding: 15px;
    margin-top: 20px;
    color: #6b6378;
    font-size: 13px;
    line-height: 1.6;
}

/* ========================================================
   INPUT TYPE RADIO
   ======================================================== */

div[role="radiogroup"] {
    background: #ffffff;
    padding: 8px;
    border-radius: 16px;
    border: 1px solid #ebe7f4;
    box-shadow: 0 5px 20px rgba(99, 102, 241, 0.06);
}

/* ========================================================
   TEXT AREA
   ======================================================== */

textarea {
    background: #ffffff !important;
    border: 2px solid #e8e2f5 !important;
    border-radius: 18px !important;
    color: #374151 !important;
    font-size: 15px !important;
    padding: 18px !important;
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.07);
    transition: all 0.25s ease;
}

textarea:focus {
    border-color: #a78bfa !important;
    box-shadow:
        0 0 0 3px rgba(167, 139, 250, 0.15),
        0 10px 30px rgba(99, 102, 241, 0.08);
}

/* ========================================================
   FILE UPLOADER
   ======================================================== */

[data-testid="stFileUploader"] {
    background:
        linear-gradient(
            135deg,
            #ffffff,
            #f8f5ff
        );

    border: 2px dashed #c4b5fd;
    border-radius: 20px;
    padding: 15px;

    box-shadow:
        0 8px 25px rgba(139, 92, 246, 0.07);

    transition: all 0.3s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: #8b5cf6;
    background: #faf8ff;

    box-shadow:
        0 12px 30px rgba(139, 92, 246, 0.13);
}

/* ========================================================
   SELECT BOX
   ======================================================== */

div[data-baseweb="select"] > div {
    background: #ffffff;
    border: 1px solid #ddd6fe;
    border-radius: 12px;
}

/* ========================================================
   PRIMARY BUTTON
   ======================================================== */

.stButton > button {
    width: 100%;

    background: linear-gradient(
        135deg,
        #a78bfa,
        #818cf8
    ) !important;

    color: white !important;

    border: none !important;
    border-radius: 15px;

    padding: 0.75rem 1.2rem;

    font-size: 16px;
    font-weight: 700;

    box-shadow:
        0 8px 20px rgba(129, 140, 248, 0.25);

    transition: all 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);

    box-shadow:
        0 12px 28px rgba(129, 140, 248, 0.35);
}

/* ========================================================
   HEADINGS
   ======================================================== */

h2, h3 {
    color: #51456f !important;
    font-weight: 750 !important;
}

/* ========================================================
   ORIGINAL + SUMMARY CARDS
   ======================================================== */

.content-card {
    background: rgba(255, 255, 255, 0.82);
    border: 1px solid #ebe6f5;
    border-radius: 22px;

    padding: 20px;

    min-height: 320px;

    box-shadow:
        0 10px 35px rgba(99, 102, 241, 0.07);
}

/* ========================================================
   SUMMARY SUCCESS BOX
   ======================================================== */

div[data-testid="stAlert"] {
    background:
        linear-gradient(
            135deg,
            #f3efff,
            #eef7ff
        ) !important;

    border: 1px solid #ddd6fe !important;

    border-radius: 20px !important;

    color: #4b5563 !important;

    padding: 20px !important;

    box-shadow:
        0 8px 25px rgba(139, 92, 246, 0.08);
}

/* ========================================================
   DOWNLOAD BUTTON
   ======================================================== */

.stDownloadButton > button {
    width: 100%;

    background: #ffffff !important;
    color: #6d5aa5 !important;

    border: 1px solid #c4b5fd !important;

    border-radius: 13px;

    font-weight: 700;

    transition: all 0.25s ease;
}

.stDownloadButton > button:hover {
    background: #f5f1ff !important;
    border-color: #a78bfa !important;
}

/* ========================================================
   INFO / WARNING
   ======================================================== */

div[data-testid="stAlert"][kind="info"] {
    background: #eef6ff !important;
    border: 1px solid #bfdbfe !important;
    border-radius: 15px !important;
}

div[data-testid="stAlert"][kind="warning"] {
    background: #fff7ed !important;
    border: 1px solid #fed7aa !important;
    border-radius: 15px !important;
}

/* ========================================================
   EXPANDER
   ======================================================== */

[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.7);
    border: 1px solid #e9e3f5;
    border-radius: 16px;
}

/* ========================================================
   PROGRESS BAR
   ======================================================== */

div[data-testid="stProgress"] > div {
    background: #ede9fe;
    border-radius: 20px;
    height: 9px;
}

div[data-testid="stProgress"] > div > div {
    background: linear-gradient(
        90deg,
        #a78bfa,
        #818cf8,
        #60a5fa
    );

    border-radius: 20px;
}

/* ========================================================
   CAPTIONS
   ======================================================== */

.stCaption {
    color: #8b8499 !important;
}

/* ========================================================
   MODEL BADGE
   ======================================================== */

.model-badge {
    display: inline-block;

    background: #f3efff;
    color: #7257a8;

    border: 1px solid #ddd6fe;

    padding: 6px 12px;

    border-radius: 30px;

    font-size: 12px;
    font-weight: 700;

    margin-bottom: 10px;
}

/* ========================================================
   FOOTER
   ======================================================== */

.footer {
    text-align: center;

    margin-top: 50px;
    padding-top: 20px;

    border-top: 1px solid #e8e2f2;

    color: #9b94a8;

    font-size: 13px;
}

/* ========================================================
   MOBILE
   ======================================================== */

@media (max-width: 768px) {

    h1 {
        font-size: 2.2rem !important;
    }

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

}

</style>
""", unsafe_allow_html=True)


# =========================================================
# MODELS
# =========================================================

MODELS = {
    "DistilBART (faster, lighter)": "sshleifer/distilbart-cnn-12-6",
    "BART Large (better quality)": "facebook/bart-large-cnn",
}

LENGTHS = {
    "Short": (30, 80),
    "Medium": (60, 150),
    "Long": (100, 250),
}

CHUNK_WORDS = 600


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource(
    show_spinner="Loading model (first run downloads it)..."
)
def load_model(model_id: str):

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    return tokenizer, model


# =========================================================
# TEXT PROCESSING
# =========================================================

def chunk_text(text: str, size: int = CHUNK_WORDS):

    words = text.split()

    return [
        " ".join(words[i:i + size])
        for i in range(0, len(words), size)
    ]


def summarize_chunk(
    text,
    tokenizer,
    model,
    min_len,
    max_len
):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    ids = model.generate(
        **inputs,
        min_length=min_len,
        max_length=max_len,
        num_beams=4,
        length_penalty=2.0,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )

    return tokenizer.decode(
        ids[0],
        skip_special_tokens=True
    )


def summarize(
    text,
    tokenizer,
    model,
    min_len,
    max_len
):

    chunks = chunk_text(text)

    progress = st.progress(
        0.0,
        text="✨ Summarizing your content..."
    )

    parts = []

    for i, chunk in enumerate(chunks, 1):

        words = len(chunk.split())

        parts.append(
            summarize_chunk(
                chunk,
                tokenizer,
                model,
                min(min_len, words // 2),
                max_len
            )
        )

        progress.progress(
            i / len(chunks),
            text=f"✨ Processing section {i}/{len(chunks)}"
        )

    progress.empty()

    combined = " ".join(parts)

    if (
        len(chunks) > 1
        and len(combined.split()) > CHUNK_WORDS
    ):

        combined = summarize_chunk(
            combined,
            tokenizer,
            model,
            min_len,
            max_len * 2
        )

    return combined


# =========================================================
# PDF READER
# =========================================================

def read_pdf(file):

    reader = PdfReader(file)

    return "\n".join(
        page.extract_text() or ""
        for page in reader.pages
    )


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙️ Settings")

model_name = st.sidebar.selectbox(
    "🤖 AI Model",
    list(MODELS.keys())
)

length = st.sidebar.radio(
    "📏 Summary Length",
    list(LENGTHS.keys()),
    index=1
)

st.sidebar.markdown(
    """
    <div class="sidebar-card">
        <b>📝 About this app</b><br><br>

        This application uses
        <b>BART-based AI models</b> to create
        abstractive summaries from your text
        and documents.<br><br>

        📄 TXT files<br>
        📕 PDF files<br>
        ✍️ Pasted text<br>
        🤖 Local AI processing<br>
        🔒 No API key required
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# MAIN HEADER
# =========================================================

st.title("📝 Text Summarizer")

st.markdown(
    """
    <div class="hero-text">
        Turn long articles, notes and documents into
        clear, concise summaries with AI.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# INPUT TYPE
# =========================================================

source = st.radio(
    "Choose your input",
    ["✍️ Paste text", "📁 Upload file"],
    horizontal=True
)

text = ""


# =========================================================
# PASTE TEXT
# =========================================================

if source == "✍️ Paste text":

    st.markdown(
        '<span class="model-badge">✍️ TEXT INPUT</span>',
        unsafe_allow_html=True
    )

    text = st.text_area(
        "Your text",
        height=300,
        placeholder=(
            "Paste an article, research paper, "
            "notes, report or any long text here..."
        ),
        label_visibility="collapsed"
    )


# =========================================================
# UPLOAD FILE
# =========================================================

else:

    st.markdown(
        '<span class="model-badge">📁 DOCUMENT INPUT</span>',
        unsafe_allow_html=True
    )

    file = st.file_uploader(
        "Upload your document",
        type=["txt", "pdf"],
        help="Supported formats: TXT and PDF"
    )

    if file:

        if file.name.lower().endswith(".pdf"):

            text = read_pdf(file)

        else:

            text = file.read().decode(
                "utf-8",
                errors="ignore"
            )

        with st.expander("👀 Preview extracted text"):

            st.write(
                text[:3000]
                + ("..." if len(text) > 3000 else "")
            )


# =========================================================
# SUMMARIZE BUTTON
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)

if st.button(
    "✨ Generate Summary",
    type="primary"
):

    word_count = len(text.split())

    if word_count < 50:

        st.warning(
            "⚠️ Please provide at least 50 words of text."
        )

    else:

        tokenizer, model = load_model(
            MODELS[model_name]
        )

        min_len, max_len = LENGTHS[length]

        summary = summarize(
            text,
            tokenizer,
            model,
            min_len,
            max_len
        )

        # =================================================
        # RESULTS
        # =================================================

        st.markdown("---")

        col1, col2 = st.columns(
            [1, 1],
            gap="large"
        )

        # -------------------------------------------------
        # ORIGINAL
        # -------------------------------------------------

        with col1:

            st.markdown(
                "### 📄 Original Text"
            )

            st.caption(
                f"{word_count:,} words"
            )

            st.markdown(
                '<div class="content-card">',
                unsafe_allow_html=True
            )

            st.write(
                text[:5000]
                + ("..." if len(text) > 5000 else "")
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )


        # -------------------------------------------------
        # SUMMARY
        # -------------------------------------------------

        with col2:

            st.markdown(
                "### ✨ AI Summary"
            )

            s_count = len(summary.split())

            reduction = max(
                0,
                100 - (s_count * 100 // word_count)
            )

            st.caption(
                f"{s_count:,} words • "
                f"{reduction}% shorter"
            )

            st.markdown(
                '<div class="content-card">',
                unsafe_allow_html=True
            )

            st.success(summary)

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

            st.markdown("<br>", unsafe_allow_html=True)

            st.download_button(
                "⬇️ Download Summary",
                summary,
                "summary.txt",
                "text/plain"
            )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        ✨ AI Text Summarizer
        &nbsp;•&nbsp;
        🤖 Powered by Hugging Face Transformers
        &nbsp;•&nbsp;
        🔒 Runs Locally
    </div>
    """,
    unsafe_allow_html=True
)

