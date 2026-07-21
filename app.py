import streamlit as st
import os
from ocr_module import process_image
from extractor import extract_fields, check_ollama_available
from db import init_db, save_invoice, query_invoices

st.set_page_config(page_title="InvoxAI - Local Invoice Extraction", layout="wide", page_icon="📄")
init_db()

# ---------- Styling ----------
st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; max-width: 1200px; }
    .hiw-banner {
        background: linear-gradient(135deg, #1a1f2e 0%, #232938 100%);
        border: 1px solid #2d3548;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
    }
    .hiw-banner h3 { margin-top: 0; color: #e8eaed; }
    .hiw-step {
        display: inline-block;
        background: #2d3548;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        margin: 0.3rem 0.3rem 0.3rem 0;
        font-size: 0.85rem;
        color: #c9cdd6;
    }
    .hiw-step b { color: #7dd3fc; }
    .badge-local {
        background: #0f3d2e; color: #4ade80; padding: 2px 10px;
        border-radius: 12px; font-size: 0.75rem; font-weight: 600;
    }
    .badge-cloud {
        background: #3d2e0f; color: #facc15; padding: 2px 10px;
        border-radius: 12px; font-size: 0.75rem; font-weight: 600;
    }
    .badge-saved {
        background: #0f3d2e; color: #4ade80; padding: 3px 12px;
        border-radius: 12px; font-size: 0.8rem; font-weight: 600;
    }
    .img-card {
        border: 1px solid #2d3548;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1.2rem;
        background: #161a25;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Session state init ----------
# Keyed by filename -> dict holding everything about that image across reruns.
# This is what keeps results visible instead of vanishing on every interaction.
if "images" not in st.session_state:
    st.session_state.images = {}

# ---------- Header ----------
st.title("📄 InvoxAI")
st.caption("Local-first invoice digitization — your data never has to leave your machine.")

st.markdown("""
<div class="hiw-banner">
<h3>⚙️ How it works</h3>
<span class="hiw-step"><b>1. Upload</b> — drop one or more invoice/receipt images</span>
<span class="hiw-step"><b>2. OCR</b> — text extracted locally</span>
<span class="hiw-step"><b>3. Extract</b> — LLM converts text into strict structured JSON</span>
<span class="hiw-step"><b>4. Validate + Repair</b> — schema-invalid output auto-corrected, zero regex fallback</span>
<span class="hiw-step"><b>5. Review & Save</b> — confidence-scored fields, edit and save each independently</span>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar: backend selector ----------
st.sidebar.header("🧠 Extraction Backend")
backend_choice = st.sidebar.radio(
    "Choose how invoices are processed",
    ["Local Ollama (private, offline)", "Mistral API (cloud, requires key)"],
    index=0,
)

ollama_model = "mistral"
mistral_api_key = None
mistral_model = "mistral-small-latest"

if backend_choice.startswith("Local"):
    backend = "ollama"
    st.sidebar.markdown('<span class="badge-local">100% LOCAL</span>', unsafe_allow_html=True)
    is_up, models = check_ollama_available()
    if is_up:
        st.sidebar.success(f"Ollama server detected ({len(models)} model(s) available)")
        if models:
            ollama_model = st.sidebar.selectbox("Model", models)
    else:
        st.sidebar.error("Ollama not detected at localhost:11434. Run `ollama serve` and `ollama pull mistral`.")
else:
    backend = "mistral_api"
    st.sidebar.markdown('<span class="badge-cloud">CLOUD API</span>', unsafe_allow_html=True)
    mistral_api_key = st.sidebar.text_input("Mistral API Key", type="password")
    mistral_model = st.sidebar.selectbox("Model", ["mistral-small-latest", "mistral-large-latest"])
    if not mistral_api_key:
        st.sidebar.warning("Enter an API key to use this backend.")

st.sidebar.divider()
st.sidebar.caption("Local mode never sends invoice data over the network. Cloud mode sends OCR text to Mistral's API.")

if st.sidebar.button("🗑️ Clear all uploaded images"):
    st.session_state.images = {}
    st.rerun()

# ---------- Main tabs ----------
tab1, tab2 = st.tabs(["📤 Upload & Process", "🔍 Search & Export"])

with tab1:
    uploaded_files = st.file_uploader(
        "Upload invoice images (you can select multiple)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    # Register newly uploaded files into session state (once each, keyed by name)
    if uploaded_files:
        os.makedirs("uploads", exist_ok=True)
        for f in uploaded_files:
            if f.name not in st.session_state.images:
                file_path = os.path.join("uploads", f.name)
                with open(file_path, "wb") as out:
                    out.write(f.getbuffer())
                st.session_state.images[f.name] = {
                    "file_path": file_path,
                    "raw_text": None,
                    "result": None,       # dict of extracted fields once extracted
                    "attempt_log": None,
                    "saved": False,
                }

    if not st.session_state.images:
        st.info("Upload one or more invoice images to get started.")

    # Render one card per image, independently — extracting/saving one
    # doesn't touch the others, and everything survives reruns.
    for fname, state in list(st.session_state.images.items()):
        with st.container():
            st.markdown('<div class="img-card">', unsafe_allow_html=True)
            col_img, col_main = st.columns([1, 2.5])

            with col_img:
                st.image(state["file_path"], use_container_width=True)
                st.caption(fname)
                if state["saved"]:
                    st.markdown('<span class="badge-saved">✅ SAVED</span>', unsafe_allow_html=True)
                if st.button("Remove", key=f"remove_{fname}"):
                    del st.session_state.images[fname]
                    st.rerun()

            with col_main:
                # Run OCR once, cache in state
                if state["raw_text"] is None:
                    with st.spinner(f"Running OCR on {fname}..."):
                        state["raw_text"] = process_image(state["file_path"])

                with st.expander("Raw OCR text", expanded=False):
                    st.text_area("raw", state["raw_text"], height=120, label_visibility="collapsed", key=f"raw_{fname}")

                extract_disabled = (backend == "mistral_api" and not mistral_api_key)

                if state["result"] is None:
                    if st.button("Extract structured data", key=f"extract_{fname}", type="primary", disabled=extract_disabled):
                        with st.spinner(f"Extracting {fname} via {backend_choice}..."):
                            result, attempt_log = extract_fields(
                                state["raw_text"],
                                backend=backend,
                                ollama_model=ollama_model,
                                mistral_api_key=mistral_api_key,
                                mistral_model=mistral_model,
                            )
                        state["attempt_log"] = attempt_log
                        if result is not None:
                            state["result"] = result.model_dump()
                        st.rerun()
                else:
                    result = state["result"]

                    with st.expander(f"Extraction attempt log ({len(state['attempt_log'])} attempt(s))"):
                        st.json(state["attempt_log"])

                    conf = result["confidence"]
                    conf_color = "🟢" if conf >= 0.8 else ("🟡" if conf >= 0.5 else "🔴")
                    st.markdown(f"**Confidence:** {conf_color} {conf:.2f}")

                    c1, c2 = st.columns(2)
                    with c1:
                        vendor = st.text_input("Vendor Name", result["vendor_name"], key=f"vendor_{fname}")
                        inv_date = st.text_input("Invoice Date (YYYY-MM-DD)", result["invoice_date"] or "", key=f"date_{fname}")
                        total = st.number_input("Total Amount", value=result["total_amount"], key=f"total_{fname}")
                    with c2:
                        tax = st.number_input("Tax Amount", value=result["tax_amount"] or 0.0, key=f"tax_{fname}")
                        categories = ["Office", "IT", "Travel", "Food", "Utilities", "Other"]
                        category = st.selectbox("Category", categories, index=categories.index(result["category"]), key=f"cat_{fname}")
                        description = st.text_input("Description", result["description"], key=f"desc_{fname}")

                    col_save, col_reextract = st.columns(2)
                    with col_save:
                        if not state["saved"]:
                            if st.button("💾 Save to database", key=f"save_{fname}"):
                                save_invoice({
                                    "vendor_name": vendor, "invoice_date": inv_date or None,
                                    "total_amount": total, "tax_amount": tax,
                                    "category": category, "description": description,
                                    "confidence": conf
                                }, state["raw_text"], state["file_path"])
                                state["saved"] = True
                                st.rerun()
                        else:
                            st.success("Saved")
                    with col_reextract:
                        if st.button("🔄 Re-extract", key=f"reextract_{fname}"):
                            state["result"] = None
                            state["attempt_log"] = None
                            state["saved"] = False
                            st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.subheader("Search Invoices")
    c1, c2 = st.columns(2)
    with c1:
        vendor_filter = st.text_input("Filter by vendor")
        category_filter = st.selectbox("Filter by category", ["All", "Office", "IT", "Travel", "Food", "Utilities", "Other"])
    with c2:
        date_from = st.text_input("From date (YYYY-MM-DD)")
        date_to = st.text_input("To date (YYYY-MM-DD)")

    df = query_invoices(
        vendor=vendor_filter or None,
        category=None if category_filter == "All" else category_filter,
        date_from=date_from or None,
        date_to=date_to or None,
    )
    st.dataframe(df, use_container_width=True)

    if not df.empty:
        st.subheader("Spending Summary")
        st.bar_chart(df.groupby("category")["total_amount"].sum())

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Export to CSV", csv, "invoices_export.csv", "text/csv")