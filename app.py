import streamlit as st
import os
from ocr_module import process_image
from extractor import extract_fields, check_ollama_available
from db import init_db, save_invoice, query_invoices, get_categories_in_use

st.set_page_config(page_title="InvoxAI - Local Invoice Extraction", layout="wide")
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
    .hiw-banner p { color: #c9cdd6; line-height: 1.6; margin-bottom: 0.8rem; }
    .hiw-steps { color: #9aa1b0; font-size: 0.9rem; line-height: 1.8; }
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
if "images" not in st.session_state:
    st.session_state.images = {}

# ---------- Header ----------
st.title("InvoxAI")
st.markdown('<span class="badge-local" style="font-size:0.9rem;">LOCAL AND PRIVATE BY DEFAULT</span>', unsafe_allow_html=True)

st.markdown("""
<div class="hiw-banner">
<h3>How it works</h3>
<div class="hiw-steps">
1. Upload an invoice or receipt image<br>
2. OCR reads the text on your machine<br>
3. AI converts the text into structured fields (vendor, date, amount, category)<br>
4. Invalid output is auto corrected, not guessed with hardcoded rules<br>
5. Review, edit, and save
</div>
<p style="margin-top:1rem; margin-bottom:0;">
<b>Local mode:</b> everything above runs on your machine, nothing is sent online.<br>
<b>Cloud mode:</b> pick "Mistral API" in the sidebar and paste your API key there. Only the extracted invoice text is sent to Mistral for that step.
</p>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar: backend selector ----------
st.sidebar.header("Extraction Backend")
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
    st.sidebar.markdown('<span class="badge-local">LOCAL</span>', unsafe_allow_html=True)
    is_up, models = check_ollama_available()
    if is_up:
        st.sidebar.success(f"Ollama server detected ({len(models)} model(s) available)")
        if models:
            ollama_model = st.sidebar.selectbox("Model", models)
    else:
        st.sidebar.error("Ollama not detected at localhost:11434. Run 'ollama serve' and 'ollama pull mistral'.")
else:
    backend = "mistral_api"
    st.sidebar.markdown('<span class="badge-cloud">CLOUD</span>', unsafe_allow_html=True)
    mistral_api_key = st.sidebar.text_input("Mistral API Key", type="password")
    mistral_model = st.sidebar.selectbox("Model", ["mistral-small-latest", "mistral-large-latest"])
    if not mistral_api_key:
        st.sidebar.warning("Enter an API key to use this backend.")

st.sidebar.divider()
st.sidebar.caption("Local mode never sends invoice data over the network. Cloud mode sends OCR text to Mistral's API.")

if st.sidebar.button("Clear all uploaded images"):
    st.session_state.images = {}
    st.rerun()

# ---------- Main tabs ----------
tab1, tab2 = st.tabs(["Upload and Process", "Search and Export"])

with tab1:
    uploaded_files = st.file_uploader(
        "Upload invoice images. You can select multiple files at once.",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

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
                    "result": None,
                    "attempt_log": None,
                    "saved": False,
                    "last_error": None,
                }

    if not st.session_state.images:
        st.info("Upload one or more invoice images to get started.")

    for fname, state in list(st.session_state.images.items()):
        with st.container():
            st.markdown('<div class="img-card">', unsafe_allow_html=True)
            col_img, col_main = st.columns([1, 2.5])

            with col_img:
                st.image(state["file_path"], use_container_width=True)
                st.caption(fname)
                if state["saved"]:
                    st.markdown('<span class="badge-saved">SAVED</span>', unsafe_allow_html=True)
                if st.button("Remove", key=f"remove_{fname}"):
                    del st.session_state.images[fname]
                    st.rerun()

            with col_main:
                if state["raw_text"] is None:
                    with st.spinner(f"Running OCR on {fname}"):
                        state["raw_text"] = process_image(state["file_path"])

                with st.expander("Raw OCR text", expanded=False):
                    st.text_area("raw", state["raw_text"], height=120, label_visibility="collapsed", key=f"raw_{fname}")

                extract_disabled = (backend == "mistral_api" and not mistral_api_key)

                if state["result"] is None:
                    if state.get("last_error"):
                        st.error(state["last_error"])

                    if st.button("Extract structured data", key=f"extract_{fname}", type="primary", disabled=extract_disabled):
                        spinner_msg = f"Extracting {fname} via {backend_choice}. Local models can take 20 to 90 seconds depending on your hardware, please wait."
                        with st.spinner(spinner_msg):
                            result, attempt_log = extract_fields(
                                state["raw_text"],
                                backend=backend,
                                ollama_model=ollama_model,
                                mistral_api_key=mistral_api_key,
                                mistral_model=mistral_model,
                            )
                        state["attempt_log"] = attempt_log
                        state["last_error"] = None
                        if result is not None:
                            state["result"] = result.model_dump()
                        else:
                            last_status = attempt_log[-1]["status"] if attempt_log else "unknown"
                            if last_status == "timeout":
                                state["last_error"] = "The model took too long to respond and the request timed out. Try a smaller/faster model, or check Ollama is running properly."
                            elif last_status == "connection_error":
                                state["last_error"] = "Could not reach Ollama. Make sure 'ollama serve' is running."
                            else:
                                state["last_error"] = "Extraction failed after all repair attempts. See the attempt log below for details."
                        st.rerun()
                else:
                    result = state["result"]

                    with st.expander(f"Extraction attempt log ({len(state['attempt_log'])} attempt(s))"):
                        st.json(state["attempt_log"])

                    conf = result["confidence"]
                    conf_label = "High" if conf >= 0.8 else ("Medium" if conf >= 0.5 else "Low")
                    st.markdown(f"**Confidence:** {conf_label} ({conf:.2f})")

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
                            if st.button("Save to database", key=f"save_{fname}"):
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
                        if st.button("Re-extract", key=f"reextract_{fname}"):
                            state["result"] = None
                            state["attempt_log"] = None
                            state["saved"] = False
                            state["last_error"] = None
                            st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.subheader("Search Invoices")
    search_text = st.text_input("Search vendor or description", placeholder="e.g. Amazon, office supplies, electricity")

    with st.expander("More filters"):
        c1, c2, c3 = st.columns(3)
        with c1:
            existing_categories = get_categories_in_use()
            category_filter = st.selectbox("Category", ["All"] + existing_categories)
        with c2:
            date_from = st.text_input("From date (YYYY-MM-DD)")
            date_to = st.text_input("To date (YYYY-MM-DD)")
        with c3:
            amount_min = st.number_input("Min amount", value=0.0, min_value=0.0)
            amount_max = st.number_input("Max amount", value=0.0, min_value=0.0, help="Leave at 0 to skip this filter")
        sort_col = st.selectbox("Sort by", ["created_at", "invoice_date", "total_amount", "vendor_name"], index=0)
        sort_desc = st.checkbox("Newest / highest first", value=True)

    total_saved = len(query_invoices())
    st.caption(f"{total_saved} invoice(s) saved in total")

    df = query_invoices(
        search_text=search_text or None,
        category=None if category_filter == "All" else category_filter,
        date_from=date_from or None,
        date_to=date_to or None,
        amount_min=amount_min if amount_min > 0 else None,
        amount_max=amount_max if amount_max > 0 else None,
        sort_by=sort_col,
        sort_desc=sort_desc,
    )

    if df.empty and total_saved == 0:
        st.info("No invoices saved yet. Extract and save one from the Upload tab first.")
    elif df.empty:
        st.info("No invoices match your current filters. Try clearing them or checking spelling.")
    else:
        st.caption(f"{len(df)} matching this search")
        display_df = df[["vendor_name", "invoice_date", "total_amount", "tax_amount", "category", "description", "confidence"]]
        st.dataframe(display_df, use_container_width=True)

        st.subheader("Spending Summary")
        col_a, col_b = st.columns(2)
        with col_a:
            st.bar_chart(df.groupby("category")["total_amount"].sum())
        with col_b:
            st.metric("Total spend (filtered)", f"{df['total_amount'].sum():.2f}")
            st.metric("Average per invoice", f"{df['total_amount'].mean():.2f}")

        csv = display_df.to_csv(index=False).encode("utf-8")
        st.download_button("Export to CSV", csv, "invoices_export.csv", "text/csv")