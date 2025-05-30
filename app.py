import glob
import yaml
import streamlit as st
import pandas as pd
import io
from rag_review_engine import RAGReviewEngine

# Load all standards from templates folder
STANDARDS = {}
for path in glob.glob("templates/*.yml"):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
        name = cfg.get('name')
        prompts = {p['id']: p['template'] for p in cfg.get('prompts', [])}
        STANDARDS[name] = prompts

# Streamlit page config
st.set_page_config(page_title="Cybersecurity Document Review (RAG)", layout="wide")
st.title("Cybersecurity Document Review (RAG)")

# Sidebar: standard and area selection
st.sidebar.header("Evaluation Settings")

# Select exactly one standard
selected_standard = st.sidebar.selectbox(
    "Choose standard", 
    options=list(STANDARDS.keys()),
    index=0
)

# Load prompts for the chosen standard
available_prompts = STANDARDS[selected_standard]

# Select areas within the chosen standard
selected_areas = st.sidebar.multiselect(
    "Choose compliance areas", 
    options=list(available_prompts.keys()),
    default=list(available_prompts.keys())
)

# Initialize RAG engine
engine = RAGReviewEngine(ssl_verify=False)

# File upload in main area
uploaded_file = st.file_uploader("Upload a cybersecurity PDF", type=["pdf"] )
# Clear previous results if a new file is uploaded
if uploaded_file:
    if st.session_state.get('last_uploaded') != uploaded_file.name:
        st.session_state.pop('df', None)
        st.session_state['last_uploaded'] = uploaded_file.name

if uploaded_file:
    st.info("Extracting and indexing document…")
    text = engine.extract_text(uploaded_file)
    engine.split_chunks(text)
    engine.build_index()
    st.success("Document indexed successfully!")

    if not selected_areas:
        st.warning("Select at least one compliance area in the sidebar.")
    else:
        if st.sidebar.button("Run Compliance Review"):
            # Build prompt dict for selected areas
            prompts_to_run = {area: available_prompts[area] for area in selected_areas}
            results = engine.review_all(prompts_to_run)

            # Parse and tabulate results
            rows = []
            for area, data in results.items():
                answer = data.get("answer", "") or ""
                lines = [ln.strip() for ln in answer.splitlines() if ln.strip()]

                # Extract score if present
                score = ""
                for ln in reversed(lines):
                    if ln.lower().startswith("score:"):
                        score = ln.split(":", 1)[1].strip()
                        break

                # Filter out score line
                content_lines = [ln for ln in lines if not ln.lower().startswith("score:")]
                conclusion = content_lines[-1] if content_lines else ""
                analysis = "\n".join(content_lines[:-1]) if len(content_lines) > 1 else ""

                rows.append({
                    "Compliance area": area,
                    "Analysis": analysis,
                    "Score": score
                })

            # Store results in session state
            st.session_state['df'] = pd.DataFrame(rows)

# Display results and download button if available
if 'df' in st.session_state:
    df = st.session_state['df']
    st.subheader("Review Results")
    st.dataframe(df, use_container_width=True)

    # Generate Excel download
    towrite = io.BytesIO()
    df.to_excel(towrite, index=False, sheet_name="Results")
    towrite.seek(0)

    st.download_button(
        label="Download results as Excel",
        data=towrite,
        file_name="review_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
