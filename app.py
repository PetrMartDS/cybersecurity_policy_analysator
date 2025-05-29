import json
import streamlit as st
import pandas as pd
import io
from rag_review_engine import RAGReviewEngine

# Load compliance prompts from JSON config
with open("prompts_config.json", "r") as f:
    COMPLIANCE_PROMPTS = json.load(f)

st.set_page_config(page_title="Cybersecurity Doc Review", layout="wide")
st.title("Cybersecurity Document Review (RAG)")

# Initialize engine (disable SSL check locally; remove verify=False in prod)
engine = RAGReviewEngine(
    ssl_verify=False
)

# File upload
uploaded_file = st.file_uploader("Upload a cybersecurity PDF", type=["pdf"])

if uploaded_file:
    st.info("Extracting and indexing document…")
    text = engine.extract_text(uploaded_file)
    engine.split_chunks(text)
    engine.build_index()
    st.success("Document indexed successfully!")

    # Prompt selection (all pre-selected)
    st.subheader("Select components to review")
    selected = st.multiselect(
        "Choose compliance areas:",
        list(COMPLIANCE_PROMPTS.keys()),
        default=list(COMPLIANCE_PROMPTS.keys())
    )

    if st.button("Run Compliance Review"):
        if not selected:
            st.warning("Select at least one component to review.")
        else:
            results = engine.review_all({k: COMPLIANCE_PROMPTS[k] for k in selected})
            rows = []
            for area, data in results.items():
                answer = data["answer"] or ""
                lines = [ln.strip() for ln in answer.split("\n") if ln.strip()]

                # Extract score
                score = ""
                for ln in reversed(lines):
                    if ln.lower().startswith("score:"):
                        score = ln.split(":", 1)[1].strip()
                        break

                # Remove score line from content
                content_lines = [ln for ln in lines if not ln.lower().startswith("score:")]

                # Last line is conclusion, rest is analysis
                if content_lines:
                    conclusion = content_lines[-1]
                    analysis = "\n".join(content_lines[:-1])
                else:
                    conclusion = ""
                    analysis = ""

                rows.append({
                    "Compliance area": area,
                    "Analysis": analysis,
                    # "Conclusion": conclusion,
                    "Score": score
                })

            df = pd.DataFrame(rows).reset_index(drop=True)

            st.subheader("Review Results")
            st.dataframe(df)
