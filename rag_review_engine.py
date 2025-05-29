# rag_review_engine.py
# =============================
# Object-oriented module to encapsulate RAG logic for cybersecurity document review.

import os
from dotenv import load_dotenv
import httpx
from openai import OpenAI
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class RAGReviewEngine:
    """
    Engine to load a PDF, build a TF-IDF index, and review compliance prompts via OpenAI.
    """
    def __init__(self, ssl_verify: bool = True):
        # Load API key
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY not found in .env. Ensure it's set without quotes."
            )
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            http_client=httpx.Client(verify=ssl_verify)
        )
        # Placeholders for index
        self.chunks = []
        self.vectorizer = None
        self.tfidf_matrix = None

    def extract_text(self, path_or_file) -> str:
        """
        Read all pages of the PDF and return concatenated text.
        """
        reader = PdfReader(path_or_file)
        return "\n\n".join(
            page.extract_text() or "" for page in reader.pages
        )

    def split_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
        """
        Split long text into overlapping chunks.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )
        self.chunks = splitter.split_text(text)
        return self.chunks

    def build_index(self) -> None:
        """
        Create a TF-IDF vectorizer over the stored chunks.
        """
        if not self.chunks:
            raise ValueError("No chunks to index. Call split_chunks() first.")
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)

    def review(self, prompt: str, top_k: int = 3) -> tuple[str | None, list[str]]:
        """
        Retrieve top_k relevant chunks by TF-IDF and send to OpenAI for QA.
        Returns (answer_text, raw_chunks).
        """
        if self.vectorizer is None or self.tfidf_matrix is None:
            raise ValueError("Index not built. Call build_index() first.")
        # Compute similarity and select chunks
        qv = self.vectorizer.transform([prompt])
        sims = cosine_similarity(qv, self.tfidf_matrix).flatten()
        idxs = sims.argsort()[-top_k:][::-1]
        selected = [self.chunks[i] for i in idxs]

        # Build conversation
        system_msg = {"role": "system", "content": "You are a cybersecurity professional with 10 years of experience in cybersecurity policies and ISO compliance."}
        user_msg = {
            "role": "user",
            "content": (
                f"Context:\n{'\n\n'.join(selected)}\n\n"
                f"Question: {prompt}\n"
                "Please limit your answer to no more than 200 words, "
                "conclude with a brief evaluation summary at the end, "
                "and append a line in the format `Score: Fully covered|Partially covered|Not covered`."
            )
        }

        try:
            resp = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[system_msg, user_msg],
                max_tokens=256,
                temperature=0
            )
            answer = resp.choices[0].message.content
        except Exception as e:
            # Return raw chunks on error
            print(f"OpenAI request error: {e}")
            answer = None
        return answer, selected

    def review_all(self, prompts: dict[str, str], top_k: int = 3) -> dict[str, dict]:
        """
        Iterate over multiple prompts, returning a dict mapping prompt name to answers and chunks.
        """
        results = {}
        for name, prompt in prompts.items():
            ans, raw = self.review(prompt, top_k=top_k)
            results[name] = {"answer": ans, "chunks": raw}
        return results

# Example usage in a Streamlit app (app.py):
# from rag_review_engine import RAGReviewEngine
# engine = RAGReviewEngine(ssl_verify=False)
# text = engine.extract_text(uploaded_file)
# engine.split_chunks(text)
# engine.build_index()
# results = engine.review_all({
#     "Security role definitions": "...",
#     "Network segmentation": "..."
# })
# for name, data in results.items(): st.write(name, data['answer'])
