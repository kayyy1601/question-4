import streamlit as st
import nltk
from PyPDF2 import PdfReader

# Ensure punkt is available
nltk.download("punkt", quiet=True)

st.set_page_config(page_title="PDF Sentence Chunker (NLTK)", layout="wide")

st.title("PDF Sentence Chunker Demo")

st.write(
    "Upload a PDF file, extract text, and split it into sentences using "
    "NLTK's `sent_tokenize`."
)

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    try:
        reader = PdfReader(uploaded_file)
        pages_text = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            pages_text.append(page_text)

        full_text = " ".join(pages_text).strip()

        st.subheader("Basic info")
        st.write(f"Number of pages: **{len(reader.pages)}**")
        st.write(f"Total characters extracted: **{len(full_text)}**")

        if not full_text:
            st.warning("No text could be extracted from this PDF.")
        else:
            sentences = nltk.sent_tokenize(full_text)
            st.success(f"Number of detected sentences: {len(sentences)}")

            # Controls to view sentences in a range
            start_idx = st.number_input(
                "Show sentences starting from index",
                min_value=0,
                max_value=max(len(sentences) - 1, 0),
                value=0,
                step=1,
            )
            end_idx = st.number_input(
                "Up to (exclusive)",
                min_value=start_idx + 1 if len(sentences) > 0 else 1,
                max_value=len(sentences),
                value=min(start_idx + 10, len(sentences)),
                step=1,
            )

            st.subheader(f"Sentences [{start_idx} : {end_idx})")
            for i in range(start_idx, end_idx):
                st.markdown(f"**{i}**. {sentences[i]}")

            with st.expander("Show raw extracted text (first 2000 characters)"):
                st.text(full_text[:2000])

            # -------------------------------
            # Step 3: Show sample indices 58 to 68
            # -------------------------------
            st.subheader("Step 3: Sample Sentences (58 to 68)")
            sample_start = 58
            sample_end = 68

            if len(sentences) < sample_end:
                st.warning(
                    f"This PDF has only {len(sentences)} sentences, so indices "
                    f"{sample_start} to {sample_end} are not fully available."
                )
            else:
                for i in range(sample_start, sample_end):
                    st.markdown(f"**{i}**. {sentences[i]}")

            # -------------------------------
            # Step 4: Apply NLTK semantic chunking on Step 3 text
            # -------------------------------
            st.subheader("Step 4: Semantic Sentence Chunking on Step 3 Text")

            if len(sentences) < sample_end:
                st.info("Skipping Step 4 because sample sentences are not available.")
            else:
                step3_text = " ".join(sentences[sample_start:sample_end])
                st.write(" Combined text from sentences 58 to 68:")
                st.write(step3_text)

                chunked_sentences = nltk.sent_tokenize(step3_text)
                st.success(f"Chunked sentence count in Step 4: {len(chunked_sentences)}")

                st.write(" Sentences after semantic chunking (Step 4):")
                for idx, sent in enumerate(chunked_sentences):
                    st.markdown(f"**{idx}**. {sent}")

    except Exception as e:
        st.error(f"Error reading PDF: {e}")
else:
    st.info("Please upload a PDF to begin.")

