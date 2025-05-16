import streamlit as st
import pandas as pd
import os
import tempfile
import json
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF

st.set_page_config(page_title="AI Data Wrangler", layout="wide")
st.title("🤖 AI-Powered Data Wrangler (Online LLM Only)")

# Force model to GPT-3.5 only for free-tier compatibility
llm_model = "gpt-3.5-turbo"

uploaded_file = st.file_uploader("Upload your raw data file (CSV, Excel, JSON, Parquet)", type=["csv", "xlsx", "xls", "json", "parquet"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    file_type = uploaded_file.name.split(".")[-1].lower()
    if file_type in ["xlsx", "xls"]:
        df = pd.read_excel(file_path)
    elif file_type == "json":
        df = pd.read_json(file_path)
    elif file_type == "parquet":
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path)

    st.subheader("🔍 Preview of Uploaded Data")
    st.dataframe(df.head(100))

    st.subheader("📊 Column Type Distribution (Before)")
    st.bar_chart(df.dtypes.value_counts())

    st.subheader("📈 Missing Value Heatmap (Before Cleaning)")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, ax=ax)
    st.pyplot(fig)

    sample_csv = df.sample(min(500, len(df))).to_csv(index=False)
    prompt = f"""
You are a data wrangling expert. A user has uploaded a dataset. Your job is to:
- Analyze missing values
- Detect incorrect data types
- Identify inconsistent categorical data
- Suggest and output Python pandas code to clean the dataset named 'df'
- Suggest meaningful column renaming if original column names are ambiguous

Rules:
- DO NOT use os, sys, subprocess or any unsafe imports
- Return ONLY the cleaning logic using pandas

Here is the CSV sample:
{sample_csv[:4000]}

Respond ONLY with executable Python code. Do not include explanations.
"""

    st.subheader("🧠 LLM Analysis & Cleaning Plan")
    with st.spinner("Running OpenAI LLM..."):
        import openai
        openai.api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else ""
        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            code_output = response.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"OpenAI request failed: {e}")
            code_output = None

    if not code_output:
        st.error("LLM failed to respond. Check your setup and try again.")
    else:
        st.code(code_output, language="python")
        if any(unsafe in code_output.lower() for unsafe in ["import os", "subprocess", "sys", "eval", "exec"]):
            st.error("⚠️ Unsafe code detected. Execution blocked.")
        else:
            try:
                local_vars = {"df": df.copy()}
                exec(code_output, {}, local_vars)
                cleaned_df = local_vars["df"]

                st.subheader("🧼 Cleaned Data Preview")
                st.dataframe(cleaned_df.head(100))

                st.subheader("📊 Column Type Distribution (After)")
                st.bar_chart(cleaned_df.dtypes.value_counts())

                st.subheader("📈 Missing Value Heatmap (After Cleaning)")
                fig2, ax2 = plt.subplots(figsize=(12, 6))
                sns.heatmap(cleaned_df.isnull(), cbar=False, ax=ax2)
                st.pyplot(fig2)

                st.subheader("📉 Correlation Matrix")
                numeric_df = cleaned_df.select_dtypes(include=["number"])
                if not numeric_df.empty:
                    fig3, ax3 = plt.subplots(figsize=(10, 8))
                    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax3)
                    st.pyplot(fig3)
                else:
                    st.info("No numeric data to show correlation matrix.")

                st.subheader("📜 Summary Report PDF")
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.multi_cell(0, 10, "Summary Report\n\n")
                pdf.multi_cell(0, 10, f"Original Shape: {df.shape}\n")
                pdf.multi_cell(0, 10, f"Cleaned Shape: {cleaned_df.shape}\n")
                pdf.multi_cell(0, 10, f"Original Columns: {list(df.columns)}\n")
                pdf.multi_cell(0, 10, f"Cleaned Columns: {list(cleaned_df.columns)}\n")
                pdf.multi_cell(0, 10, f"Cleaning Code:\n{code_output[:1000]}...")

                pdf_path = os.path.join(tempfile.gettempdir(), "report.pdf")
                pdf.output(pdf_path)
                with open(pdf_path, "rb") as pdf_file:
                    st.download_button("📅 Download PDF Summary Report", pdf_file.read(), "summary_report.pdf")

                csv_cleaned = cleaned_df.to_csv(index=False)
                st.download_button("Download Cleaned Data CSV", csv_cleaned, "cleaned_data.csv")
            except Exception as e:
                st.error(f"⚠️ Failed to apply cleaning code: {e}")

    os.remove(file_path)
