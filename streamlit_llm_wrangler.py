import streamlit as st
import pandas as pd
import os
import tempfile
import subprocess
import json
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF

st.set_page_config(page_title="AI Data Wrangler", layout="wide")
st.title("🤖 AI-Powered Data Wrangler (Offline + Free)")

llm_model = st.selectbox("Select LLM for Data Wrangling", ["mistral (Offline via Ollama)", "gpt-4 (Online via OpenAI)"])

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

    st.subheader("📎 Outlier Detection (Before Cleaning)")
    numeric_cols = df.select_dtypes(include='number')
    if not numeric_cols.empty:
        melted = numeric_cols.melt(var_name='variable', value_name='value')
        if melted.empty:
            st.info("No numeric data for outlier visualization (melted data empty).")
        else:
            fig4, ax4 = plt.subplots(figsize=(12, 6))
            sns.boxplot(y="variable", x="value", data=melted, ax=ax4)
            st.pyplot(fig4)
    else:
        st.info("No numeric data for outlier visualization.")

    sample_csv = df.sample(min(500, len(df))).to_csv(index=False)
    sample_file = os.path.join(tempfile.gettempdir(), "sample.csv")
    with open(sample_file, "w") as f:
        f.write(sample_csv)

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
    with st.spinner("Running selected LLM..."):
        if llm_model.startswith("mistral"):
            result = subprocess.run([
                "ollama", "run", "mistral", prompt
            ], capture_output=True, text=True)
            code_output = result.stdout if result.returncode == 0 else None
        elif llm_model.startswith("gpt-4"):
            import openai
            openai.api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else ""
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2
                )
                code_output = response.choices[0].message.content.strip()
            except Exception as e:
                st.error(f"GPT-4 request failed: {e}")
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

                st.subheader("📎 Outlier Detection (After Cleaning)")
                numeric_clean = cleaned_df.select_dtypes(include='number')
                if not numeric_clean.empty:
                    melted_clean = numeric_clean.melt(var_name='variable', value_name='value')
                    if melted_clean.empty:
                        st.info("No numeric data for outlier visualization (melted data empty).")
                    else:
                        fig5, ax5 = plt.subplots(figsize=(12, 6))
                        sns.boxplot(y="variable", x="value", data=melted_clean, ax=ax5)
                        st.pyplot(fig5)
                else:
                    st.info("No numeric data for outlier visualization.")

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
                    st.download_button("📥 Download PDF Summary Report", pdf_file.read(), "summary_report.pdf")

                csv_cleaned = cleaned_df.to_csv(index=False)
                st.download_button("Download Cleaned Data CSV", csv_cleaned, "cleaned_data.csv")
            except Exception as e:
                st.error(f"⚠️ Failed to apply cleaning code: {e}")

    os.remove(file_path)
