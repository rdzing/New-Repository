import streamlit as st
import os
from src.main import process_omr  # your main function

st.title("OMR Sheet Evaluator")
uploaded_files = st.file_uploader("Upload OMR images", accept_multiple_files=True, type=["png", "jpg", "jpeg"])

if uploaded_files:
    results = []
    for file in uploaded_files:
        with open(file.name, "wb") as f:
            f.write(file.getbuffer())
        result = process_omr(file.name)  # Replace with your function
        results.append(result)

    # Save to CSV
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv("results.csv", index=False)
    st.success("Processing complete! Download your results below:")
    st.download_button("Download CSV", data=df.to_csv(index=False), file_name="results.csv")
