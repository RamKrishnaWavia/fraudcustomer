import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
import openai
import os

# Set OpenAI API Key (must be set in Streamlit Cloud -> Secrets)
openai.api_key = os.getenv("OPENAI_API_KEY")

# ---------------------------
# Helper Functions
# ---------------------------

def detect_fraud_customers(df):
    fraud_customers = []

    # Rule 1: ≥3 refunds in last 30 days
    last_30 = df[df['Refund_Date'] >= (df['Refund_Date'].max() - pd.Timedelta(days=30))]
    refunds_30 = last_30.groupby("Customer_ID").size()
    fraud_customers += refunds_30[refunds_30 >= 3].index.tolist()

    # Rule 2: Refunds for 3 consecutive days
    daily_refunds = df.groupby(["Customer_ID", "Refund_Date"]).size().reset_index(name="count")
    for cust, grp in daily_refunds.groupby("Customer_ID"):
        grp = grp.sort_values("Refund_Date")
        grp["consec"] = (grp["Refund_Date"].diff().dt.days == 1).astype(int)
        grp["streak"] = grp["consec"].groupby((grp["consec"] != grp["consec"].shift()).cumsum()).cumsum()
        if (grp["streak"] >= 2).any():
            fraud_customers.append(cust)

    return list(set(fraud_customers))


def generate_ai_insights(df):
    try:
        prompt = f"""
        Analyze the refund dataset and provide:
        1. Key refund behavior patterns
        2. Top fraud risks
        3. Suggested actions to minimize fraud
        Dataset Summary:
        Total Refund Value: {df['Refund_Value'].sum()}
        Refund %: {round((df['Refund_Value'].sum()/df['sales_without_delivery_charge'].sum())*100,2)}%
        Unique Customers: {df['Customer_ID'].nunique()}
        """
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"⚠️ AI Insights not available: {e}"


def download_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Fraud_Customers")
    return output.getvalue()

# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="Refund & Fraud Detection Dashboard", layout="wide")

st.title("🛡️ Refund & Fraud Detection Dashboard")

# File uploader with template
with st.expander("📥 Download Template"):
    st.markdown("Download the input Excel template, fill it with data, then re-upload.")
    template = pd.DataFrame({
        "Customer_ID": [],
        "order_date": [],
        "sales_without_delivery_charge": []
    })
    st.download_button("⬇️ Download Template", data=template.to_csv(index=False), file_name="refund_template.csv")

uploaded_file = st.file_uploader("Upload Refund Data (Excel/CSV)", type=["csv", "xlsx"])

if uploaded_file:
    # Load data
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Ensure required cols
    required_cols = ["Customer_ID", "order_date", "sales_without_delivery_charge"]
    if not all(col in df.columns for col in required_cols):
        st.error(f"❌ File must contain columns: {required_cols}")
        st.stop()

    # Rename for consistency
    df["Refund_Date"] = pd.to_datetime(df["order_date"])
    df["Refund_Value"] = df["sales_without_delivery_charge"]

    # ---------------------------
    # Fraud Detection
    # ---------------------------
    fraud_customers = detect_fraud_customers(df)
    fraud_df = df[df["Customer_ID"].isin(fraud_customers)].groupby("Customer_ID", as_index=False)["Refund_Value"].sum()

    # ---------------------------
    # Summary Cards
    # ---------------------------
    total_refund = df["Refund_Value"].sum()
    refund_pct = round((df["Refund_Value"].sum() / df["sales_without_delivery_charge"].sum())*100, 2)
    total_days = df["Refund_Date"].nunique()
    top_sku = "N/A"  # Placeholder (add if SKU data present)
    high_risk_count = len(fraud_customers)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("💰 Total Refund Value", f"₹{total_refund:,.0f}")
    c2.metric("📊 Refund % of Orders", f"{refund_pct}%")
    c3.metric("📅 Total Refund Days", total_days)
    c4.metric("🥭 Top Refund SKU", top_sku)
    c5.metric("👤 High-Risk Customers", high_risk_count)

    # ---------------------------
    # Charts
    # ---------------------------
    st.subheader("📊 Refund Analysis Charts")

    tab1, tab2, tab3, tab4 = st.tabs(["By Customer", "By Date", "Fraud Customers", "Pivot Views"])

    with tab1:
        fig = px.bar(df.groupby("Customer_ID", as_index=False)["Refund_Value"].sum(),
                     x="Customer_ID", y="Refund_Value", title="Refunds by Customer")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = px.line(df.groupby("Refund_Date", as_index=False)["Refund_Value"].sum(),
                      x="Refund_Date", y="Refund_Value", title="Refund Trend Over Time")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if not fraud_df.empty:
            fig = px.bar(fraud_df, x="Customer_ID", y="Refund_Value", title="Fraud Customers Refund Value")
            st.plotly_chart(fig, use_container_width=True)
            st.download_button("⬇️ Download Fraud Customer List", data=download_excel(fraud_df),
                               file_name="fraud_customers.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.success("✅ No fraud customers detected")

    with tab4:
        fig = px.box(df, x="Customer_ID", y="Refund_Value", title="Refund Distribution per Customer")
        st.plotly_chart(fig, use_container_width=True)

    # ---------------------------
    # AI Insights
    # ---------------------------
    st.subheader("🤖 AI Insights & Recommendations")
    insights = generate_ai_insights(df)
    st.write(insights)
