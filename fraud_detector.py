import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import openai
import datetime

# ========== SETUP ==========
st.set_page_config(page_title="Refund Fraud Detection", layout="wide")

# Sidebar for upload
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload Excel/CSV", type=["xlsx", "csv"])

# OpenAI key (must set in Streamlit Cloud → Secrets)
openai.api_key = st.secrets.get("OPENAI_API_KEY", "")

# Required columns
required_columns = ["customer_id", "order_date", "sales_without_delivery_charge", "refund_comment"]

# ========== FILE HANDLING ==========
if uploaded_file:
    if uploaded_file.name.endswith("csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Validation
    if not all(col in df.columns for col in required_columns):
        st.error(f"❌ File must contain these columns: {required_columns}")
    else:
        # Convert dates
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

        # Refund Value: only where refund_comment is not empty
        df["Refund_Value"] = np.where(df["refund_comment"].notna() & (df["refund_comment"] != ""),
                                      df["sales_without_delivery_charge"], 0)

        # Refund days
        df["Refund_Day"] = df["order_date"].dt.date

        # ===== Summary Stats =====
        total_refund_value = df["Refund_Value"].sum()
        total_orders = df.shape[0]
        refund_orders = df[df["Refund_Value"] > 0].shape[0]
        refund_pct = (refund_orders / total_orders * 100) if total_orders > 0 else 0
        total_refund_days = df.loc[df["Refund_Value"] > 0, "Refund_Day"].nunique()
        top_refund_sku = df.loc[df["Refund_Value"] > 0, "product_name"].mode()[0] if not df[df["Refund_Value"] > 0].empty else "N/A"
        high_risk_customers = df.loc[df["Refund_Value"] > 0, "customer_id"].nunique()

        # ===== Fraud Detection =====
        fraud_customers = []

        for cust, cust_df in df.groupby("customer_id"):
            cust_df = cust_df[cust_df["Refund_Value"] > 0].sort_values("order_date")

            # 1. ≥3 refunds in last 30 days
            rolling_window = cust_df.set_index("order_date").rolling("30D")["Refund_Value"].count()
            if (rolling_window >= 3).any():
                fraud_customers.append(cust)

            # 2. 3 consecutive days refunds
            dates = cust_df["order_date"].dt.date.drop_duplicates().sort_values()
            consec = 1
            for i in range(1, len(dates)):
                if (dates.iloc[i] - dates.iloc[i-1]).days == 1:
                    consec += 1
                    if consec >= 3:
                        fraud_customers.append(cust)
                        break
                else:
                    consec = 1

        fraud_customers = list(set(fraud_customers))
        fraud_df = df[df["customer_id"].isin(fraud_customers)]

        # ========== UI ==========
        st.title("🛡️ Refund Fraud Detection Dashboard")

        # ===== Summary Cards =====
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("💰 Total Refund Value", f"₹{total_refund_value:,.0f}")
        col2.metric("📊 Refund % of Orders", f"{refund_pct:.2f}%")
        col3.metric("📅 Refund Days", total_refund_days)
        col4.metric("🥭 Top Refund SKU", top_refund_sku)
        col5.metric("👤 High-Risk Customers", high_risk_customers)

        # ===== Charts =====
        st.subheader("📈 Refund Analysis by Dimensions")

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Hub", "Dark Store", "Category", "Society ID", "Fraud Customers"])

        with tab1:
            fig, ax = plt.subplots()
            sns.barplot(x="Hub", y="Refund_Value", data=df, estimator=sum, ci=None, ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)

        with tab2:
            fig, ax = plt.subplots()
            sns.barplot(x="dark_store", y="Refund_Value", data=df, estimator=sum, ci=None, ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)

        with tab3:
            fig, ax = plt.subplots()
            sns.barplot(x="category", y="Refund_Value", data=df, estimator=sum, ci=None, ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)

        with tab4:
            fig, ax = plt.subplots()
            sns.barplot(x="society_id", y="Refund_Value", data=df, estimator=sum, ci=None, ax=ax)
            plt.xticks(rotation=90)
            st.pyplot(fig)

        with tab5:
            st.write("🚨 Fraud Customers Detected")
            st.dataframe(fraud_df[["customer_id", "order_date", "Refund_Value", "refund_comment"]])

            # Download option
            buffer = BytesIO()
            fraud_df.to_excel(buffer, index=False)
            st.download_button(
                label="📥 Download Fraud Customers (Excel)",
                data=buffer.getvalue(),
                file_name="fraud_customers.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            csv = fraud_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Fraud Customers (CSV)",
                data=csv,
                file_name="fraud_customers.csv",
                mime="text/csv"
            )

        # ===== AI Insights =====
        st.subheader("🤖 AI Insights & Recommendations")
        if openai.api_key:
            try:
                prompt = f"""
                Analyze this refund dataset summary:
                - Total refund value = {total_refund_value}
                - Refund % = {refund_pct:.2f}%
                - Refund days = {total_refund_days}
                - Top refund SKU = {top_refund_sku}
                - High-risk customers = {high_risk_customers}

                Provide insights, patterns, and business recommendations to reduce fraud and improve controls.
                """
                response = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": "You are a data analyst."},
                              {"role": "user", "content": prompt}]
                )
                st.info(response.choices[0].message.content)
            except Exception as e:
                st.warning(f"⚠️ AI Insights unavailable: {e}")
        else:
            st.warning("⚠️ Set your OPENAI_API_KEY in Streamlit Cloud Secrets for AI insights.")

else:
    st.info("⬆️ Upload a file to begin analysis. Download input template below.")

    # Template file
    template_df = pd.DataFrame(columns=required_columns)
    buffer = BytesIO()
    template_df.to_excel(buffer, index=False)
    st.download_button(
        label="📥 Download Template Excel",
        data=buffer.getvalue(),
        file_name="refund_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
