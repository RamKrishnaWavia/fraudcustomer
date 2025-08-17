import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
import openai
import os

# --- Set OpenAI API Key (using environment variable) ---
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Helper Functions ---
def validate_data(df, required_cols):
    """Validates the DataFrame and handles potential issues."""
    if not all(col in df.columns for col in required_cols):
        st.error(f"❌ File must contain columns: {required_cols}.  Found columns: {df.columns.tolist()}")
        return False

    # Convert columns to the correct datatypes
    try:
        df['order_date'] = pd.to_datetime(df['order_date'])
        df['sales_without_delivery_charge'] = pd.to_numeric(df['sales_without_delivery_charge'], errors='coerce') #convert to numeric, coerce makes any non-numeric value NaN
    except ValueError as e:
        st.error(f"❌ Error converting data types: {e}. Check date and numeric columns.")
        return False

    return df  # Return the (potentially modified) dataframe


def detect_fraud_customers(df, refund_threshold=3, consecutive_days_threshold=2):
    """Detects fraudulent customers based on refund patterns."""
    #Added a try/except block, just in case there is any other error
    try:
        fraud_customers = []

        # Rule 1: ≥ refund_threshold refunds in last 30 days
        if 'Refund_Date' in df.columns: #Make sure the column exists first.
            last_30 = df[df['Refund_Date'] >= (df['Refund_Date'].max() - pd.Timedelta(days=30))]
            refunds_30 = last_30.groupby("Customer_ID").size()
            fraud_customers += refunds_30[refunds_30 >= refund_threshold].index.tolist()

        # Rule 2: Refunds for consecutive_days_threshold consecutive days
        daily_refunds = df.groupby(["Customer_ID", "Refund_Date"]).size().reset_index(name="count")
        for cust, grp in daily_refunds.groupby("Customer_ID"):
            grp = grp.sort_values("Refund_Date")
            grp["consec"] = (grp["Refund_Date"].diff().dt.days == 1).astype(int)
            grp["streak"] = grp["consec"].groupby((grp["consec"] != grp["consec"].shift()).cumsum()).cumsum()
            if (grp["streak"] >= consecutive_days_threshold).any():
                fraud_customers.append(cust)

        return list(set(fraud_customers))
    except Exception as e:
        st.error(f"An error occurred inside detect_fraud_customers: {e}")
        return [] # Return an empty list to avoid further errors.


def generate_ai_insights(df):
    """Generates AI insights about the refund data."""
    try:
        # Basic Information for the Prompt
        total_refund = df['Refund_Value'].sum()
        if df['sales_without_delivery_charge'].abs().sum() != 0:
            refund_pct = round((df["Refund_Value"].sum() / df["sales_without_delivery_charge"].sum())*100, 2)
        else:
            refund_pct = 0  # Handle the case where the total sales is zero.
        unique_customers = df['Customer_ID'].nunique()
        highest_refund_date = df.groupby('Refund_Date')['Refund_Value'].sum().idxmax()
        highest_refund_date_value = df.groupby('Refund_Date')['Refund_Value'].sum().max()
        # Provide context
        prompt = f"""
        Analyze the refund dataset and provide:
        1. Key refund behavior patterns (consider trends, customer behavior).
        2. Top fraud risks (based on the data and fraud detection rules).
        3. Suggested actions to minimize fraud.
        Dataset Summary:
        Total Refund Value: ₹{total_refund:,.0f}
        Refund %: {refund_pct}%
        Unique Customers: {unique_customers}
        Highest Refund Date: {highest_refund_date.strftime('%Y-%m-%d')} (₹{highest_refund_date_value:,.0f})
        """

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"⚠️ AI Insights not available: {e}"


def download_excel(df):
    """Downloads the DataFrame to an Excel file."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Fraud_Customers")
    return output.getvalue()

# --- Streamlit UI ---

st.set_page_config(page_title="Refund & Fraud Detection Dashboard", layout="wide")

st.title("🛡️ Refund & Fraud Detection Dashboard")

# --- File Uploader and Data Processing ---
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
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, sep=',')  # Added delimiter (you may need to change this)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"❌ Error loading the file: {e}")
        st.stop() # Stop execution if there's a file loading issue.

    # --- Data Validation and Preprocessing ---
    required_cols = ["Customer_ID", "order_date", "sales_without_delivery_charge", "refund_comment", "product_name"] #Now required_cols has refund_comment and product_name

    # --- **Column Renaming (if needed)** ---
    # Only rename columns *before* validating the data.
    rename_dict = {
        "customer_id": "Customer_ID",  # Rename "customer_id" to "Customer_ID"
        "order_date": "order_date",  #No change needed
        "sales_without_delivery_charge": "sales_without_delivery_charge", #No change needed
        "refund_comment": "refund_comment", # no change needed
        "product_name": "product_name"
    }
    df = df.rename(columns=rename_dict)

    df = validate_data(df, required_cols)

    if df is not False:  #Proceed only if validation is successful
        # --- Create the `Refund_Date` column, before calling the function.
        df["Refund_Date"] = pd.to_datetime(df["order_date"])  # <--- MOVE THIS UP HERE

        # --- Calculate Refund_Value based on refund_comment---
        if "refund_comment" in df.columns:
           df = df[df["refund_comment"].notna()] #remove the cases where the comment is empty.
           df["Refund_Value"] = df["sales_without_delivery_charge"]
        else:
           df["Refund_Value"] = 0  # Or handle the case where refund_comment is missing

        # --- Fraud Detection (with configurable thresholds) ---
        with st.expander("⚙️ Fraud Detection Settings"):
            refund_threshold = st.slider("Refunds in Last 30 Days Threshold", min_value=1, max_value=10, value=3, step=1)
            consecutive_days_threshold = st.slider("Consecutive Refund Days Threshold", min_value=1, max_value=5, value=2, step=1)
        fraud_customers = detect_fraud_customers(df, refund_threshold, consecutive_days_threshold)

        fraud_df = df[df["Customer_ID"].isin(fraud_customers)].groupby("Customer_ID", as_index=False)["Refund_Value"].sum()

        # --- Summary Cards ---
        # --- Summary Cards ---
        total_refund = df["Refund_Value"].sum()
        st.write("Total Refund Value (DEBUG):", total_refund) # Debugging
        if df['sales_without_delivery_charge'].sum() != 0:
           refund_pct = round((df["Refund_Value"].sum() / df["sales_without_delivery_charge"].sum())*100, 2) #No need for the conditional statement. sales_without_delivery_charge is used
        else:
            refund_pct = 0 # Handle the case where the total sales is zero.
        total_days = df["Refund_Date"].nunique()

        # --- Top Refund SKU Calculation ---
        if 'product_name' in df.columns:
            top_sku = df.groupby('product_name')['Refund_Value'].sum().nlargest(1).index[0]
        else:
            top_sku = "N/A"  # or handle the missing product_name column

        high_risk_count = len(fraud_customers)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("💰 Total Refund Value", f"₹{total_refund:,.0f}")
        c2.metric("📊 Refund % of Orders", f"{refund_pct}%")
        c3.metric("📅 Total Refund Days", total_days)
        c4.metric("🥭 Top Refund SKU", top_sku) # Display Top SKU
        c5.metric("👤 High-Risk Customers", high_risk_count)

        # --- Charts ---
        st.subheader("📊 Refund Analysis Charts")

        tab1, tab2, tab3, tab4 = st.tabs(["By Customer", "By Date", "Fraud Customers", "Pivot Views"])

        with tab1:
            fig = px.bar(df.groupby("Customer_ID", as_index=False)["Refund_Value"].sum(),
                         x="Customer_ID", y="Refund_Value", title="Refunds by Customer")
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            fig = px.line(df.groupby("Refund_Date")["Refund_Value"].sum().reset_index(),
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

        # --- AI Insights ---
        with st.spinner("🧠 Generating AI Insights..."):
            st.subheader("🤖 AI Insights & Recommendations")
            insights = generate_ai_insights(df)
            st.write(insights)

        # --- Display Raw Data (Optional) ---
        with st.expander("🗂️ Show Raw Data"):
            st.dataframe(df)
