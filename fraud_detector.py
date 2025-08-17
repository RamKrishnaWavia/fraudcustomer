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

    # Check for missing values
    if df.isnull().any().any():
        st.warning("⚠️ Missing values detected.  Rows with missing values will be dropped for processing.")
        df = df.dropna()  # Or use imputation based on your needs

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
        refund_pct = round((df['Refund_Value'].sum()/df['sales_without_delivery_charge'].sum())*100,2) #Use sales_without_delivery_charge
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
    required_cols = ["Customer_ID", "order_date", "sales_without_delivery_charge", "refund_comment"] #Now required_cols has refund_comment

    # --- **Column Renaming (if needed)** ---
    # Only rename columns *before* validating the data.
    rename_dict = {
        "customer_id": "Customer_ID",  # Rename "customer_id" to "Customer_ID"
        "order_date": "order_date",  #No change needed
        "sales_without_delivery_charge": "sales_without_delivery_charge", #No change needed
        "refund_comment": "refund_comment" # no change needed
    }
    df = df.rename(columns=rename_dict)

    df = validate_data(df, required_cols)

    if df is not False:  #Proceed only if validation is successful
        # --- Create the `Refund_Date` column, before calling the function.
        df["Refund_Date"] = pd.to_datetime(df["order_date"])  # <--- MOVE THIS UP HERE

        # --- Calculate Refund_Value based on refund_comment---
        if "refund_comment" in df.columns:
           df["Refund_Value"] = df.apply(lambda row: row["sales_without_delivery_charge"] if pd.notna(row["refund_comment"]) else 0, axis=1)
        else:
           df["Refund_Value"] = 0  # Or handle the case where refund_comment is missing

        # --- Fraud Detection (with configurable thresholds) ---
        with st.expander("⚙️ Fraud Detection Settings"):
            refund_threshold = st.slider("Refunds in Last 30 Days Threshold", min_value=1, max_value=10, value=3, step=1)
            consecutive_days_threshold = st.slider("Consecutive Refund Days Threshold", min_value=1, max_value=5, value=2, step=1)
        fraud_customers = detect_fraud_customers(df, refund_threshold, consecutive_days_threshold)

        fraud_df = df[df["Customer_ID"].isin(fraud_customers)].groupby("Customer_ID", as_index=False)["Refund_Value"].sum()

        # --- Summary Cards ---
        total_refund = df["Refund_Value"].sum()
        refund_pct = round((df["Refund_Value"].sum() / df["sales_without_delivery_charge"].sum())*100, 2) #No need for the conditional statement. sales_without_delivery_charge is used
        total_days = df["Refund_Date"].nunique()
        top_sku = "N/A"  # Placeholder (add if SKU data present - requires SKU data)
        high_risk_count = len(fraud_customers)

        #Summary Cards
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("💰 Total Refund Value", f"₹{total_refund:,.0f}")
        c2.metric("📊 Refund % of Orders", f"{refund_pct}%")
        c3.metric("📅 Total Refund Days", total_days)
        c4.metric("🥭 Top Refund SKU", top_sku) #Requires implementation later
        c5.metric("👤 High-Risk Customers", high_risk_count)

        # --- AI Insights ---
        st.subheader("🤖 AI Insights & Recommendations")
        # --- Generate AI Insights ---
        insights = generate_ai_insights(df)
        st.write(insights)

        # --- Visual Dashboard ---
        st.subheader("📊 Visual Dashboard")
        tab1, tab2, tab3, tab4 = st.tabs(["Refunds by SKU", "Refunds by Reason", "Refund Trend", "Top Performers"])

        with tab1:
            # Implement a bar chart for refunds by SKU.
            # Assuming you have a 'product_name' column, replace 'product_name' if your column name is different
            if 'product_name' in df.columns:
                sku_refunds = df.groupby('product_name')['Refund_Value'].sum().reset_index()
                fig_sku = px.bar(sku_refunds, x='product_name', y='Refund_Value', title='Refunds by SKU (Product Name)')
                st.plotly_chart(fig_sku, use_container_width=True)
            else:
                st.warning("⚠️ 'product_name' column not found.  Cannot display Refunds by SKU chart.")


        with tab2:
            # Implement a pie chart for refunds by reason.
            # Assuming a 'refund_comment' is present, and can be grouped into categories
            if 'refund_comment' in df.columns:
                # Simple example:  Categorize based on keywords in the comment
                def categorize_reason(comment):
                    if pd.isna(comment):
                        return "Other"
                    comment = comment.lower()
                    if "quality" in comment or "bad" in comment:
                        return "Quality"
                    if "damage" in comment or "broken" in comment:
                        return "Damage"
                    if "fraud" in comment or "scam" in comment:
                        return "Customer Fraud"
                    return "Other"

                df['refund_reason'] = df['refund_comment'].apply(categorize_reason)
                reason_refunds = df.groupby('refund_reason')['Refund_Value'].sum().reset_index()
                fig_reason = px.pie(reason_refunds, names='refund_reason', values='Refund_Value',
                                     title='Refunds by Reason')
                st.plotly_chart(fig_reason, use_container_width=True)
            else:
                st.warning("⚠️ 'refund_comment' column not found.  Cannot display Refunds by Reason chart.")


        with tab3:
            # Implement a line chart for the refund trend (daily or weekly).
            # Group by `Refund_Date`
            refund_trend = df.groupby('Refund_Date')['Refund_Value'].sum().reset_index()
            fig_trend = px.line(refund_trend, x='Refund_Date', y='Refund_Value', title='Refund Trend Over Time (Daily)')
            st.plotly_chart(fig_trend, use_container_width=True)

        with tab4:
            # Implement a table for top SKUs, customers, and riders.
            # Top 10 SKUs
            if 'product_name' in df.columns:
                top_skus = df.groupby('product_name')['Refund_Value'].sum().nlargest(10).reset_index()
                top_skus.rename(columns={'Refund_Value': 'Total Refund Value'}, inplace=True)
                st.subheader("Top 10 SKUs by Refund Value")
                st.dataframe(top_skus, use_container_width=True)
            else:
                st.warning("⚠️ 'product_name' column not found.  Cannot display Top SKUs table.")

            # Top 10 Customers
            top_customers = df.groupby('Customer_ID')['Refund_Value'].sum().nlargest(10).reset_index()
            top_customers.rename(columns={'Refund_Value': 'Total Refund Value'}, inplace=True)
            st.subheader("Top 10 Customers by Refund Value")
            st.dataframe(top_customers, use_container_width=True)

            # (Placeholder for Top Riders - requires rider data)
            st.subheader("Top Riders (Placeholder - requires rider data)")
            st.write("Rider data not available in this dataset.") # If rider data isn't in your input.

        # --- Recommendations Panel ---
        st.subheader("🚦 Recommendations")
        # --- Refund Blocking Criteria Suggestions ---
        st.write("Based on your data, here are some potential refund blocking criteria:")
        # -- Example Recommendations (Customize These) --
        if 'product_name' in df.columns:
            #SKU Recommendation
            sku_refunds = df.groupby('product_name')['Refund_Value'].sum().reset_index()
            total_sales_by_sku = df.groupby('product_name')['sales_without_delivery_charge'].sum().reset_index()
            sku_data = pd.merge(sku_refunds, total_sales_by_sku, on='product_name', how='left')
            sku_data['refund_percentage'] = (sku_data['Refund_Value'] / sku_data['sales_without_delivery_charge']) * 100
            high_refund_skus = sku_data[sku_data['refund_percentage'] > 15]
            if not high_refund_skus.empty:
                st.markdown("<span style='color:red;'>⚠️ Block SKUs with Refund % > 15%:</span>", unsafe_allow_html=True)
                for index, row in high_refund_skus.iterrows():
                   st.write(f"  - Block SKU '{row['product_name']}' (Refund %: {row['refund_percentage']:.2f}%)")

        # Customer Recommendation
        customer_refunds = df.groupby('Customer_ID')['Refund_Value'].agg(['sum', 'count']).reset_index()
        customer_refunds.columns = ['Customer_ID', 'Total Refund Value', 'Refund Count']
        high_refund_customers = customer_refunds[customer_refunds['Refund Count'] > 5]  # Example threshold
        if not high_refund_customers.empty:
            st.markdown("<span style='color:red;'>⚠️ Flag Customers with > 5 Refund Requests:</span>", unsafe_allow_html=True)
            for index, row in high_refund_customers.iterrows():
               st.write(f"  - Flag Customer ID '{row['Customer_ID']}' (Refund Count: {row['Refund Count']})")
