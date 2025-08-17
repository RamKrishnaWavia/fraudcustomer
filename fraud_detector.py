import streamlit as st
import pandas as pd
import io
from datetime import datetime, timedelta

st.set_page_config(page_title="Refund Fraud Detection", layout="wide")

st.title("🛑 Refund Fraud Detection Dashboard")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload Refund Data (CSV/Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # --- Read File ---
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Ensure necessary columns exist
    required_cols = ["Customer_ID", "Refund_Date", "Refund_Value"]
    if not all(col in df.columns for col in required_cols):
        st.error(f"❌ File must contain these columns: {required_cols}")
    else:
        df["Refund_Date"] = pd.to_datetime(df["Refund_Date"], errors="coerce")

        # --- Fraud Detection Rules ---
        today = datetime.today()
        last_30_days = today - timedelta(days=30)

        # Refund count in last 30 days
        refund_counts = (
            df[df["Refund_Date"] >= last_30_days]
            .groupby("Customer_ID")["Refund_Date"]
            .count()
            .reset_index()
            .rename(columns={"Refund_Date": "Refund_Count_Last30Days"})
        )

        # Consecutive refund days
        cons_refunds = (
            df.groupby("Customer_ID")["Refund_Date"]
            .apply(lambda x: x.sort_values().diff().eq("1 days").astype(int).groupby(x.sort_values().cumsum()).cumsum().max())
            .reset_index()
        )
        cons_refunds = cons_refunds.groupby("Customer_ID")["Refund_Date"].max().reset_index()
        cons_refunds = cons_refunds.rename(columns={"Refund_Date": "Consecutive_Refund_Days"})

        # Merge fraud features
        fraud_data = refund_counts.merge(cons_refunds, on="Customer_ID", how="left").fillna(0)

        # Fraud customers
        fraud_customers = fraud_data[
            (fraud_data["Refund_Count_Last30Days"] >= 3) | 
            (fraud_data["Consecutive_Refund_Days"] >= 3)
        ]

        # --- Summary Cards ---
        total_customers = df["Customer_ID"].nunique()
        total_refunds = len(df)
        fraud_count = len(fraud_customers)
        cons_3days_count = len(fraud_data[fraud_data["Consecutive_Refund_Days"] >= 3])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("👥 Total Customers", total_customers)
        col2.metric("📦 Total Refunds", total_refunds)
        col3.metric("🚨 Fraud Customers (≥3 Refunds)", fraud_count)
        col4.metric("⚠️ 3-Day Consecutive Refund Customers", cons_3days_count)

        # --- Show Fraud Customers ---
        st.subheader("🚨 Potential Fraud Customers")
        if not fraud_customers.empty:
            st.dataframe(fraud_customers, use_container_width=True)
        else:
            st.success("✅ No fraud customers detected!")

        # --- Export Options ---
        st.subheader("📤 Export Options")

        if not fraud_customers.empty:
            # Excel with conditional formatting
            fraud_excel = io.BytesIO()
            with pd.ExcelWriter(fraud_excel, engine="xlsxwriter") as writer:
                fraud_customers.to_excel(writer, index=False, sheet_name="Fraud_Customers")
                workbook  = writer.book
                worksheet = writer.sheets["Fraud_Customers"]

                # Define formats
                red_format = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})   # Red
                yellow_format = workbook.add_format({'bg_color': '#FFEB9C', 'font_color': '#9C6500'}) # Yellow

                # Apply conditional formatting for refund count ≥3 in last 30 days
                refund_col = fraud_customers.columns.get_loc("Refund_Count_Last30Days") + 1
                worksheet.conditional_format(
                    1, refund_col, len(fraud_customers), refund_col,
                    {"type": "cell", "criteria": ">=", "value": 3, "format": red_format}
                )

                # Apply conditional formatting for 3 consecutive days refunds
                cons_col = fraud_customers.columns.get_loc("Consecutive_Refund_Days") + 1
                worksheet.conditional_format(
                    1, cons_col, len(fraud_customers), cons_col,
                    {"type": "cell", "criteria": ">=", "value": 3, "format": yellow_format}
                )

            st.download_button(
                label="⬇️ Download Fraud Customers (Excel, Highlighted)",
                data=fraud_excel.getvalue(),
                file_name="fraud_customers_highlighted.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            # Simple CSV option
            fraud_csv = fraud_customers.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Fraud Customers (CSV)",
                data=fraud_csv,
                file_name="fraud_customers.csv",
                mime="text/csv"
            )

        # Full refund data export
        refund_csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download All Refund Data (CSV)",
            data=refund_csv,
            file_name="refund_data.csv",
            mime="text/csv"
        )

        refund_excel = io.BytesIO()
        with pd.ExcelWriter(refund_excel, engine="xlsxwriter") as writer:
            df.to_excel(refund_excel, index=False, sheet_name="Refund_Data")
        st.download_button(
            label="⬇️ Download All Refund Data (Excel)",
            data=refund_excel.getvalue(),
            file_name="refund_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
