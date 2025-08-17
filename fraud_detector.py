import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------
# Streamlit Page Config
# --------------------------
st.set_page_config(page_title="Refund Analyzer", layout="wide")

st.title("📊 Refund Analyzer Dashboard")

# --------------------------
# File Upload
# --------------------------
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Ensure date formatting
    if 'order_date' in df.columns:
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')

    # Refund filtering: if refund_comment is not empty/null
    df['is_refund'] = df['refund_comment'].notna() & (df['refund_comment'].astype(str).str.strip() != "")
    df_refund = df[df['is_refund']].copy()

    # Refund value = sale_value
    df_refund['refund_value'] = df_refund['sale_value']

    # --------------------------
    # KPIs
    # --------------------------
    total_orders = len(df)
    total_refunds = len(df_refund)
    total_sales = df['sale_value'].sum()
    total_refund_value = df_refund['refund_value'].sum()
    refund_percent = (total_refund_value / total_sales * 100) if total_sales > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Orders", f"{total_orders:,}")
    col2.metric("Refund Orders", f"{total_refunds:,}")
    col3.metric("Total Refund Value", f"₹{total_refund_value:,.0f}")
    col4.metric("Refund % of Sales", f"{refund_percent:.2f}%")

    st.divider()

    # --------------------------
    # Pareto: Refund by Society
    # --------------------------
    st.subheader("🏠 Refunds by Society")

    society_group = df_refund.groupby("society_id", as_index=False)['refund_value'].sum()
    society_group = society_group.sort_values(by="refund_value", ascending=False)
    society_group['cumperc'] = 100 * society_group['refund_value'].cumsum() / society_group['refund_value'].sum()

    fig, ax = plt.subplots(figsize=(10,5))
    ax.bar(society_group['society_id'].astype(str), society_group['refund_value'])
    ax2 = ax.twinx()
    ax2.plot(society_group['society_id'].astype(str), society_group['cumperc'], color="red", marker="o")

    ax.set_title("Pareto - Refund Value by Society")
    ax.set_ylabel("Refund Value (₹)")
    ax2.set_ylabel("Cumulative %")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

    st.dataframe(society_group.head(20))  # show top 20 societies

    st.divider()

    # --------------------------
    # Pareto: Refund by Reason
    # --------------------------
    st.subheader("💬 Refunds by Reason")

    reason_group = df_refund.groupby("refund_comment", as_index=False)['refund_value'].sum()
    reason_group = reason_group.sort_values(by="refund_value", ascending=False)
    reason_group['cumperc'] = 100 * reason_group['refund_value'].cumsum() / reason_group['refund_value'].sum()

    fig, ax = plt.subplots(figsize=(10,5))
    ax.bar(reason_group['refund_comment'].astype(str), reason_group['refund_value'])
    ax2 = ax.twinx()
    ax2.plot(reason_group['refund_comment'].astype(str), reason_group['cumperc'], color="red", marker="o")

    ax.set_title("Pareto - Refund Value by Reason")
    ax.set_ylabel("Refund Value (₹)")
    ax2.set_ylabel("Cumulative %")
    ax.tick_params(axis='x', rotation=90)
    st.pyplot(fig)

    st.dataframe(reason_group.head(20))  # show top 20 refund reasons
