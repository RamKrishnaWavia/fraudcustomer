import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import timedelta

# ----------------------------
# Helper Functions
# ----------------------------
def detect_fraud(df):
    fraud_customers = []
    today = df['report_date'].max()

    for cust, group in df.groupby('customer_id'):
        group = group.sort_values('report_date')

        # Check >=3 refunds in last 30 days
        recent_refunds = group[group['report_date'] >= (today - timedelta(days=30))]
        if recent_refunds.shape[0] >= 3:
            fraud_customers.append(cust)
            continue

        # Check for 3 consecutive days refunds
        dates = group['report_date'].drop_duplicates().sort_values()
        consec_count = 1
        for i in range(1, len(dates)):
            if (dates.iloc[i] - dates.iloc[i-1]).days == 1:
                consec_count += 1
                if consec_count >= 3:
                    fraud_customers.append(cust)
                    break
            else:
                consec_count = 1

    return list(set(fraud_customers))


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Refund Analyzer", layout="wide")

st.title("💰 Refund Analyzer with AI Insights")

# Downloadable template
with open("refund_template.xlsx", "wb") as f:
    pd.DataFrame(columns=[
        "report_date","customer_id","bb_id","order_id","TYPE","DC_name","fc_name","fc_id",
        "dark_store","category","Hub","society_name","sub_category","skuid","brand",
        "product_name","pack_size","hsn","GST_percent","Cess_percent","sales_without_tax",
        "SGST_Value","CGST_Value","IGST_Value","CESS_Value","order_date","quantity","mrp",
        "cost_price","selling_price","sale_value","slot_charges","is_indent","refund_comment",
        "subscription_id","sales_without_delivery_charge","discount_amount",
        "is_free_coupon_product","user_source","delivery_status","society_id","block_name",
        "tag","order_ver","bb_order_id","fo_customer","cost_price_processed"
    ]).to_excel(f, index=False)

st.download_button(
    label="📥 Download Input Template",
    data=open("refund_template.xlsx", "rb").read(),
    file_name="refund_template.xlsx"
)

uploaded_file = st.file_uploader("Upload Refund Data (Excel)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Preprocess dates
    df['report_date'] = pd.to_datetime(df['report_date'], errors='coerce')
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')

    # Refund value
    df['refund_value'] = df.apply(lambda x: x['sales_without_delivery_charge'] if pd.notnull(x['refund_comment']) else 0, axis=1)

    # Metrics
    total_refund_value = df['refund_value'].sum()
    refund_orders = df[df['refund_value'] > 0].shape[0]
    total_orders = df.shape[0]
    refund_pct = (refund_orders / total_orders * 100) if total_orders > 0 else 0
    total_refund_days = df[df['refund_value'] > 0]['report_date'].nunique()
    top_refund_sku = df.groupby('skuid')['refund_value'].sum().idxmax() if refund_orders > 0 else "N/A"

    # Fraud detection
    fraud_customers = detect_fraud(df)
    high_risk_count = len(fraud_customers)

    # ----------------------------
    # Summary Cards
    # ----------------------------
    st.subheader("📊 Refund Summary")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Refund Value (₹)", f"{total_refund_value:,.2f}")
    col2.metric("Refund % of Orders", f"{refund_pct:.2f}%")
    col3.metric("Total Refund Days", total_refund_days)
    col4.metric("Top Refund SKU", top_refund_sku)
    col5.metric("High-Risk Customers", high_risk_count)

    # ----------------------------
    # Charts
    # ----------------------------
    st.subheader("📈 Refund Analysis Charts")

    chart_tabs = st.tabs(["By Hub", "By Dark Store", "By Category", "By Society ID", "Fraud Customers"])

    with chart_tabs[0]:
        fig = px.bar(df.groupby('Hub')['refund_value'].sum().reset_index(), x="Hub", y="refund_value", title="Refund Value by Hub")
        st.plotly_chart(fig, use_container_width=True)

    with chart_tabs[1]:
        fig = px.bar(df.groupby('dark_store')['refund_value'].sum().reset_index(), x="dark_store", y="refund_value", title="Refund Value by Dark Store")
        st.plotly_chart(fig, use_container_width=True)

    with chart_tabs[2]:
        fig = px.bar(df.groupby('category')['refund_value'].sum().reset_index(), x="category", y="refund_value", title="Refund Value by Category")
        st.plotly_chart(fig, use_container_width=True)

    with chart_tabs[3]:
        fig = px.bar(df.groupby('society_id')['refund_value'].sum().reset_index(), x="society_id", y="refund_value", title="Refund Value by Society ID")
        st.plotly_chart(fig, use_container_width=True)

    with chart_tabs[4]:
        fraud_df = df[df['customer_id'].isin(fraud_customers)]
        if fraud_df.empty:
            st.success("✅ No fraud customers detected based on current rules.")
        else:
            fig = px.histogram(fraud_df, x="customer_id", y="refund_value", histfunc="sum", title="Fraud Customers Refund Value")
            st.plotly_chart(fig, use_container_width=True)

    # ----------------------------
    # AI Insights + Recommendations
    # ----------------------------
    st.subheader("🤖 AI Insights")
    st.text_area("AI Observations", 
                 f"- Refund % is {refund_pct:.2f}%.\n"
                 f"- High-risk customers detected: {high_risk_count}.\n"
                 f"- SKU {top_refund_sku} contributes most refunds.\n"
                 f"- Refunds spread across {total_refund_days} days.")

    st.subheader("💡 Recommendations Panel")
    st.write("""
    - 🚦 Apply **blocking criteria** for customers with ≥3 refunds in last 30 days.  
    - 🚫 Flag customers refunding for **3 consecutive days**.  
    - 📦 Investigate top refund SKUs for packaging/delivery issues.  
    - 🏢 Deep dive into hubs/dark stores with consistently higher refund %.  
    """)
