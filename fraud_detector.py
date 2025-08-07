import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# --- 1. Data Input (Simulated or from Your Data Source) ---
# ---  Option 1: Simulate data  ---
def generate_sample_data():
    # Simulate a mix of legitimate and potentially fraudulent data
    num_customers = 20
    data = {
        'customer_id': [],
        'order_id': [],
        'order_date': [],
        'order_amount': [],
        'refund_request_date': [],
        'refund_amount': [],
        'refund_reason': []
    }

    now = datetime.now()
    for customer_id in range(1, num_customers + 1):
        num_orders = 3 + (customer_id % 4) # Vary orders per customer
        for order_num in range(1, num_orders + 1):
            order_date = now - timedelta(days= (order_num + (customer_id * 2)) % 30) # Create variety in order dates
            order_amount = 25 + (order_num * 5) + (customer_id % 15)
            refund_chance = 0.1 if customer_id < 5 else (0.3 if customer_id < 8 else 0.05)  # Higher refund chance for some customers
            if st.session_state.debug_mode and customer_id >=18:  # Force fraud for debugging
                refund_chance = 0.85
            if (customer_id % 4) == 0 and order_num == num_orders:  # More fraud
                refund_chance = 0.75

            if st.session_state.debug_mode and customer_id == 2: # force fraud for debug
                refund_chance = 0.85


            # Use numpy's random.choice for a more direct and reliable approach
            if np.random.choice([True, False], p=[refund_chance, 1 - refund_chance]):
                refund_amount = order_amount * (0.2 + (0.1 if customer_id < 5 else 0))  # Vary refund amount
                refund_amount = min(refund_amount,order_amount)
                refund_request_date = order_date + timedelta(days=1)  # Make sure the order is before refund.
                refund_reason = pd.Series(['Missing Item','Damaged Item','Poor Quality']).sample(1).iloc[0] if refund_amount > 0 else None

            else:
                refund_amount = 0
                refund_request_date = None
                refund_reason = None

            data['customer_id'].append(customer_id)
            data['order_id'].append(customer_id*1000 + order_num)
            data['order_date'].append(order_date)
            data['order_amount'].append(order_amount)
            data['refund_request_date'].append(refund_request_date)
            data['refund_amount'].append(refund_amount)
            data['refund_reason'].append(refund_reason)
    return pd.DataFrame(data)


# --- 2. Data Processing & Fraud Detection Functions ---
@st.cache_data  # Use caching to avoid re-processing on every rerun
def process_data(df:pd.DataFrame):
    # --- Preprocessing ---
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['refund_request_date'] = pd.to_datetime(df['refund_request_date'])

    # --- Calculate Metrics ---
    df['refund_to_order_ratio'] = (df['refund_amount'] / df['order_amount']) * 100

    customer_stats = df.groupby('customer_id').agg(
        total_orders=('order_id', 'count'),
        total_refunds=('refund_amount', 'sum'),
        num_refunds=('refund_amount', 'count'),
        total_order_value=('order_amount', 'sum')
    )
    customer_stats['refund_ratio'] = (customer_stats['total_refunds'] / customer_stats['total_order_value']) * 100
    customer_stats['refund_to_order_percentage'] = (customer_stats['num_refunds'] / customer_stats['total_orders']) * 100
    return df, customer_stats

@st.cache_data(show_spinner=True)
def detect_fraud(customer_stats:pd.DataFrame, threshold_config:dict):
    # --- Apply Rules ---
    fraud_customers = customer_stats[
        (customer_stats['num_refunds'] >= threshold_config["num_refunds"]) |  # Rule 1
        (customer_stats['refund_ratio'] >= threshold_config["refund_ratio"]) |  # Rule 2
        (customer_stats['refund_to_order_percentage'] >= threshold_config["refund_to_order_percentage"])  # Rule 3
    ]

    fraud_customers = fraud_customers.sort_values(by=["refund_ratio"],ascending=False)  # Sort to show most suspicious customers first.

    return fraud_customers

# --- 3. Streamlit UI ---
st.title("Fraud Detection Tool for Daily Subscriptions")

# ---  Debugging mode  ---
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False
st.sidebar.write("Debug Mode: Click the checkbox to inject fraudulent data.")
st.session_state.debug_mode = st.sidebar.checkbox("Activate Debug Mode", value=st.session_state.debug_mode)


# --- Data Input Selection ---
st.sidebar.header("Data Input")
data_source = st.sidebar.radio("Choose Data Source:",
                               ("Simulated Data", "Upload CSV"),
                               index=0)  # Default to simulated data

if data_source == "Simulated Data":
    df = generate_sample_data()
    st.write("Using Simulated Data.")
    # Show raw data if in debug mode or if the user wants to view it
    if st.session_state.debug_mode or st.sidebar.checkbox("Show Raw Simulated Data"):
        st.subheader("Raw Simulated Data")
        st.dataframe(df)
elif data_source == "Upload CSV":
    upload_data = st.file_uploader("Upload your order/refund data (CSV)", type=["csv"])
    if upload_data is not None:
        df = pd.read_csv(upload_data)
    else:
        st.warning("Please upload a CSV file or download the template.")
        # --- CSV Template Download Option ---
        template_data = {
            'customer_id': [1, 2, 3],
            'order_id': [101, 102, 103],
            'order_date': ['2024-11-01', '2024-11-02', '2024-11-03'],
            'order_amount': [50.00, 75.00, 60.00],
            'refund_request_date': ['', '2024-11-02', ''],
            'refund_amount': [0.00, 25.00, 0.00],
            'refund_reason': ['', 'Damaged Item', '']
        }
        template_df = pd.DataFrame(template_data)
        csv_template = template_df.to_csv(index=False)
        st.download_button(
            label="Download CSV Template",
            data=csv_template,
            file_name="refund_data_template.csv",
            mime="text/csv",
        )
        df = pd.DataFrame()  # Ensure df exists even with an error.

else:
    st.error("Invalid data source selection.")
    df = pd.DataFrame() # Ensure df exists even with an error.


# --- 4. Fraud Detection and Display ---
if not df.empty:
    # --- Configure Thresholds  ---
    st.sidebar.header("Threshold Configuration")
    threshold_config = {
        "num_refunds": st.sidebar.slider("Minimum Number of Refunds", 1, 10, 2),
        "refund_ratio": st.sidebar.slider("Refund Ratio Threshold (%)", 0, 100, 20),
        "refund_to_order_percentage": st.sidebar.slider("Refunds per Order (%)", 0, 100, 20)
    }


    # --- Process Data ---
    df, customer_stats = process_data(df)

    # --- Detect Fraud ---
    fraud_customers = detect_fraud(customer_stats, threshold_config)

    # --- Display Results ---
    if not fraud_customers.empty:
        st.subheader("Potential Fraud Customers")
        st.dataframe(fraud_customers)

        # Detailed View (for more in-depth analysis)
        customer_id_to_investigate = st.selectbox("Select Customer ID for Details", fraud_customers.index.tolist())
        if customer_id_to_investigate:
            customer_details = df[df['customer_id'] == customer_id_to_investigate].sort_values(by="order_date",ascending=False) # show most recent first
            st.subheader(f"Customer {customer_id_to_investigate} Order/Refund History")
            st.dataframe(customer_details)


    else:
        st.success("No suspicious activity detected based on current thresholds.")
