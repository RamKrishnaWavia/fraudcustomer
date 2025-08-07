import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# --- 1. Data Input (Simulated or from Your Data Source) ---
# ---  Option 1: Simulate data  ---
def generate_sample_data():
    # Simulate a mix of legitimate and potentially fraudulent data (using fewer columns for demo)
    num_customers = 20
    data = {
        'Report_date': [],
        'customer_id': [],
        'order_id': [],
        'order_date': [],
        'quantity': [],
        'selling_price': [],
        'refund_comment': [],
        'refund_amount': [],
        'refund_reason': [],
        # Add all the other columns, potentially with dummy data
        'bb_id': [],  # Example: Add dummy data
        'TYPE': [],
        'DC_name': [],
        'fc_name': [],
        'fc_id': [],
        'dark_store': [],
        'category': [],
        'Hub': [],
        'society_name': [],
        'sub_category': [],
        'skuid': [],
        'brand': [],
        'product_name': [],
        'pack_size': [],
        'hsn': [],
        'GST_percent': [],
        'Cess_percent': [],
        'sales_without_tax': [],
        'SGST_Value': [],
        'CGST_Value': [],
        'IGST_Value': [],
        'CESS_Value': [],
        'mrp': [],
        'cost_price': [],
        'sale_value': [],
        'slot_charges': [],
        'is_indent': [],
        'subscription_id': [],
        'sales_without_delivery_charge': [],
        'discount_amount': [],
        'is_free_coupon_product': [],
        'user_source': [],
        'delivery_status': [],
        'society_id': [],
        'block_name': [],
        'tag': [],
        'order_ver': [],
        'bb_order_id': [],
        'fo_customer': [],
        'cost_price_processed': []
    }

    now = datetime.now()
    for customer_id in range(1, num_customers + 1):
        num_orders = 3 + (customer_id % 4)  # Vary orders per customer
        for order_num in range(1, num_orders + 1):
            order_date = now - timedelta(days=(order_num + (customer_id * 2)) % 30)  # Create variety in order dates
            quantity = 1 + (order_num % 3)  # Quantity 1,2, or 3
            selling_price = 25 + (order_num * 5) + (customer_id % 15) # Price
            refund_chance = 0.1 if customer_id < 5 else (0.3 if customer_id < 8 else 0.05)  # Higher refund chance for some customers
            if st.session_state.debug_mode and customer_id >= 18:  # Force fraud for debugging
                refund_chance = 0.85
            if (customer_id % 4) == 0 and order_num == num_orders:  # More fraud
                refund_chance = 0.75

            if st.session_state.debug_mode and customer_id == 2:  # force fraud for debug
                refund_chance = 0.85

            # Use numpy's random.choice for a more direct and reliable approach
            if np.random.choice([True, False], p=[refund_chance, 1 - refund_chance]):
                refund_amount = selling_price * quantity * (0.2 + (0.1 if customer_id < 5 else 0))  # Vary refund amount
                refund_amount = min(refund_amount,selling_price * quantity) # ensure refund amount doesn't exceed order value
                refund_request_date = order_date + timedelta(days=1)  # Make sure the order is before refund.
                # Corrected refund reason generation
                refund_reason = np.random.choice(list(refund_reasons.keys()), p=get_probabilities(list(refund_reasons.keys()))) if refund_amount > 0 else None
                refund_comment = "Sample Refund Comment" if refund_reason else None
            else:
                refund_amount = 0
                refund_request_date = None
                refund_reason = None
                refund_comment = None

            data['Report_date'].append(datetime.now().strftime('%d-%m-%Y'))  #  Today's Date
            data['customer_id'].append(customer_id)
            data['order_id'].append(customer_id * 1000 + order_num)
            data['order_date'].append(order_date.strftime('%d-%m-%Y'))  # Correct date format
            data['quantity'].append(quantity)
            data['selling_price'].append(selling_price)
            data['refund_comment'].append(refund_comment)
            data['refund_amount'].append(refund_amount)
            data['refund_reason'].append(refund_reason)

            # Add dummy data for the other columns.
            data['bb_id'].append(12345)  # Example
            data['TYPE'].append('Subscription')
            data['DC_name'].append('Example DC')
            data['fc_name'].append('Example FC')
            data['fc_id'].append(100)
            data['dark_store'].append('Example DS')
            data['category'].append('Example Category')
            data['Hub'].append('Example Hub')
            data['society_name'].append('Example Society')
            data['sub_category'].append('Example Sub')
            data['skuid'].append(98765)
            data['brand'].append('Example Brand')
            data['product_name'].append('Example Product')
            data['pack_size'].append('Example Pack')
            data['hsn'].append('12345678')
            data['GST_percent'].append(5.0)
            data['Cess_percent'].append(0.0)
            data['sales_without_tax'].append(selling_price * quantity)  # Example
            data['SGST_Value'].append(0.0)
            data['CGST_Value'].append(0.0)
            data['IGST_Value'].append(0.0)
            data['CESS_Value'].append(0.0)
            data['mrp'].append(selling_price + 5)  # Example: MRP is a bit higher.
            data['cost_price'].append(selling_price - 3) # Example: Cost Price is a bit lower.
            data['sale_value'].append(selling_price * quantity) # example value, can be computed.
            data['slot_charges'].append(0.0)
            data['is_indent'].append('N')
            data['subscription_id'].append(9999) # example
            data['sales_without_delivery_charge'].append(selling_price * quantity)
            data['discount_amount'].append(0.0)
            data['is_free_coupon_product'].append('N')
            data['user_source'].append('bbDaily')
            data['delivery_status'].append('Delivered')
            data['society_id'].append(123) #example
            data['block_name'].append('A')
            data['tag'].append('Example Tag')
            data['order_ver'].append('v2 orders')
            data['bb_order_id'].append(1234567890) #example
            data['fo_customer'].append('Y')
            data['cost_price_processed'].append(selling_price - 3)

    return pd.DataFrame(data)

# --- 1.a  Define Refund Reason Categories and Mapping---
refund_reasons = {
    "LESS_PACKED_QUANTITY": "Quantity",
    "Unable to locate address": "Delivery",
    "V2 Open Orders": "Order Management",
    "Leakage / Damage": "Quality",
    "Wrong Product": "Order Issue",
    "Quality issue": "Quality",
    "Marked Delivered But Not Received": "Delivery",
    "Partial delivery": "Delivery",
    "Delay in delivery": "Delivery",
    "Grammage difference": "Quality",
    "EXCESS_PACKED_QUANTITY": "Quantity",
    "Product Not Delivered": "Delivery",
    "Not available": "Availability",
    "Weight Variance": "Quality",
    "OPS request": "Order Management",
    "Wrongly ordered": "Order Issue",
    "MRP issue": "Pricing",
    "Expired product": "Quality",
    "Free product not delivered": "Order Issue",
    "Open order refund": "Order Management",
    "Coupon code issue": "Pricing",
    "damaged": "Quality",
    "missing items": "Quantity",
    "Closing the society": "Other"
}

# Helper function to normalize probabilities, ensuring they sum to 1
def get_probabilities(reasons):
    """
    Generates a list of probabilities for the given refund reasons.
    """
    num_reasons = len(reasons)
    # Define the base probabilities. Adjust these as needed to reflect your data
    # Changed these to all equal to avoid the problem with the sum not equaling 1.
    base_probability = 1.0 / num_reasons  # Equal probability for each reason
    probabilities = [base_probability] * num_reasons

    return probabilities



# --- 2. Data Processing & Fraud Detection Functions ---
@st.cache_data  # Use caching to avoid re-processing on every rerun
def process_data(df: pd.DataFrame):

    # --- Preprocessing ---
    # --- Convert string-formatted dates to datetime objects ---
    date_columns = ['order_date', 'Report_date']  # Add all date columns here.

    for col in date_columns:
      try:
          df[col] = pd.to_datetime(df[col], format='%d-%m-%Y')
      except ValueError:
          st.warning(f"Could not parse the date format for column '{col}'. Please check the format in your CSV file.  Expected format: DD-MM-YYYY")
          return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()  # Return empty DataFrames to avoid processing errors


    # ---  Clean and Categorize Refund Reasons ---
    # Ensure refund_reason is a string (important for consistency) and handle missing values
    df['refund_reason'] = df['refund_reason'].astype(str)  # Convert to string
    df['refund_reason'] = df['refund_reason'].str.title() # Standardize capitalization

    #Map refund reasons to categories
    df['refund_category'] = df['refund_reason'].map(refund_reasons).fillna("Other")

    # --- Calculate Metrics ---
    df['sale_value'] = df['quantity'] * df['selling_price']  # Calculate sale value
    #df['refund_to_order_ratio'] = (df['refund_amount'] / df['sale_value']) * 100 # use sale_value
    # IMPORTANT: Ensure there are no division by zero errors
    df['refund_to_order_ratio'] = df.apply(lambda row: (row['refund_amount'] / row['sale_value']) * 100 if row['sale_value'] > 0 else 0, axis=1)


    customer_stats = df.groupby('customer_id').agg(
        total_orders=('order_id', 'count'),
        total_refunds=('refund_amount', 'sum'),
        num_refunds=('refund_amount', 'count'),
        total_order_value=('sale_value', 'sum') # use sale value
    )
    customer_stats['refund_ratio'] = (customer_stats['total_refunds'] / customer_stats['total_order_value']) * 100
    customer_stats['refund_to_order_percentage'] = (customer_stats['num_refunds'] / customer_stats['total_orders']) * 100

    # Calculate the count for each category of refund reason.
    category_counts = df.groupby(['customer_id', 'refund_category']).size().unstack(fill_value=0)
    return df, customer_stats, category_counts  # Return the category counts

@st.cache_data(show_spinner=True)
def detect_fraud(customer_stats: pd.DataFrame, threshold_config: dict):
    # --- Apply Rules ---
    # Check if 'num_refunds' exists in customer_stats columns
    if 'num_refunds' not in customer_stats.columns:
        st.warning("The 'num_refunds' column is missing in the data.  Check your data or processing.")
        return pd.DataFrame() # Return an empty DataFrame to avoid errors

    if 'refund_ratio' not in customer_stats.columns:
        st.warning("The 'refund_ratio' column is missing in the data. Check your data or processing.")
        return pd.DataFrame()  # Return an empty DataFrame

    if 'refund_to_order_percentage' not in customer_stats.columns:
        st.warning("The 'refund_to_order_percentage' column is missing in the data. Check your data or processing.")
        return pd.DataFrame()  # Return an empty DataFrame


    fraud_customers = customer_stats[
        (customer_stats['num_refunds'] >= threshold_config["num_refunds"]) |  # Rule 1
        (customer_stats['refund_ratio'] >= threshold_config["refund_ratio"]) |  # Rule 2
        (customer_stats['refund_to_order_percentage'] >= threshold_config["refund_to_order_percentage"])  # Rule 3
    ]

    fraud_customers = fraud_customers.sort_values(by=["refund_ratio"], ascending=False)  # Sort to show most suspicious customers first.

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
        try:
            df = pd.read_csv(upload_data)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            df = pd.DataFrame() # Reset to empty dataframe on error
    else:
        st.warning("Please upload a CSV file or download the template.")
        # --- CSV Template Download Option ---
        # Create the template with all columns *except* refund_amount and refund_reason
        template_data = {
            'Report_date': ['07-08-2025', '07-08-2025', '07-08-2025'],
            'customer_id': [2446017, 2192296, 2192296],
            'bb_id': [10868036, 6265092, 6265092],
            'order_id': [1175332450, 1175332457, 1175332458],
            'TYPE': ['Subscription', 'Subscription', 'Subscription'],
            'DC_name': ['Ahmedabad-DC', 'Chennai-DC', 'Chennai-DC'],
            'fc_name': ['Ahmedabad-FV-FMCG-DC', 'Chennai-FV-DC', 'Chennai-FV-DC'],
            'fc_id': [213, 207, 207],
            'dark_store': ['Thaltej V2 DS', 'Kelambakkam V2 DS', 'Kelambakkam V2 DS'],
            'category': ['Milk', 'Breakfast, Snacks & Branded Foods', 'Milk'],
            'Hub': ['Thaltej V2 Hub', 'Kelambakkam V2 Hub', 'Kelambakkam V2 Hub'],
            'society_name': ['Shaligram Plush', 'Pacifica Aurum happiness tower', 'Pacifica Aurum happiness tower'],
            'sub_category': ['All Milk', 'Biscuits & Cookies', 'All Milk'],
            'skuid': [40090894, 40174324, 40151383],
            'brand': ['Amul', 'Britannia', 'Aavin'],
            'product_name': ['Taaza Milk', 'JimJam Flavoured Sandwich Biscuits', 'Pasteurised Standardised Milk'],
            'pack_size': ['500 ml', '57 g', '500 ml Pouch'],
            'hsn': ['04012000_a', '19053290_d', '04012000_a'],
            'GST_percent': [0, 18, 0],
            'Cess_percent': [0, 0, 0],
            'sales_without_tax': [56, 8.47, 22],
            'SGST_Value': [0, 0.76, 0],
            'CGST_Value': [0, 0.76, 0],
            'IGST_Value': [0, 0, 0],
            'CESS_Value': [0, 0, 0],
            'order_date': ['05-08-2025', '05-08-2025', '05-08-2025'],
            'quantity': [2, 1, 1],
            'mrp': [28, 10, 22],
            'cost_price': [24.89, 8.4, 21.53],
            'selling_price': [28, 10, 22],
            'sale_value': [56, 10, 22],
            'slot_charges': [0, 0, 0],
            'is_indent': ['N', 'Y', 'N'],
            'refund_comment': ['', '', ''],
            'subscription_id': [9613982, 16113215, 16110043],
            'sales_without_delivery_charge': [56, 10, 22],
            'discount_amount': [0, 0, 0],
            'is_free_coupon_product': [0, 0, 0],
            'user_source': ['bbDaily', 'bbDaily', 'bbDaily'],
            'delivery_status': [1, 1, 1],
            'society_id': [33535, 7931, 7931],
            'block_name': ['B', 'B', 'B'],
            'tag': ['', '', ''],
            'order_ver': ['v2 orders', 'v2 orders', 'v2 orders'],
            'bb_order_id': [1753822588, 1753785814, 1753785404],
            'fo_customer': ['N', 'N', 'N'],
            'cost_price_processed': [0, 8.4, 0]
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
    df = pd.DataFrame()  # Ensure df exists even with an error.

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
    df, customer_stats, category_counts = process_data(df)  # Get category counts

    # --- Detect Fraud ---
    if not customer_stats.empty:
        fraud_customers = detect_fraud(customer_stats, threshold_config)

        # --- Display Results ---
        if not fraud_customers.empty:
            st.subheader("Potential Fraud Customers")
            st.dataframe(fraud_customers)

            # Detailed View (for more in-depth analysis)
            customer_id_to_investigate = st.selectbox("Select Customer ID for Details", fraud_customers.index.tolist())
            if customer_id_to_investigate:
                customer_details = df[df['customer_id'] == customer_id_to_investigate].sort_values(by="order_date", ascending=False)  # show most recent first
                st.subheader(f"Customer {customer_id_to_investigate} Order/Refund History")
                st.dataframe(customer_details)

                # Display refund reason counts by category
                st.subheader("Refund Reason Breakdown")
                if customer_id_to_investigate in category_counts.index:
                    category_data = category_counts.loc[customer_id_to_investigate]
                    st.bar_chart(category_data)
                else:
                    st.write("No refund reasons found for this customer.")
        else:
            st.success("No suspicious activity detected based on current thresholds.")
    else:
        st.write("No customer data to analyze.")
