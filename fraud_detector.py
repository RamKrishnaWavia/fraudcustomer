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
        'Report_date': [],  # Add Report_date
        'customer_id': [],  # Changed to customer_id
        'order_id': [],
        'order_date': [], # changed to order_date
        'Refund_Date': [], # Added Refund_Date
        'quantity': [], # Renamed to refund_quantity
        'selling_price': [],  # add selling_price for sale_value calculation
        'comment': [],  #  Changed to comment = refund reason
        'refund_amount': [], # changed to amount
        'sub_id': [], # added sub_id
        # Add all the other columns, potentially with dummy data - based on the NEW template
        'city': [],
        'DC': [],
        'DS': [],
        'hub': [],
        'agent_name': [],
        'role': [],
        'society_name': [],
        'block': [],
        'flat_no': [],
        'bb_id': [],  # Example: Add dummy data
        'customer_name': [],
        'customer_category': [],
        'bbdaily_category': [],
        'brand': [],
        'product_id': [],
        'product': [],
        'pack_size': [],
        'cee': [],
        'is_fo_customer': [],
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
                comment = np.random.choice(list(refund_reasons.keys()), p=get_probabilities(list(refund_reasons.keys()))) if refund_amount > 0 else None
            else:
                refund_amount = 0
                refund_request_date = None
                comment = None

            data['Report_date'].append(datetime.now().strftime('%d-%m-%Y'))  #  Today's Date
            data['customer_id'].append(customer_id)
            data['order_id'].append(customer_id * 1000 + order_num)
            data['order_date'].append(order_date.strftime('%d-%m-%Y'))  # Correct date format
            data['Refund_Date'].append(refund_request_date.strftime('%d-%m-%Y') if refund_request_date else '') # Add Refund_Date
            data['quantity'].append(quantity) # Renamed to refund_quantity
            data['selling_price'].append(selling_price) # Include selling_price
            data['comment'].append(comment) # Changed to refund_comment
            data['refund_amount'].append(refund_amount)  # Changed to amount
            data['sub_id'].append(12345)

            # Add dummy data for the other columns. - Based on the new template
            data['city'].append('Example City')
            data['DC'].append('Example DC')
            data['DS'].append('Example DS')
            data['hub'].append('Example Hub')
            data['agent_name'].append('Agent Name')
            data['role'].append('Agent Role')
            data['society_name'].append('Example Society')
            data['block'].append('A')
            data['flat_no'].append('101')
            data['bb_id'].append(12345)  # Example
            data['customer_name'].append('Customer Name')
            data['customer_category'].append('Regular')
            data['bbdaily_category'].append('Milk')
            data['brand'].append('Example Brand')
            data['product_id'].append(98765)
            data['product'].append('Example Product')
            data['pack_size'].append('Example Pack')
            data['cee'].append('Y')
            data['is_fo_customer'].append('Y')

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
    date_columns = ['order_date', 'Report_date', 'Refund_Date']  # Add all date columns here.
    #st.write(df.dtypes) # Debugging - print the dtypes before the conversion
    for col in date_columns:
        if col in df.columns: # Check if column exists
            try:
                df[col] = pd.to_datetime(df[col], format='%d-%m-%Y', errors='coerce')  #  Handle potential errors
            except (ValueError, TypeError) as e:  # Catch both ValueError and TypeError
                st.error(f"Could not parse the date format for column '{col}'. Please check the format in your CSV file.  Expected format: DD-MM-YYYY.  Error: {e}")
                #st.stop() # Stop execution if the format is wrong
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()  # Return empty DataFrames to avoid processing errors
        else:
            st.warning(f"Column '{col}' not found in the uploaded data.  Skipping date conversion for this column.")


    # ---  Clean and Categorize Refund Reasons ---
    # Ensure comment is a string (important for consistency) and handle missing values
    if 'comment' in df.columns: # changed to comment
        df['refund_reason'] = df['comment'].astype(str)  # Convert to string
        df['refund_reason'] = df['refund_reason'].str.title() # Standardize capitalization
    else:
        st.warning("The 'comment' column is missing in the data. Skipping refund categorization.")

    #Map refund reasons to categories
    if 'refund_reason' in df.columns:
        df['refund_category'] = df['refund_reason'].map(refund_reasons).fillna("Other")
    else:
        df['refund_category'] = "Other"  # or create an empty column

    # --- Calculate Metrics ---
    #df['sale_value'] = df['quantity'] * df['selling_price']  # Calculate sale value  --> now calculating in generate_sample_data()
    # IMPORTANT: Ensure there are no division by zero errors
    if 'amount' in df.columns and 'selling_price' in df.columns and 'quantity' in df.columns:
        df['refund_to_order_ratio'] = df.apply(lambda row: (row['amount'] / (row['selling_price'] * row['quantity'])) * 100 if (row['selling_price'] * row['quantity']) > 0 else 0, axis=1)
    else:
        df['refund_to_order_ratio'] = 0  # or create an empty column

    customer_stats = pd.DataFrame()
    category_counts = pd.DataFrame()

    if not df.empty:
      # Filter the DataFrame to include only rows where 'customer_id' is not null
      df_filtered = df[df['customer_id'].notna()]
      if not df_filtered.empty: # Make sure there is data.
        customer_stats = df_filtered.groupby('customer_id').agg(
            total_orders=('order_id', 'count'),
            total_refunds=('amount', 'sum'), #changed to amount
            num_refunds=('amount', 'count'), # changed to amount
            total_order_value=('sale_value', 'sum') # use sale value
        )
        if not customer_stats.empty: # Checking if the aggregation resulted in an empty df.
            customer_stats['refund_ratio'] = (customer_stats['total_refunds'] / customer_stats['total_order_value']) * 100
            customer_stats['refund_to_order_percentage'] = (customer_stats['num_refunds'] / customer_stats['total_orders']) * 100

            # Calculate the count for each category of refund reason.
            if 'refund_category' in df.columns:
                category_counts = df_filtered.groupby(['customer_id', 'refund_category']).size().unstack(fill_value=0)

    return df, customer_stats, category_counts  # Return the category counts

@st.cache_data(show_spinner=True)
def detect_fraud(customer_stats: pd.DataFrame, threshold_config: dict):
    # --- Apply Rules ---
    # Check if 'num_refunds' exists in customer_stats columns
    if not customer_stats.empty:  # Check if customer_stats is not empty
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
    else:
        return pd.DataFrame() # Return empty DF if customer_stats is empty

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
            'customer_id': ['2446017', '2192296', '2192296'],
            'order_id': [1175332450, 1175332457, 1175332458],
            'TYPE': ['Subscription', 'Subscription', 'Subscription'],
            'DC_name': ['Ahmedabad-DC', 'Chennai-DC
