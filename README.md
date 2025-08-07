# Fraud Detection Tool for Daily Subscription Deliveries

This Streamlit application helps identify customers who may be abusing the refund system in a daily subscription-based delivery service.  It analyzes order and refund data to detect patterns indicative of fraudulent activity.

## Features

*   **Data Input:**
    *   Simulated data for testing and demonstration.
    *   CSV file upload for real data analysis.
*   **Fraud Detection:**
    *   Automated rules based on refund frequency, refund ratio, and other metrics.
    *   Configurable thresholds for rule sensitivity.
*   **Results Display:**
    *   Interactive tables to display potential fraud customers.
    *   Detailed order/refund history for selected customers.
*   **User Interface:**
    *   Streamlit-based interface for easy interaction.
    *   Sidebar controls for data source selection and threshold configuration.

## How to Run

1.  **Install Python and Pip:** Make sure you have Python and pip installed on your system or within your virtual environment if working locally.
2.  **Create a Project Directory:**  Create a new directory for your project (e.g., `fraud_detection_app`).
3.  **Place Files:**  Save the following files into the project directory:
    *   `fraud_detector.py`:  (The main Python script containing the Streamlit app code - provided in previous responses.)
    *   `requirements.txt`:  (Specifies project dependencies, see content above.)
    *   `README.md`:  (This file, see content above.)
4.  **Install Dependencies:** Open a terminal or command prompt, navigate to your project directory, and run:
    ```bash
    pip install -r requirements.txt
    ```
5.  **Run Locally (Optional):**  To test locally before deploying to Streamlit Cloud:
    ```bash
    streamlit run fraud_detector.py
    ```
    This will open the app in your web browser.
6.  **Deploy to Streamlit Cloud:**
    *   Create a GitHub repository (or use an existing one) and upload the `fraud_detector.py`, `requirements.txt`, and `README.md` files.
    *   Go to [Streamlit Cloud](https://streamlit.io/cloud) and click "New App."
    *   Select the repository where you saved your code.
    *   Choose the branch (usually "main" or "master").
    *   Specify the file path to your Python script (e.g., `fraud_detector.py`).
    *   Click "Deploy!"

## Data Requirements

*   The CSV data (if used) should include columns for:
    *   `customer_id`: Unique customer identifier.
    *   `order_id`: Unique order identifier.
    *   `order_date`: Date of the order (YYYY-MM-DD format).
    *   `order_amount`: Total amount of the order.
    *   `refund_request_date`: Date the refund was requested (YYYY-MM-DD format) - can be blank if there was no refund.
    *   `refund_amount`: Amount of the refund.
    *   `refund_reason`:  Reason for the refund (e.g., "Missing item", "Damaged item").

## Configuration

*   **Thresholds:** The app allows you to configure the thresholds for identifying potential fraud. Adjust these in the sidebar of the app. You may need to experiment with different thresholds to optimize the balance between detecting fraud and minimizing false positives.
*   **Data Source:** Select either "Simulated Data" (for testing) or "Upload CSV" to use your own data.

## Future Improvements

*   Connect to a live database to access real-time order/refund data.
*   Implement machine learning models for more sophisticated fraud detection.
*   Add alert notifications (e.g., email, Slack) when potential fraud is detected.
*   Integrate with existing customer management systems.

## License

This project is licensed under the [MIT License](LICENSE). (If you want a license file, create one and add the link).

## Contact

[Your Name/Email/GitHub Link]
