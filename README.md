# Food Expiry Predictor

A Streamlit app that trains a Bayesian network from food-storage data and estimates a food's shelf-life category: **Short** (up to 7 days), **Medium** (8–15 days), or **Long** (more than 15 days).

> Predictions are estimates, not food-safety advice. Always follow product labels and local food-safety guidance.

## Run locally

1. Install Python 3.12.
2. Install the dependencies from `requirements.txt`.
3. Start the app with `streamlit run app.py`.
4. Choose **Synthetic demo data** to preview it, or upload your own CSV.

## CSV format

The CSV must include these columns:

| Column | Expected values |
| --- | --- |
| `Food_type` | Food category, such as Dairy or Produce |
| `Storage_condition` | Storage category, such as Refrigerated or Pantry |
| `Packaging` | Packaging category, such as Sealed or Opened |
| `Temperature` | Numeric measurement |
| `Humidity` | Numeric measurement |
| `Expiry_days` | Numeric shelf life in days |

Rows with missing or invalid required values are excluded. Temperature and humidity are discretized into up to three levels based on the uploaded dataset. The Bayesian network is trained from the selected data each time the app runs.

The repository currently does not include `food_expiry_dataset.csv`; use your own dataset or the clearly labeled synthetic demo data.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), sign in with GitHub and select **Create app**.
3. Choose this repository, the `main` branch, and `app.py` as the entry point.
4. Deploy. The dependencies are specified in `requirements.txt`.
