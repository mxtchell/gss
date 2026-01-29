"""
Test AnswerRocket API Connection & Query Dataset
=================================================

SETUP:
  - Python 3.10 required (answer_rocket SDK only works with 3.10)
  - Python path: /Users/mitchelltravis/cursor/.direnv/python-3.10/bin/python3
  - SDK reads env vars: AR_URL and AR_TOKEN (not AR_BASE_URL / AR_API_KEY)

ENV VARS:
  AR_URL=https://reckitt.poc.answerrocket.com
  AR_TOKEN=<your token>

KEY LEARNINGS:
  - DATABASE_ID is the shared DB connection ID, same across datasets
  - DATASET_ID is per-CSV, but you do NOT pass it to execute_sql_query
  - execute_sql_query takes DATABASE_ID, not DATASET_ID
  - Query CSVs with: SELECT * FROM read_csv('filename.csv')

USAGE:
  AR_URL=https://reckitt.poc.answerrocket.com AR_TOKEN=arc-xxx /Users/mitchelltravis/cursor/.direnv/python-3.10/bin/python3 test_ar_connection.py
"""
import os
from answer_rocket import AnswerRocketClient

# Shared DB connection (same for all datasets on this instance)
DATABASE_ID = "B855F1B7-35EA-46E1-B1D7-1630EEA5CA82"

# Dataset IDs (for reference only - not used in SQL queries)
DATASETS = {
    "gss": {"id": "d762aa87-6efb-47c4-b491-3bdc27147d4e", "table": "read_csv('gss_max_ready_2026_v6.csv')"},
    "vms": {"id": "99a835f3-76c4-4241-877d-9ca0b24ac5b8", "table": "read_csv('reckitt_vms_survey_long.csv')"},
}

print("=" * 60)
print("AnswerRocket Connection Test")
print("=" * 60)

url = os.getenv("AR_URL")
token = os.getenv("AR_TOKEN")
print(f"URL:   {url}")
print(f"Token: {token[:25]}..." if token else "Token: NOT SET")
print(f"DB ID: {DATABASE_ID}")

client = AnswerRocketClient()
print("Client created.\n")

for name, ds in DATASETS.items():
    print(f"--- {name.upper()} ({ds['table']}) ---")
    sql = f"SELECT COUNT(*) as row_count FROM {ds['table']}"
    try:
        result = client.data.execute_sql_query(DATABASE_ID, sql, row_limit=5)
        if result.success and hasattr(result, "df") and result.df is not None:
            count = result.df["row_count"].iloc[0]
            print(f"  OK: {count:,} rows\n")
        else:
            print(f"  FAIL: {getattr(result, 'error', 'No data returned')}\n")
    except Exception as e:
        print(f"  ERROR: {e}\n")

print("=" * 60)
