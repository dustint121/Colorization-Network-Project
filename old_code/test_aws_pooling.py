import dotenv
dotenv.load_dotenv()

import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import pandas as pd

# ---- Existing MEGA / S3 setup ----

ACCESS_KEY = os.getenv("MEGA_ACCESS_KEY")
SECRET_KEY = os.getenv("MEGA_SECRET_KEY")
S4_ENDPOINT = os.getenv("MEGA_S4_ENDPOINT")
S4_REGION = os.getenv("MEGA_S4_REGION")
BUCKET_NAME = os.getenv("MEGA_BUCKET_NAME")

session = boto3.session.Session()
s3 = session.client(
    service_name="s3",
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    endpoint_url=S4_ENDPOINT,
    region_name=S4_REGION,
)

# ---- Lambda client ----

aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

lambda_client = boto3.client(
    "lambda",
    region_name="us-west-1",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
)

# ---- Load data ----

df = pd.read_csv("IMDb-Face.csv")
urls = df["url"].tolist()
names = df["name"].tolist()

num_urls = len(urls)
print(f"Total URLs: {num_urls}")

# ---- Range to process ----
# NOTE: done with up to 120000
START_INDEX = 120000
END_INDEX = 150000  # inclusive

# Tune this based on your environment and Lambda concurrency
MAX_WORKERS = 32


def invoke_one(args):
    """Invoke one Lambda asynchronously."""
    index, url = args
    payload = {
        "url": url,
        "index": int(index),
    }

    resp = lambda_client.invoke(
        FunctionName="process_image_urls_to_MEGA",
        InvocationType="Event",
        Payload=json.dumps(payload),
    )
    return index, resp.get("StatusCode")


def main():
    url_set = set()
    current_name = None
    items = []

    # Build items with url_set + name-based reset
    for index, (url, name) in enumerate(zip(urls, names)):
        if index < START_INDEX or index > END_INDEX:
            continue

        # Detect name change and reset url_set
        if current_name is None:
            current_name = name
            url_set.clear()
            # print(f"Starting new name group: {current_name}")
        elif name != current_name:
            current_name = name
            url_set.clear()
            print(f"Name changed, resetting url_set. New name: {current_name}")

        # Skip duplicate URLs within the current name group
        if url in url_set:
            continue

        url_set.add(url)
        items.append((index, url))

    print(
        f"Invoking Lambda for {len(items)} unique URLs "
        f"(indexes {START_INDEX}-{END_INDEX})\n"
    )
    sleep_time = 5  # Adjust as needed to avoid hitting rate limits
    time.sleep(sleep_time)

    t0 = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(invoke_one, item) for item in items]
        for fut in as_completed(futures):
            index, status = fut.result()
            print(f"Invoked Lambda for index {index}, StatusCode={status}")

    t1 = time.time()
    print(f"Done. Elapsed time: {t1 - t0:.2f} seconds")


if __name__ == "__main__":
    main()
