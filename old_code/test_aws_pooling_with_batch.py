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
# NOTE: done with up to 1250000
    # 12 seconds per 10000, 2 minutes per 100,000; about 32 minutes for everything(1.6 million rows)
    # from 100 hours to 32 minutes by batching and parallelism
START_INDEX = 1290000
END_INDEX = len(urls) - 1  # inclusive

MAX_WORKERS = 32          # thread pool for sending batches
BATCH_SIZE = 100          # urls per Lambda invocation


def invoke_batch(batch_items):
    """
    Invoke one Lambda asynchronously with a batch of items.
    batch_items: list of dicts: [{"url": ..., "index": ...}, ...]
    """
    payload = {"items": batch_items}
    resp = lambda_client.invoke(
        FunctionName="process_image_urls_to_MEGA",
        InvocationType="Event",   # async
        Payload=json.dumps(payload),
    )
    # For logging, return first/last index and status
    indices = [item["index"] for item in batch_items]
    return (min(indices), max(indices)), resp.get("StatusCode")


def main():
    url_set = set()
    current_name = None
    batches = []
    current_batch = []

    # Build items with url_set + name-based reset
    for index, (url, name) in enumerate(zip(urls, names)):
        if index < START_INDEX or index > END_INDEX:
            continue

        # Detect name change and reset url_set
        if current_name is None:
            current_name = name
            url_set.clear()
            print(f"Starting new name group: {current_name}")
        elif name != current_name:
            current_name = name
            url_set.clear()
            print(f"\nName changed, resetting url_set. New name: {current_name}")

        # Skip duplicate URLs within the current name group
        if url in url_set:
            continue

        url_set.add(url)

        # Add to current batch
        current_batch.append({"url": url, "index": int(index)})

        if len(current_batch) >= BATCH_SIZE:
            batches.append(current_batch)
            current_batch = []

    # Flush last partial batch
    if current_batch:
        batches.append(current_batch)

    total_items = sum(len(b) for b in batches)
    print(
        f"Prepared {len(batches)} batches "
        f"({total_items} unique URLs, indexes {START_INDEX}-{END_INDEX})"
    )

    t0 = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(invoke_batch, batch) for batch in batches]

        for fut in as_completed(futures):
            (min_idx, max_idx), status = fut.result()
            print(
                f"Invoked Lambda for batch indexes {min_idx}-{max_idx}, "
                f"StatusCode={status}"
            )

    t1 = time.time()
    print(f"Done. Elapsed time: {t1 - t0:.2f} seconds")


if __name__ == "__main__":
    main()
