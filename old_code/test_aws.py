import dotenv
dotenv.load_dotenv()   

import boto3
import os
import json
import pandas as pd

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
    )  # S4 supports path-style.[web:11]


print("Buckets:")
for b in s3.list_buckets()["Buckets"]:
    print(" -", b["Name"])


# # 2. Upload a test object of image type
# test_key = "old lady.jpg"
# test_body = open("old lady.jpg", "rb").read()  # read local file as bytes

# s3.put_object(Bucket=BUCKET_NAME, Key=test_key, Body=test_body, ContentType="image/jpeg")  # set content type for correct handling
# print(f"Uploaded object s3://{BUCKET_NAME}/{test_key}")





# url = "https://images-na.ssl-images-amazon.com/images/M/MV5BMjM3ODk1NjEzOV5BMl5BanBnXkFtZTgwOTAyNjU1MzE@._V1_.jpg"
# index = 0
