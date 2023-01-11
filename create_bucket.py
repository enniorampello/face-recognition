from google.cloud import storage
from google.oauth2 import service_account

creds = service_account.Credentials.from_service_account_file("credentials.json")
client = storage.Client(credentials=creds)

def create_bucket(bucket_name):
    try:
        bucket = client.create_bucket(bucket_name)
        print(f"Bucket {bucket.name} created")
        return True
    except Exception as e:
        print(e)
        return False

create_bucket("bucket-embeddings")