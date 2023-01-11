import os
import random
import gradio as gr

from google.cloud import storage
from google.oauth2 import service_account

from feature_pipeline import preprocess
from training_pipline import train

use_bucket = True

creds = service_account.Credentials.from_service_account_file("credentials.json")
client = storage.Client(credentials=creds)

def upload_to_bucket(client, bucket_name, file_name, object_name):
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(object_name)
        blob.upload_from_filename(file_name)

        return True
    except Exception as e:
        print(e)
        return False


def image_upload(_, img, name):
    file_id = random.randint(0, 1000000)
    
    if os.path.exists(f"./dataset/{name}") == False:
        os.mkdir(f"./dataset/{name}")
    
    img.save(f"dataset/{name}/{name}_{file_id}.jpg")

    embeddings_filename = preprocess()

    if use_bucket:
        upload_to_bucket(client, "bucket-faces", f"./dataset/{name}/{name}_{file_id}.jpg", f"{name}/{name}_{file_id}.jpg")
        upload_to_bucket(client, "bucket-embeddings", embeddings_filename, embeddings_filename)

    train(client, "bucket-embeddings", embeddings_filename)

    return "Model retrained!"
        

gr.Interface(
    image_upload, 
    [
        gr.Markdown("""
        # Hello!! Enter your first name and upload one picture of your face.
        ## The face recognition model will be retrained with the knowledge you gave it of your face.
        """),
        gr.Webcam(source="webcam", type="pil", label="Upload a beautiful picture of yourself."),
        gr.Textbox(placeholder="write your name here...", label="Your name.")
    ], 
    outputs="text"
    ).launch(server_name="0.0.0.0")
