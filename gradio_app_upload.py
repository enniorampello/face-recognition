import os
import random
import gradio as gr

from google.cloud import storage
from google.oauth2 import service_account

from feature_pipeline import preprocess
from training_pipline import train

use_bucket = False

def upload_to_bucket(client, bucket_name, file_name, object_name):
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(object_name)
        blob.upload_from_filename(file_name)

        return True
    except Exception as e:
        print(e)
        return False


def image_upload(name, img):
    file_id = random.randint(0, 1000000)
    
    if os.path.exists(f"./dataset/{name}") == False:
        os.mkdir(f"./dataset/{name}")
    
    img.save(f"dataset/{name}/{name}_{file_id}.jpg")

    if use_bucket:
        creds = service_account.Credentials.from_service_account_file("credentials.json")
        client = storage.Client(credentials=creds)
        upload_to_bucket(client, "bucket-faces", f"./dataset/{name}/{name}_{file_id}.jpg", f"{name}/{name}_{file_id}.jpg")

    preprocess()
    train()

    return "Model retrained!"
        

gr.Interface(
    image_upload, 
    [
        gr.inputs.Textbox(placeholder="write your name here..."), 
        gr.inputs.Image(type="pil", label="Input")
    ], 
    outputs="text"
    ).launch(share=True)