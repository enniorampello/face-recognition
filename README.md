# Face Recognition ML pipeline
This project is composed by three different components: feature pipeline, model retraining and inference pipeline.
## Feature pipeline
[//]: <insert image here>
This pipeline is hosted on a [Hugging Face space](https://www.google.com) and performs the following tasks:
1. Receive an image from the user (in Hugging Face).
2. Create multiple different versions of the images by applying filters, rotating it, brighten it, etc...
3. For each of the resulting images, generate an embedding using [<insert model name>](https://www.google.com)
4. Upload the embeddings to a Google Drive folder.
5. Trigger model retraining.
## Model retraining
[//]: <insert image here>
The upload of new embeddings to the Google Drive folder triggers a retraining of the classifier model, which will be updated to classify any new face that it received. 
The model will then be stored as a Hugging Face model once the training is complete.
## Inference pipeline
[//]: <insert image here>
The users can upload a picture of a person in the [Hugging Face space](https://www.google.com) and they will receive a message as output containing either the name of that person or a warning that the person might not be stored in the database.


## general info 
- The dataset is stored in `./dataset/<first name of the person>`
- The feature pipeline creates a 128d vector for each image in the dataset folder along with some augmentations. Once this is done, the encodings are stored in the file `./encodings.pkl`
- The training pipeline then makes use of the encoding file and trains a knn classifier and stores the model in the file `./model.pkl`. The training file also stores the labels (before using a label encoder) in the file 'labels.pkl'
- The inference pipeline makes use of all these files and runs the inference in the gradio UI

[//]: comment