# Face Recognition ML pipeline
This project is composed by three different components: feature pipeline, model retraining and inference pipeline. The pipeline diagram can be seen below. 
![Interactive UI for inference (1)](https://user-images.githubusercontent.com/48623568/211796472-e0f8f9de-2b70-4ebb-9cff-bf8a6386f3cb.jpg)

## Feature pipeline

This pipeline is hosted on a GCP (Google Cloud Platform) instance and performs the following tasks:
1. Receive an image from the user (from a Gradio UI).
2. Create multiple different versions of the images by applying filters, rotating it, brighten it, etc...
3. For each of the resulting images, generate an embedding using the [Histogram of Gradients (HOG)](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients) method
4. Upload the original image and the embeddings to a GCP bucket. 
5. Trigger model retraining.

## Model retraining

Once the training pipeline is called, the embeddings are taken from the bucket, and the training of the KNN classifier is performed. The model is then stored on the local device as our inference pipeline uses the same GCP instance to run the inference. 

## Inference pipeline

Our inference pipeline is in real-time, hence the Gradio UI shows the live webcam with all the predictions on each face. Note: This is not a very well trained model, and hence there would be several false positives. 


## Gradio public links:
- The app to upload an image is provided [here](https://1d5d09cf-5ef3-47f6.gradio.live/)
- The app to run the real-time inference is [here](https://c3c30239-c028-48a3.gradio.live/)
## Project structure

- The dataset is stored in `./dataset/<first name of the person>`.
- The feature pipeline creates a 128d vector for each image in the dataset folder along with some augmentations. Once this is done, the encodings are stored in the file `./encodings.pkl`.
- The training pipeline then makes use of the encoding file and trains a knn classifier and stores the model in the file `./model.pkl`. The training file also stores the labels (before using a label encoder) in the file 'labels.pkl'.
- The inference pipeline makes use of all these files and runs the inference in the gradio UI.
