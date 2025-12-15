A real-time hand sign recognition system built as a minor academic project, using MediaPipe for hand joint detection and MobileNetV2 (TensorFlow) for machine learning–based classification. 
The system is designed to run locally and demonstrates the complete pipeline from crating dataset to live prediction.
Key features-
1. Real-time hand detection using MediaPipe
2.Image-based sign classification using MobileNetV2
3.Model training and fine-tuning pipeline
4. Live prediction with confidence scores

IMPORTANT NOTE-
1.Dataset used to train the model is not included as the dataset consists of a large number of raw and augmented images, making it impractical to push to GitHub.
2.The initially trained model is NOT included.
  Reason: The base model suffered from underfitting due to limited dataset size

HOW TO RUN-
Method 1 -> This method is used to directly use the recogniton model it does not require any new dataset.
Clone the files named requirements, static, templates, app.py
1. Clone all the files mentioned above in the same folder.
2. Install Dependencies ->pip install -r requirements.txt.
3. Run the app.py file then go to the locally hosted website.

Method 2-> Use this method if you want to make your own custom dataset and train the model from scratch.
1. Clone all the files in the repository into the same folder.
2. Install Dependencies (using pip install -r requirements.txt).
3. Capture Dataset (using python capture_dataset.py) -> This will help you capture your own data.
4. Augment Dataset (python augment_dataset.py) -> Augementing data is neccessary as it provides large number of variations for the model to learn from.
5. Train / Fine-Tune Model (python train_model.py, python fine_tune_model.py) -> First traint the model then fine tune it to increase it's accuracy.
6. Live Prediction (python live_prediction.py) -> This can directly show your model.
7. Use the Website (python app.py) -> This file will start a locally hosted website for the model.


