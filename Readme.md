# PathVQA (Visual Question Answering) Web Application

## Overview

This project aims to build a web application for Visual Question Answering (VQA) using deep learning techniques. VQA involves answering questions about images, combining computer vision and natural language processing (NLP) capabilities.

## Dataset

### Domain: MedicalVQA
### Dataset: PathVQA
[PathVQA GitHub Repository](https://github.com/UCSD-AI4H/PathVQA)

In the realm of medical image analysis, the ability to comprehend and interpret pathological images is crucial for diagnosis and treatment planning. To enhance this capability, we propose a deep learning project focused on Medical Visual Question Answering (VQA) using the PathVQA dataset. PathVQA is a curated dataset containing question-answer pairs associated with pathology images sourced from authoritative textbooks and digital libraries.

### Problem Statement

Given the PathVQA dataset, the problem is to develop a robust deep learning model that can accurately answer questions based on pathology images. This involves understanding complex medical images, interpreting associated questions, and generating appropriate answers. The challenge lies in handling the diversity of questions, variations in image characteristics, and ensuring the model's reliability in providing accurate answers.

### Challenges

- Pathological images are complex, requiring the model to grasp subtle details and patterns.
- Understanding natural language questions, including medical terminology, adds complexity.
- Varied question types result in ambiguous answers.
- Limited training data poses challenges in capturing the full diversity of pathological cases.
- Ensuring ethical use, patient privacy, and compliance with medical regulations are crucial.
- Developing interpretable models to explain the reasoning behind answers is challenging.

### Objectives

- Create a model for answering open-ended and binary questions using pathology images.
- Achieve high accuracy in answering questions across all question types.
- Improve model interpretability to understand its decision-making process better.

### Dataset Understanding

#### Dataset Description

The PathVQA dataset consists of 4,998 pathology images and 32,799 question-answer pairs. These pairs include both open-ended questions and binary "yes/no" questions, sourced from pathology textbooks and digital libraries. The images are referenced by the associated question-answer pairs.

#### Dataset Statistics

**Dataset Characteristics**
- Size: 4,998 images, 32,799 question-answer pairs
- Format: Images in JPEG format, questions, and answers in English text
- Source: Pathology textbooks ("Textbook of Pathology" and "Basic Pathology"), Pathology Education Informational Resource (PEIR)
- License: Released under the MIT License

### Performance Metrics

**Metrics**
- Yes/No Accuracy: Accuracy of model-generated answers for binary "yes/no" questions.
- Free-form Accuracy: Accuracy of model-generated answers for open-ended questions.
- Overall Accuracy: Accuracy of model-generated answers across all question types.

**Evaluation**
- Training Set: 19,703 QAs, 2,835 images
- Validation Set: 6,318 QAs, 1,068 images.
- Test Set: 6,778 QAs, 1095 images.

## Data Preparation

The dataset is preprocessed to extract questions, answers, and image paths. Separate subsets of data are created for questions with yes/no answers and free-form answers. This approach allows training separate models for each question type, potentially improving model performance by focusing on specific types of questions.

## Model Architecture

The VQA model architecture consists of two main components: an image feature extractor and a question feature extractor. The image feature extractor utilizes the VGG-19 architecture pretrained on ImageNet to extract high-level features from images. The question feature extractor employs BERT (Bidirectional Encoder Representations from Transformers) to encode question text into contextual embeddings. Attention mechanisms are used to fuse image and question features, allowing the model to focus on relevant image regions while answering questions.

## Model Training

Two separate models are trained for questions with yes/no answers and free-form answers. This approach allows the models to specialize in predicting different types of answers, potentially improving overall accuracy. The models are trained using the Adam optimizer with a cross-entropy loss function. During training, the models learn to predict answers based on image-question pairs from the training dataset.

## Inference Generation

The trained models are used to generate inferences on new image-question pairs. Given an input image and question, the models predict an answer using the learned features and parameters. The predicted answer is then displayed to the user along with the input image.

## Web Application

The web application allows users to interactively receive predicted answers in real-time. The application is built using Flask, a lightweight web framework for Python. Users can choose between the two trained models based on the type of question they are asking (yes/no or free-form). The application provides a simple and intuitive interface for exploring the VQA capabilities of the models.

## Running the Application

To run the application locally, follow these steps:

1. Clone the GitHub repository to your local machine.
2. Ensure you have Python installed on your system.
3. Install the required Python packages.
4. Navigate to the project directory in the terminal.
5. Run the Flask application by executing the command `python app.py`.
6. Open a web browser and go to `http://localhost:5000` to access the application.
7. Choose the index from the test dataset.
8. Choose the model type (yes/no or free-form) and click the "Submit" button.
9. The predicted answer will be displayed on the screen along with the input image.

## Conclusion

This VQA web application demonstrates the power of deep learning techniques for understanding and answering questions about images. By training separate models for different types of questions, we can achieve higher accuracy and better performance on specific tasks. The application provides a user-friendly interface for exploring the capabilities of the models and can be extended with additional features and improvements in the future.
