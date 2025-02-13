Recognizing UVA Landmarks with Neural Networks

![UVA Grounds](https://giving.virginia.edu/sites/default/files/2019-02/jgi-teaser-image.jpg)

Dataset Overview
This project utilizes a dataset of images of UVA landmarks for classification. The dataset consists of:
- Images: Various UVA buildings and landmarks.
- Labels: Each image is labeled with its corresponding UVA landmark name.
- Splits: The dataset is divided into training, validation, and test sets.

Data Preprocessing
- Image Resizing: Standardized dimensions for CNN input.
- Normalization: Pixel values are normalized to [0,1].
- Augmentation: Random rotations, flips, and shifts to improve generalization.


Project Goal
The goal is to train a neural network to classify UVA buildings and landmarks from images. The project is part of Codeathon 2 and focuses on deep learning for image classification.

 Technologies Used
- Python 
- TensorFlow / Keras (Deep Learning Framework)
- scikit-learn (Data Preprocessing)
- NumPy & Pandas (Data Handling)
- Matplotlib & Seaborn (Visualization)

How to Run the Project
Clone the Repository
```bash
git clone https://github.com/yourusername/UVA-landmark-classification.git
cd UVA-landmark-classification
Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
Run the Jupyter Notebook
bash
Copy
Edit
jupyter notebook
Open the notebook and execute the cells to train the model.

Model Architecture
The project uses a Convolutional Neural Network (CNN) with:

Multiple convolutional layers
ReLU activations for feature extraction
MaxPooling layers for dimensionality reduction
Dropout layers for regularization
Softmax activation for classification
Training Details
Optimizer: Adam
Loss Function: Categorical Crossentropy
Evaluation Metrics: Accuracy, Confusion Matrix
Results & Performance
Achieved above 85% accuracy on custom CNN and above a 95% accuracy on the pre-trained model.
Confusion Matrix & Accuracy Graphs for model evaluation.
Next Steps
Improve accuracy using hyperparameter tuning.
Implement transfer learning with models like ResNet.
Deploy as a web app for real-time landmark classification.
Acknowledgments
University of Virginia for organizing this Codeathon.
