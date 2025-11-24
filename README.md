# 🛡️ DGA Domain Detector

A sophisticated deep learning solution for detecting Domain Generation Algorithm (DGA) generated domains using Natural Language Processing techniques.

## 📋 Project Overview

Domain Generation Algorithms (DGAs) are commonly used by malware to generate domain names for command and control servers. This project implements a hybrid CNN-LSTM deep learning model to classify domain names as either legitimate or DGA-generated with high accuracy.

### Key Features

- **Character-level NLP**: Treats domain names as sequences of characters
- **Hybrid Architecture**: Combines CNN for pattern detection with Bidirectional LSTM for sequence learning
- **High Performance**: Achieves ~99% accuracy on the test set
- **Interactive Web Interface**: Simple Streamlit app for real-time predictions
- **Batch Processing**: Analyze multiple domains from CSV files

## 🏗️ Model Architecture

```
1. Embedding Layer (64 dimensions)
2. Conv1D + MaxPooling (128 filters, kernel size 3)
3. Conv1D + MaxPooling (64 filters, kernel size 3)
4. Bidirectional LSTM (64 units)
5. Dense Layer (32 units, ReLU)
6. Output Layer (1 unit, Sigmoid)
```

### Model Performance Metrics

- **Accuracy**: ~99%
- **AUC Score**: ~0.99
- **Precision**: ~99%
- **Recall**: ~99%

## 📁 Project Structure

```
dga-detector/
├── dga_detection_training.ipynb  # Jupyter notebook for training
├── app.py                         # Streamlit web interface
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── dga_websites.csv              # DGA domains dataset
├── legit_websites.csv            # Legitimate domains dataset
│
└── Generated files (after training):
    ├── dga_detector_model.h5             # Trained Keras model
    ├── dga_detector_model_savedmodel/    # TensorFlow SavedModel format
    ├── tokenizer.pkl                     # Character tokenizer
    ├── model_config.pkl                  # Model configuration
    ├── best_dga_model.h5                 # Best checkpoint
    ├── training_history.png              # Training plots
    ├── confusion_matrix.png              # Confusion matrix
    └── roc_curve.png                     # ROC curve
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) GPU with CUDA support for faster training

### Installation

1. **Clone or download the project files**

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Training the Model

1. **Open Jupyter Notebook**:
```bash
jupyter notebook dga_detection_training.ipynb
```

2. **Run all cells** in the notebook to:
   - Load and explore the datasets
   - Preprocess the data
   - Build and train the model
   - Evaluate performance
   - Save model artifacts

**Note**: Training may take 15-30 minutes depending on your hardware. The notebook uses Early Stopping, so it will stop when the model stops improving.

### Running the Web Interface

After training the model, launch the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 💻 Usage

### Web Interface Features

1. **Single Domain Analysis**:
   - Enter a domain name in the text input
   - Click "Analyze Domain"
   - View prediction, confidence score, and domain features

2. **Batch Analysis**:
   - Upload a CSV file with a 'domain' column
   - Click "Analyze All"
   - Download results as CSV

3. **Sample Domains**:
   - Test with pre-loaded legitimate and DGA examples

### Python API Usage

```python
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = tf.keras.models.load_model('dga_detector_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('model_config.pkl', 'rb') as f:
    config = pickle.load(f)

# Predict a domain
def predict_domain(domain):
    char_tokens = ' '.join(list(domain.lower()))
    sequence = tokenizer.texts_to_sequences([char_tokens])
    padded = pad_sequences(sequence, maxlen=config['max_length'], padding='post')
    score = model.predict(padded)[0][0]
    return "DGA" if score > 0.5 else "Legitimate", score

# Example
label, confidence = predict_domain("google")
print(f"Domain: google | Prediction: {label} | Confidence: {confidence:.2%}")
```

## 📊 Dataset Information

### DGA Domains Dataset
- **File**: `dga_websites.csv`
- **Size**: 337,502 samples
- **Source**: Generated domains from various DGA families

### Legitimate Domains Dataset
- **File**: `legit_websites.csv`
- **Size**: 337,400 samples
- **Source**: Real-world legitimate domain names

**Total**: 674,902 domain samples (balanced dataset)

## 🔬 Technical Details

### Character-Level Tokenization

The model treats each character as a token, allowing it to learn character patterns and combinations that distinguish legitimate domains from DGA-generated ones.

### Features Analyzed

- Domain length
- Character entropy
- Unique characters count
- Digit count
- Vowel/consonant ratio
- Special characters

### Training Configuration

- **Batch Size**: 128
- **Epochs**: 20 (with Early Stopping)
- **Optimizer**: Adam
- **Loss Function**: Binary Cross-Entropy
- **Validation Split**: 20%
- **Test Split**: 20%

### Callbacks

- **Early Stopping**: Monitors validation loss with patience of 5 epochs
- **Learning Rate Reduction**: Reduces learning rate when validation loss plateaus
- **Model Checkpoint**: Saves the best model based on validation accuracy

## 📈 Results Visualization

The training notebook generates several visualizations:

1. **Training History**: Shows accuracy, loss, AUC, precision, and recall over epochs
2. **Confusion Matrix**: Visual representation of classification performance
3. **ROC Curve**: Shows the model's true positive vs false positive rate

## 🛠️ Troubleshooting

### Common Issues

**Issue**: Out of memory during training
- **Solution**: Reduce batch size in the training notebook (line with `batch_size = 128`)

**Issue**: Model files not found when running the app
- **Solution**: Make sure you've run the entire training notebook first to generate model files

**Issue**: Slow predictions
- **Solution**: Use GPU acceleration by installing `tensorflow-gpu` or reduce the model complexity

### GPU Support

To use GPU acceleration:
```bash
pip install tensorflow-gpu==2.15.0
```

Verify GPU is available:
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

## 📚 Resources

### What are DGAs?

Domain Generation Algorithms are used by malware to:
- Generate pseudo-random domain names
- Establish communication with C&C servers
- Evade blacklist-based security measures
- Make takedown efforts more difficult

### Model Insights

The hybrid CNN-LSTM architecture is effective because:
- **CNN layers**: Detect local character patterns (n-grams)
- **LSTM layers**: Capture long-range dependencies in the sequence
- **Character-level**: Robust to variations in domain structure

## 🔮 Future Enhancements

Potential improvements for this project:

1. **Multi-class Classification**: Classify specific DGA families
2. **Real-time Monitoring**: Integration with DNS traffic
3. **Feature Engineering**: Add TLD analysis, WHOIS data
4. **Ensemble Methods**: Combine multiple models
5. **API Deployment**: Create REST API for integration
6. **Model Explainability**: Add SHAP/LIME for interpretability

## 📝 License

This project is for educational and research purposes.

## 👤 Author

Developed for cybersecurity research and threat detection applications.

## 🙏 Acknowledgments

- Dataset sources for DGA and legitimate domains
- TensorFlow and Keras teams for the deep learning framework
- Streamlit team for the web framework

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the Jupyter notebook comments
3. Ensure all dependencies are correctly installed

---

**⚠️ Disclaimer**: This tool is for educational and research purposes. Always verify critical security decisions with multiple sources and security experts.
