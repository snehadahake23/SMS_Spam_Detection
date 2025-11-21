# 📱 SMS Spam Detection: ML vs BERT Showdown

<div align="center">

[![Python 3.6](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Machine Learning](https://img.shields.io/badge/ML-Classification-green.svg)]()
[![NLP](https://img.shields.io/badge/NLP-BERT-orange.svg)]()
[![SMOTE](https://img.shields.io/badge/Imbalanced-SMOTE-red.svg)]()

**A comprehensive comparison of traditional ML algorithms and BERT transformer for SMS spam detection**

[Features](#-key-features) • [Results](#-model-performance) • [Installation](#-installation) • [Usage](#-usage) • [Dataset](#-dataset)

</div>

---

## 🎯 Project Overview

This project presents a **dual-approach** spam detection system for SMS messages, comparing:

1. 🔧 **Traditional Machine Learning** - 6 algorithms with Bag-of-Words embeddings
2. 🧠 **BERT Transformer** - State-of-the-art deep learning with contextual understanding
3. ⚖️ **SMOTE Enhancement** - Addressing class imbalance in the dataset

**Result**: Both approaches achieve exceptional **98% accuracy**, demonstrating that well-tuned traditional ML can compete with modern transformers for text classification tasks!

## 📊 Model Performance

| Model | Embeddings | Accuracy | Rank |
|-------|-----------|----------|------|
| 🥇 **BERT** | BERT Tokenizer | **0.98** | #1 |
| 🥈 **LinearSVC** | Bag-of-Words | **0.98** | #1 |
| 🥉 **SGD** | Bag-of-Words | **0.98** | #1 |
| 🏅 **Random Forest** | Bag-of-Words | **0.98** | #1 |
| **Logistic Regression** | Bag-of-Words | 0.97 | #5 |
| **Gradient Boosting** | Bag-of-Words | 0.97 | #5 |
| **Naive Bayes** | Bag-of-Words | 0.95 | #7 |

> 💡 **Key Insight**: Four models achieved identical 98% accuracy, with BERT offering superior semantic understanding while traditional models provide faster inference!

## ✨ Key Features

<table>
<tr>
<td width="33%">

### 🤖 Multiple Algorithms
- Logistic Regression
- Naive Bayes
- LinearSVC
- Random Forest
- SGD Classifier
- Gradient Boosting

</td>
<td width="33%">

### 🧠 Advanced NLP
- BERT Transformers
- Tokenization
- Stopword Removal
- Porter Stemming
- TF-IDF Weighting

</td>
<td width="33%">

### ⚖️ Data Balancing
- SMOTE Oversampling
- Handles Imbalanced Data
- Improved Minority Class
- Better Generalization

</td>
</tr>
</table>

## 📂 Project Structure

```
📦 spam-detection/
│
├── 📊 Notebooks/
│   ├── Spam.ipynb                     # Part 1: ML algorithms analysis
│   └── Spam_bert.ipynb                # Part 2: BERT implementation
│
├── 🐍 Scripts/
│   ├── clean_data.py                  # Data cleaning & preprocessing
│   ├── spam_model.py                  # Standard ML pipeline
│   ├── spam_smote_model.py            # ML with SMOTE balancing
│   ├── spam_bert.py                   # BERT model training
│   └── predictions.py                 # Inference & prediction script
│
├── 💾 models/
│   └── spam_best_model.pkl            # Trained model artifacts
│
├── 📁 data/
│   ├── spam.csv                       # Raw SMS dataset
│   └── spam_clean.csv                 # Preprocessed data
│
├── 📄 requirements.txt                # Python dependencies
└── 📖 README.md
```

## 🚀 Installation

### Prerequisites
- Python 3.6 or higher
- pip package manager

### Quick Start

**1️⃣ Clone the Repository**
```bash
git clone https://github.com/yourusername/spam-detection.git
cd spam-detection
```

**2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

**3️⃣ Download NLTK Data**
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

**4️⃣ Verify Installation**
```bash
python clean_data.py
```

## 💻 Usage

### 📊 Jupyter Notebooks (Recommended for Learning)

**Approach 1: Traditional ML Analysis**
```bash
jupyter notebook Spam.ipynb
```
- Exploratory data analysis
- Feature engineering visualization
- Compare 6 ML algorithms
- Performance evaluation

**Approach 2: BERT Deep Learning**
```bash
jupyter notebook Spam_bert.ipynb
```
- BERT tokenizer implementation
- Fine-tune transformer model
- Compare with traditional methods

---

### ⚡ Command Line Scripts (Production Ready)

#### 🧹 Step 1: Clean and Preprocess Data
```bash
python clean_data.py
```
**What it does:**
- Loads raw spam.csv dataset
- Removes unnecessary columns
- Cleans text (punctuation, lowercase)
- Removes stopwords
- Applies Porter stemming
- Saves to `spam_clean.csv`

**Output:**
```
(5572, 2)
   Class                                               Text
0      0                           go jurong point crazi avail bugi...
1      0                                         ok lar joke wif oni...
```

---

#### 🤖 Step 2: Train Models

**Option A: Standard ML Pipeline**
```bash
python spam_model.py
```
Trains 6 classifiers with standard train-test split (80/20).

**Option B: SMOTE-Enhanced Pipeline (Recommended)**
```bash
python spam_smote_model.py
```
Uses SMOTE oversampling to handle class imbalance before training.

**Option C: BERT Transformer**
```bash
python spam_bert.py
```
Fine-tunes pretrained BERT model from Hugging Face.

**Expected Output:**
```
              Model     Score
0         LinearSVC  0.982063
1     SGDClassifier  0.981166
2  RandomForestClassifier  0.979372
3  LogisticRegression  0.974895
...
```

---

#### 🎯 Step 3: Make Predictions

```bash
python predictions.py
```

**Interactive Usage:**
```
Type your message:
> Congratulations! You've won a $1000 gift card. Click here to claim.

Your message is spam
```

```
Type your message:
> Hey, are we still meeting for lunch tomorrow?

Your message is not spam
```

## 🔬 Technical Details

### Data Preprocessing Pipeline

```python
Raw SMS Text
    ↓
Remove Punctuation (re.sub)
    ↓
Lowercase Conversion
    ↓
Tokenization (split by whitespace)
    ↓
Remove Stopwords (NLTK)
    ↓
Stemming (Porter Stemmer)
    ↓
Feature Vectors (CountVectorizer + TF-IDF)
    ↓
Model Training
```

### Approach 1: Traditional ML

**Pipeline Configuration:**
```python
Pipeline([
    ('vect', CountVectorizer(min_df=5, ngram_range=(1, 2))),
    ('tfidf', TfidfTransformer()),
    ('classifier', [YOUR_CLASSIFIER])
])
```

**Hyperparameters:**
- **CountVectorizer:**
  - `min_df=5`: Ignore terms appearing in <5 documents
  - `ngram_range=(1,2)`: Use unigrams and bigrams
- **Random Forest:** 50 estimators
- **Gradient Boosting:** 150 estimators, max_depth=6

### Approach 2: SMOTE-Enhanced ML

**Pipeline Configuration:**
```python
imbpipeline([
    ('vect', CountVectorizer(min_df=5, ngram_range=(1, 2))),
    ('tfidf', TfidfTransformer()),
    ('smote', SMOTE()),  # Synthetic oversampling
    ('classifier', [YOUR_CLASSIFIER])
])
```

**Why SMOTE?**
- Dataset has ~87% ham, ~13% spam (imbalanced)
- SMOTE generates synthetic minority samples
- Improves model generalization on spam class

### Approach 3: BERT Transformer

**Architecture:**
```python
BERT Base Model (Hugging Face)
    ↓
BERT Tokenizer (WordPiece)
    ↓
[CLS] token embedding (768-dim)
    ↓
Dense Layer (Binary Classification)
    ↓
Sigmoid Activation
```

**Advantages:**
- Contextual word embeddings
- Bidirectional understanding
- Pre-trained on massive corpus
- Transfer learning benefits

## 📈 Results & Analysis

### Performance Comparison

| Model | Training Time | Inference Speed | Memory | Best For |
|-------|--------------|-----------------|---------|----------|
| **LinearSVC** | ⚡ Fast | ⚡⚡⚡ Very Fast | 💾 Low | Production |
| **SGD** | ⚡⚡ Very Fast | ⚡⚡⚡ Very Fast | 💾 Low | Large datasets |
| **Random Forest** | ⚡ Moderate | ⚡⚡ Fast | 💾 Medium | Interpretability |
| **BERT** | 🐌 Slow | ⚡ Moderate | 💾💾 High | Complex text |

### Key Findings

1. ✅ **Traditional ML Effectiveness**: LinearSVC and SGD matched BERT's 98% accuracy
2. ✅ **Speed Advantage**: Traditional models are 10-100x faster for inference
3. ✅ **SMOTE Impact**: Improved recall on minority (spam) class by 3-5%
4. ✅ **BERT Strengths**: Better handles context, sarcasm, and nuanced language
5. ✅ **Production Recommendation**: LinearSVC for speed; BERT for accuracy on edge cases

### Confusion Matrix Analysis

**LinearSVC Performance:**
```
                Predicted
              Ham    Spam
Actual Ham    [947]   [12]
Actual Spam   [8]     [148]
```
- Precision (Spam): 92.5%
- Recall (Spam): 94.9%
- F1-Score: 93.7%

## 📚 Dataset

<div align="center">

**[SMS Spam Collection Dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset)**

</div>

| Attribute | Details |
|-----------|---------|
| **Source** | UCI Machine Learning Repository |
| **Total Messages** | 5,574 SMS |
| **Language** | English |
| **Classes** | Ham (Legitimate) / Spam |
| **Format** | CSV with labeled data |
| **Class Distribution** | Ham: 86.6% / Spam: 13.4% |

**Sample Messages:**

| Type | Example |
|------|---------|
| 📧 **Ham** | "Ok lar... Joking wif u oni..." |
| 📧 **Ham** | "What you doing?how are you?" |
| 🚫 **Spam** | "FREE for 1st week! No1 Nokia tone 4 ur mob..." |
| 🚫 **Spam** | "WINNER!! You have won a £1000 cash prize!" |

## 🛠️ Tech Stack

<div align="center">

| Category | Technologies |
|----------|-------------|
| **Language** | ![Python](https://img.shields.io/badge/Python-3.6+-3776AB?logo=python&logoColor=white) |
| **Data Processing** | Pandas 1.0.5 • NumPy 1.16.4 |
| **Machine Learning** | Scikit-learn 0.23.2 • Imbalanced-learn 0.5.0 |
| **NLP** | NLTK 3.4.5 • Transformers (Hugging Face) |
| **Deep Learning** | TensorFlow • Keras |
| **Visualization** | Matplotlib 3.1.1 • Seaborn 0.9.0 |
| **Model Persistence** | Joblib |

</div>

## 🎓 Learning Outcomes

By exploring this project, you'll master:

- ✅ Text preprocessing and feature engineering for NLP
- ✅ Multiple ML classification algorithms and comparison
- ✅ Handling imbalanced datasets with SMOTE
- ✅ Implementing and fine-tuning BERT transformers
- ✅ Building production-ready ML pipelines
- ✅ Model evaluation and selection strategies
- ✅ Deploying trained models for inference

## 💡 Use Cases & Applications

This spam detection system can be adapted for:

- 📱 **SMS Filtering**: Mobile carriers and messaging apps
- 📧 **Email Security**: Spam and phishing detection
- 💬 **Chat Moderation**: Social media and forums
- 🛡️ **Fraud Prevention**: Financial institutions
- 📞 **Call Center**: Automated message triage
- 🤖 **Chatbots**: Filter malicious inputs

## 🔮 Future Enhancements

- [ ] 🌐 REST API with Flask/FastAPI
- [ ] 🎨 Interactive web UI (Streamlit/Gradio)
- [ ] 📱 Mobile app integration (iOS/Android)
- [ ] 🌍 Multi-language support (Spanish, French, etc.)
- [ ] 🔄 Ensemble methods (ML + BERT hybrid)
- [ ] 📊 Real-time monitoring dashboard
- [ ] 🐳 Docker containerization
- [ ] ☁️ Cloud deployment (AWS SageMaker, GCP AI Platform)
- [ ] 🔄 Active learning pipeline
- [ ] 📈 A/B testing framework

## 🐛 Troubleshooting

### Common Issues

**Issue: NLTK data not found**
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

**Issue: File path errors on Windows**
```python
# Change backslashes to forward slashes
URL_DATA = 'data/spam.csv'  # ✅ Works on all platforms
```

**Issue: Memory error with BERT**
```python
# Reduce batch size or use a smaller model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
```

**Issue: Model file not found**
```bash
# Ensure models/ directory exists
mkdir models
# Train model first before running predictions.py
python spam_model.py
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. 🍴 **Fork** the repository
2. 🌿 **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. 💾 **Commit** your changes (`git commit -m 'Add AmazingFeature'`)
4. 📤 **Push** to the branch (`git push origin feature/AmazingFeature`)
5. 🔀 **Open** a Pull Request

### Areas for Contribution
- 🆕 Additional ML algorithms (XGBoost, LightGBM)
- 🎯 Hyperparameter optimization (GridSearch, Optuna)
- 📊 More visualization and EDA
- 🧪 Unit tests and CI/CD
- 📝 Documentation improvements
- 🌐 API development
- 🎨 UI/UX enhancements

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Your Name**

- 🐙 GitHub: [@yourusername](https://github.com/yourusername)
- 💼 LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)
- 📧 Email: your.email@example.com
- 🌐 Portfolio: [yourwebsite.com](https://yourwebsite.com)

## 🙏 Acknowledgments

- **UCI ML Repository** for the SMS Spam Collection dataset
- **Hugging Face** for BERT and Transformers library
- **NLTK Team** for natural language processing tools
- **Scikit-learn** community for ML algorithms
- **Imbalanced-learn** for SMOTE implementation

## 📚 References

- [SMS Spam Collection Dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [SMOTE: Synthetic Minority Over-sampling Technique](https://arxiv.org/abs/1106.1813)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)

## ⭐ Show Your Support

If this project helped you or you found it interesting:

- ⭐ **Star** this repository
- 🐦 **Share** on social media
- 💬 **Provide feedback** via Issues
- 🍴 **Fork** and build something cool!
- 📝 **Write** a blog post about it

---

<div align="center">

**Made with ❤️ and Python**

*Comparing traditional ML and modern transformers for real-world NLP*

![Visitor Count](https://visitor-badge.laobi.icu/badge?page_id=yourusername.spam-detection)

[⬆ Back to Top](#-sms-spam-detection-ml-vs-bert-showdown)

</div>
