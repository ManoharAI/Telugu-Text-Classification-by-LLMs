>**Title:**              Fine-tuning LLMs for Domain-Specific Text Classification
 
>**Language:**    Telugu


# Telugu Text Classification Project Report

## 1. Project Overview

This project focuses on fine-tuning a BERT-based Language Model for domain-specific text classification in Telugu. The goal is to classify Telugu news articles into predefined categories, leveraging the power of pre-trained transformers for improved performance in a low-resource language context.

---

## 2. Setup and Environment

The project utilized common Python libraries for data manipulation, machine learning, and deep learning, including `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, and significantly, the `transformers` and `datasets` libraries from Hugging Face.

- **Python Version**: 3.x
- **Key Libraries**: `transformers` (v4.57.1), `pytorch` (v2.8.0+cu126)
- **Device**: CUDA (GPU available) for accelerated training.

---

## 3. Data Acquisition and Exploration

The dataset used is the "Telugu EENADU News Articles" from Kaggle, consisting of news articles categorized into several classes.

- **Dataset Source**: Kaggle (`shubhamjain27/telugu-news-articles`)
- **Training Samples**: 16,421
- **Test Samples**: 4,106
- **Original Columns**: `title`, `text`, `category`, `t`
- **Target Column**: `category` (renamed to `label` for model training)
- **Number of Classes**: 5
- **Class Mapping**:
    
    ```json
    {"eenadu_sports": 0, "eenadu_national": 1, "eenadu_business": 2, "eenadu_crime": 3, "eenadu_cinema": 4}
    ```
    
- **Missing Values**: No missing values found in either training or test sets.
- **Text Length Statistics (Training Data)**:
    - Mean: 1145 characters
    - Median: 808 characters
    - Max: 14003 characters

**Visualizations of Data Distribution**: 

![image.png](attachment:9cc329ee-f54b-4ee4-a776-fe0462711521:image.png)

---

## 4. Model Details

- **Base Model**: `l3cube-pune/telugu-bert`
- **Architecture**: BERT (Bidirectional Encoder Representations from Transformers)
- **Task**: Multi-class Sequence Classification
- **Number of Labels (Output Classes)**: 5
- **Model Parameters**: 237,560,069
- **Tokenizer**: AutoTokenizer from `l3cube-pune/telugu-bert`
    - Vocabulary Size: 197,285

---

## 5. Training Configuration and Process

The model was fine-tuned using the Hugging Face `Trainer` API with the following key configurations:

- **Evaluation Strategy**: Per epoch
- **Saving Strategy**: Per epoch, best model saved based on 'f1' score.
- **Number of Training Epochs**: 3
- **Training Batch Size (per device)**: 16
- **Evaluation Batch Size (per device)**: 32
- **Learning Rate**: 2e-5
- **Max Sequence Length**: 512 tokens
- **GPU Used**: Tesla T4

![image.png](attachment:8c81b350-d023-4da1-b212-ef3ba172f597:image.png)

---

## 6. Evaluation Results

The model's performance was evaluated on the held-out test set. The metrics used were Accuracy, Precision, Recall, and F1-Score (weighted average).

**Test Set Performance**:

- **Accuracy**: 96.27%
- **Precision**: 96.29%
- **Recall**: 96.27%
- **F1-Score**: 96.27%
- **Loss**: 0.1458

**Classification Report**: 

```
 precision    recall  f1-score   support

eenadu_sports       0.99      0.98      0.99      1154
eenadu_national     0.94      0.93      0.93       909
eenadu_business     0.95      0.98      0.96       791
eenadu_crime        0.93      0.94      0.94       650
eenadu_cinema       0.99      0.99      0.99       602

    accuracy                           0.96      4106
   macro avg       0.96      0.96      0.96      4106
weighted avg       0.96      0.96      0.96      4106
```

**Confusion Matrix**: 

![image.png](attachment:41480303-acd8-46a7-94e7-0e1b6d71e3eb:image.png)

---

## 7. Sample Predictions

The model was tested with sample texts from the test set and custom user input. The predictions demonstrated high confidence and accuracy for correctly classified samples.

---

## 8. Conclusion

The fine-tuned Telugu BERT model achieved excellent performance with a test accuracy and F1-score exceeding 96%. This demonstrates the effectiveness of transfer learning for text classification in Telugu, particularly for news article categorization.

**Key Findings**:

- The model shows strong performance across all categories, with particularly high precision and recall for `eenadu_sports` and `eenadu_cinema` classes (F1-scores of 0.99).
- The `eenadu_crime` and `eenadu_national` categories showed slightly lower but still robust performance (F1-scores of 0.94 and 0.93 respectively).
- The balanced performance across precision and recall metrics indicates the model is neither over-predicting nor under-predicting any particular class.
- Fine-tuning a pre-trained Telugu BERT model significantly reduces training time while achieving high accuracy, making it suitable for practical deployment.
