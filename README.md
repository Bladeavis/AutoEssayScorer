# Automatic Prediction of Language Proficiency (CEFR Levels) on German Data with Linguistic Features

This project explores the use of deep learning models for automatically classifying German text into proficiency levels as defined by the Common European Framework of Reference for Languages (CEFR). The study compares two machine learning models: a Feedforward Neural Network (FFNN) and a Long Short-Term Memory (LSTM) network, evaluating their performance across multiple datasets and linguistic features.

## Overview

- **Objective:** Classify German texts into CEFR levels (A1–C2) using linguistic features.
- **Applications:** Language learning platforms, educational assessments, and personalized learning tools.
- **Key Features:** Grammatical, lexical, readability, and syntactic features for text analysis.

---

## Dataset

The dataset used in this project is a combination of labeled essays sourced from the **DISKO**, **Falko**, and **Merlin** corpora, along with a custom dataset personally created by  [Elias](https://github.com/EliasAhlers).


| Dataset                     | A1         | A2         | B1         | B2         | C1         | C2         |
|-----------------------------|------------|------------|------------|------------|------------|------------|
| **Merlin**                   | 57 texts   | 306 texts  | 331 texts  | 293 texts  | 42 texts   | 4 texts    |
| **FalkoC2**                  | 0 texts    | 0 texts    | 0 texts    | 0 texts    | 0 texts    | 95 texts   |
| **FalkoMixed**               | 0 texts    | 0 texts    | 0 texts    | 87 texts   | 90 texts   | 70 texts   |
| **AhlersA1**                 | 122 texts  | 0 texts    | 0 texts    | 0 texts    | 0 texts    | 0 texts    |
| **DISKO**                    | 0 texts    | 0 texts    | 27 texts   | 216 texts  | 294 texts  | 58 texts   |
|                              |            |            |            |            |            |            |
| **My Data**                  |            |            |            |            |            |            |
| **DataCombined**             | 179 texts  | 306 texts  | 358 texts  | 596 texts  | 426 texts  | 227 texts  |
| **DataCleanUP**              | 179 texts  | 306 texts  | 358 texts  | 552 texts  | 426 texts  | 227 texts  |


Data is categorized into six CEFR levels `Cefr` (A1, A2, B1, B2, C1, C2) and further grouped into three broader categories `NCefr` (A, B, C).

### Data Processing

The final combined dataset, named **DataCombined**, initially consisted of 2,092 essays. To balance the data for machine learning purposes, 44 essays from the B2 level were removed due to an overrepresentation at this level. This preprocessing step reduced the dataset to 2,048 essays, resulting in the final dataset named **DataCleanUP**.

**Final Dataset Statistics:**
- Total essays: 2,048 (after preprocessing)
- Features: 21 linguistic and structural features


## Models and Methodology

### Feedforward Neural Network (FFNN)
- **Architecture:**
  - Two hidden layers with ReLU activation
  - Dropout regularization
  - Softmax output
- **Optimizer:** Adam (learning rate: 0.001)
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score

### Long Short-Term Memory (LSTM)
- **Architecture:**
  - LSTM layers with dropout
  - Fully connected layer
  - Softmax output
- **Optimizer:** Adam (learning rate: 0.001)

### Preprocessing Steps
1. **Text Features:** TF-IDF (5,000 features)
2. **Numerical Features:** Normalization with StandardScaler
3. **Data Splits:** 60% training, 20% validation, 20% testing

---

## Results

### Performance Metrics

| Task       | Model | Accuracy | Precision | Recall | F1-Score |
|------------|-------|----------|-----------|--------|----------|
| 3 Levels   | FFNN  | 82.4%    | 82.5%     | 82.4%  | 82.4%    |
| 3 Levels   | LSTM  | 83.4%    | 83.6%     | 83.4%  | 83.2%    |
| 6 Levels   | FFNN  | 71.2%    | 72.4%     | 71.2%  | 71.0%    |
| 6 Levels   | LSTM  | 69.7%    | 70.4%     | 69.7%  | 69.5%    |

---

## Key Insights
- **3-Level Task:** LSTM outperformed FFNN, particularly in distinguishing beginner (A-level) texts.
- **6-Level Task:** FFNN achieved better performance due to its ability to handle overlapping features.

---

## Challenges and Limitations

1. Difficulty in distinguishing neighboring levels (e.g., B1 vs. B2).
2. Misclassification due to overlapping linguistic features.
3. Lack of advanced architectures like transformers.

---

## Future Work

- Experiment with transformer-based models (e.g., BERT, XLM-RoBERTa).
- Explore additional features like discourse markers and pragmatic cues.
- Refine datasets to better capture learner-specific nuances.

---

## ## References

1. **Julia Hancke. (2013):** Automatic prediction of CEFR proficiency levels based on linguistic features of learner language. Master’s thesis, University of Tübingen.
2. **Edit Szügyi, Sören Etler, Andrew Beaton, and Manfred Stede. (2019):** Automated assessment of language proficiency on German data. *Proceedings of the 15th Conference on Natural Language Processing (KONVENS)*, pages 30–37. University of Potsdam.
3. **Jeanine Treffers-Daller, Phil Parslow, and Sara Williams. (2018):** Back to basics: How measures of lexical diversity can help discriminate between CEFR levels. *Applied Linguistics*, 39(3):302–327.
4. **Zarah Weiss and Detmar Meurers. (2018):** Modeling the readability of German targeting adults and children. *27th International Conference on Computational Linguistics (Coling 2018)*, pages 303–313.


---

## Contact

For questions or assistance, please contact [Nurhayat Altunok](mailto:nualt100@uni-duesseldorf.de).
