# Loan Default Prediction with Random Forest  
# Random Forest 기반 채무불이행 예측 모델

## 1. Project Overview / 프로젝트 개요

This project builds a machine learning model to predict loan default risk from applicants' financial and loan-related information. The main evaluation metric is F1-score, because the target variable is imbalanced: approximately 75.4% non-default and 24.6% default.

본 프로젝트는 대출 신청자의 금융 및 대출 관련 정보를 바탕으로 채무불이행 위험 여부를 예측하는 머신러닝 모델을 구축한 과제입니다. 타깃 변수는 약 75.4%가 정상 상환, 약 24.6%가 채무불이행으로 구성되어 있어 클래스 불균형이 존재하며, 이에 따라 주요 평가지표로 F1-score를 사용했습니다.

The overall strategy was:

전체 전략은 다음과 같습니다.

1. Perform basic 5-Fold Cross Validation screening with Logistic Regression, Decision Tree, and Random Forest.
2. Select promising tree-based models, Decision Tree and Random Forest.
3. Tune hyperparameters using Optuna.
4. Select the best model based on F1-score.
5. Retrain the final model on the full training data and generate test predictions.

---

## 2. Repository Structure / 저장소 구조

```text
ml-loan-default-prediction-rf/
├── notebooks/
│   └── ml-loan-default-prediction.ipynb
├── data/
│   └── README.md
├── output/
│   └── sample_id_status.csv
├── models/
│   └── .gitkeep
├── README.md
├── requirements.txt
└── .gitignore
```

`train.csv`, `test.csv`, and `.pkl` model files are excluded from Git tracking.  
`train.csv`, `test.csv`, `.pkl` 모델 파일은 Git 추적 대상에서 제외했습니다.

---

## 3. Dataset / 데이터셋

The training data contains approximately 119,000 rows and 28 columns, while the test data contains approximately 29,700 rows and 27 columns. The target variable is Status, where 0 means non-default and 1 means default.

학습 데이터는 약 119,000행 × 28열, 테스트 데이터는 약 29,700행 × 27열로 구성되어 있습니다. 예측 대상은 Status이며, 0은 정상 상환, 1은 채무불이행을 의미합니다.

The final feature set contains:

최종 사용한 feature는 다음과 같습니다.

- 4 numerical features: loan_amount, term, income, Credit_Score
- 21 categorical features
- Excluded features: Sample_ID, year

Sample_ID was excluded because it is only an identifier, and year was excluded because it had the same value across all rows.

Sample_ID는 단순 식별자이므로 제외했고, year는 전체 값이 동일하여 모델 학습에 유의미한 정보를 제공하지 않는다고 판단해 제외했습니다.

---

## 4. EDA Summary / EDA 요약

### 4.1 Missing Values / 결측치 처리

The main missing values were found in income, loan_limit, and approv_in_adv. Numerical missing values were imputed with the median, and categorical missing values were imputed with the most frequent value.

주요 결측치는 income, loan_limit, approv_in_adv 등에서 확인되었습니다. 수치형 변수는 중앙값으로, 범주형 변수는 최빈값으로 결측치를 대체했습니다.

- income: median imputation
- term: median imputation
- categorical variables: most frequent imputation

### 4.2 Class Imbalance / 클래스 불균형

The target distribution was approximately:

타깃 분포는 대략 다음과 같았습니다.

- Status = 0: 75.4%
- Status = 1: 24.6%

Because accuracy can be misleading under class imbalance, F1-score was used as the main metric. For applicable models, class_weight='balanced' was used to reduce the effect of imbalance.

클래스 불균형 환경에서는 accuracy만으로 모델을 평가하면 다수 클래스 위주의 예측이 과대평가될 수 있습니다. 따라서 precision과 recall의 균형을 함께 반영하는 F1-score를 주요 평가지표로 사용했고, 가능한 모델에는 class_weight='balanced'를 적용했습니다.

---

## 5. Modeling Approach / 모델링 접근법

The preprocessing pipeline was built with ColumnTransformer and Pipeline to prevent data leakage during cross validation.

전처리 과정에서는 ColumnTransformer와 Pipeline을 사용하여 교차검증 과정에서 데이터 누수가 발생하지 않도록 구성했습니다.

- Numerical features: median imputation + standard scaling
- Categorical features: most frequent imputation + one-hot encoding
- Model and preprocessing were combined into a single pipeline

The following models were compared using Stratified 5-Fold Cross Validation:

다음 모델들을 Stratified 5-Fold Cross Validation으로 비교했습니다.

| Model | CV F1-score |
|---|---:|
| Logistic Regression | 0.6364 ± 0.0059 |
| Decision Tree | 0.6550 ± 0.0041 |
| Random Forest | 0.6541 ± 0.0085 |

Logistic Regression showed the lowest F1-score and was excluded from the final candidates. Decision Tree and Random Forest showed better performance, so both were selected for Optuna hyperparameter tuning.

Logistic Regression은 가장 낮은 F1-score를 보여 최종 후보에서 제외했습니다. Decision Tree와 Random Forest는 상대적으로 높은 성능을 보여 Optuna 하이퍼파라미터 튜닝 대상으로 선정했습니다.

---

## 6. Hyperparameter Tuning / 하이퍼파라미터 튜닝

Optuna was used instead of GridSearchCV because GridSearchCV searches all parameter combinations, which can be expensive for a large dataset and Random Forest. Optuna searches promising parameter regions more efficiently based on previous trials.

GridSearchCV는 지정한 모든 파라미터 조합을 전부 탐색하기 때문에, 데이터 규모가 크고 Random Forest처럼 계산 비용이 큰 모델에서는 시간이 많이 소요될 수 있습니다. 따라서 이전 trial 결과를 바탕으로 유망한 파라미터 영역을 효율적으로 탐색하는 Optuna를 사용했습니다.

### Decision Tree Tuning Result / Decision Tree 튜닝 결과

- Best CV F1-score: 0.6557
- Main parameters:
  - max_depth=10
  - min_samples_split=23
  - min_samples_leaf=10
  - max_features=None
  - criterion='gini'

### Random Forest Tuning Result / Random Forest 튜닝 결과

- Best CV F1-score: 0.6796
- Main parameters:
  - n_estimators=300
  - max_depth=30
  - min_samples_split=11
  - min_samples_leaf=8
  - max_features=0.5

Random Forest achieved the best cross-validation F1-score and was selected as the final model.

Random Forest는 가장 높은 교차검증 F1-score를 기록하여 최종 모델로 선정했습니다.

---

## 7. Final Validation Result / 최종 검증 성능

The final tuned Random Forest model was trained on X_train and evaluated on a hold-out validation set X_val, which was not used during Optuna tuning.

최종 튜닝된 Random Forest 모델은 X_train으로 학습한 뒤, Optuna 튜닝 과정에서 사용하지 않은 hold-out validation set인 X_val에서 최종 성능을 확인했습니다.

- Validation F1-score: 0.6868
- Default class precision: 0.79
- Default class recall: 0.61
- Default class F1-score: 0.69

The model achieved relatively high precision for default prediction, but recall was lower, meaning that some actual default cases were still missed.

모델은 default로 예측한 샘플에 대해서는 비교적 높은 precision을 보였지만, recall은 상대적으로 낮아 실제 default 샘플 중 일부는 탐지하지 못했습니다.

---

## 8. How to Run / 실행 방법

bash git clone https://github.com/<your-username>/ml-loan-default-prediction-rf.git cd ml-loan-default-prediction-rf  python -m venv .venv source .venv/bin/activate  pip install -r requirements.txt jupyter notebook notebooks/ml-loan-default-prediction.ipynb 

Place train.csv and test.csv under the data/ directory before running the notebook.

노트북 실행 전 train.csv와 test.csv를 data/ 폴더에 넣어야 합니다.

---

## 9. Tech Stack / 사용 기술

- Python
- pandas, NumPy
- matplotlib, seaborn
- scikit-learn
- Optuna
- Jupyter Notebook

---

## 10. Reference / 참고 자료

- Minsik Oh, 10_2 NB_RF_XGBoost_Practice, Machine Learning 1, 2026.
- ChatGPT 5.5 was used as an auxiliary tool for checking validation strategy, EDA visualization code, fit/transform concepts, model screening syntax, and Optuna tuning setup.

---

## 11. Notes / 참고 사항

This repository is intended for educational and portfolio purposes. Course-provided raw data and serialized model files are excluded from Git tracking.

본 저장소는 학습 및 포트폴리오 목적의 프로젝트입니다. 수업 제공 원본 데이터와 직렬화된 모델 파일은 Git 추적 대상에서 제외하였습니다. 
