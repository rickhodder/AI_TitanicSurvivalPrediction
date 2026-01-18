# Product Requirements Document (PRD)
## Titanic Survival Prediction

### Document Information
- **Project Name:** Titanic Survival Prediction
- **Version:** 1.0
- **Date:** January 18, 2026
- **Author:** Rick Hodder
- **Status:** Draft

---

## 1. Executive Summary
### 1.1 Project Overview
To get hands-on experience with Machine Learning using a simple Python project. This project focuses on learning ML fundamentals through the classic Titanic survival prediction problem using a notebook-driven development approach.

### 1.2 Problem Statement
Creating a model that, given a passenger's information for the Titanic, predicts whether they would survive the disaster.

### 1.3 Success Metrics

**Model Performance Metrics:**
- **Accuracy:** ≥80% on test set
- **Precision:** ≥78% (minimize false positives)
- **Recall:** ≥75% (maximize true positives)
- **F1 Score:** ≥0.77 (balanced precision and recall)
- **AUC-ROC:** ≥0.85 (model discrimination ability)

**Learning & Project Metrics:**
- Complete all 5 exploratory notebooks
- Implement ≥3 different ML algorithms
- Create ≥5 meaningful visualizations
- Document findings and insights
- Ability to explain model results and feature importance

---

## 2. Objectives and Goals
### 2.1 Primary Objectives
- [ ] Understand the end-to-end ML workflow from data exploration to model evaluation
- [ ] Learn data preprocessing and feature engineering techniques
- [ ] Compare multiple ML algorithms and understand their strengths/weaknesses
- [ ] Develop skills in model evaluation and interpretation

### 2.2 Key Results
- [ ] Successfully preprocess Titanic dataset with proper handling of missing values
- [ ] Engineer at least 3 new meaningful features
- [ ] Train and compare at least 3 different models (e.g., Logistic Regression, Random Forest, Gradient Boosting)
- [ ] Achieve target accuracy metrics on holdout test set
- [ ] Create comprehensive documentation and visualizations

---

## 3. User Stories and Use Cases
### 3.1 Target Users
**Primary User:** Data science learner (yourself) gaining practical ML experience

### 3.2 User Stories
1. **As a learner, I want to explore the Titanic dataset visually so that I understand the relationships between features and survival**
2. **As a learner, I want to experiment with different preprocessing techniques so that I understand their impact on model performance**
3. **As a learner, I want to compare multiple ML algorithms so that I learn when to use each approach**
4. **As a learner, I want to interpret my model's predictions so that I understand what drives survival predictions**

### 3.3 Use Cases
**Use Case 1: Exploratory Data Analysis**
- Load Titanic dataset
- Visualize distributions and relationships
- Identify missing data patterns
- Discover insights about survival factors

**Use Case 2: Model Development**
- Preprocess data with different strategies
- Engineer features based on domain knowledge
- Train multiple models with cross-validation
- Evaluate and compare model performance

---

## 4. Functional Requirements
### 4.1 Data Requirements
- **Dataset:** Titanic passenger data (Kaggle competition dataset)
- **Features:** 
  - Passenger demographics (Age, Sex, Class)
  - Ticket information (Fare, Ticket number)
  - Cabin details
  - Family relationships (SibSp, Parch)
  - Embarkation port
- **Target Variable:** Survived (0 = Did not survive, 1 = Survived)
- **Dataset Size:** ~891 training records, ~418 test records

### 4.2 Model Requirements
- **Model Type:** Binary Classification
- **Expected Accuracy:** ≥80% on test set
- **Training Requirements:** Standard laptop/desktop, < 5 minutes training time
- **Interpretability:** High (should be able to explain predictions)

### 4.3 Feature Engineering
- [ ] Handle missing values (Age, Cabin, Embarked, Fare)
- [ ] Encode categorical variables (Sex, Embarked, Pclass)
- [ ] Create family size feature (SibSp + Parch + 1)
- [ ] Create "is_alone" feature
- [ ] Extract title from Name (Mr., Mrs., Miss., Master, etc.)
- [ ] Create age bins/groups
- [ ] Create fare bins/groups
- [ ] Feature scaling for numerical variables

### 4.4 Model Training & Evaluation
- [ ] 80/20 train/test split with stratification
- [ ] 5-fold cross-validation
- [ ] Evaluate with multiple metrics: accuracy, precision, recall, F1, AUC-ROC
- [ ] Compare at least 3 algorithms:
  - Logistic Regression (baseline)
  - Random Forest
  - Gradient Boosting (XGBoost or LightGBM)
- [ ] Create confusion matrix and classification report
- [ ] Analyze feature importance

---

## 5. Non-Functional Requirements
### 5.1 Performance
- Model training time: < 5 minutes per model
- Notebook execution time: < 10 minutes per notebook
- Prediction time: Near-instantaneous for batch predictions

### 5.2 Reproducibility
- Use fixed random seeds (random_state=42)
- Document all preprocessing steps
- Save processed datasets for consistency
- Version control all notebooks and code

### 5.3 Maintainability
- Clear markdown documentation in notebooks
- Descriptive variable names
- Comments for complex operations
- Modular code that can be refactored into functions

---

## 6. Technical Architecture
### 6.1 Technology Stack
- **Programming Language:** Python 3.9+
- **ML Framework:** scikit-learn (primary)
- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Development Environment:** Jupyter Notebooks (primary), VS Code
- **Version Control:** Git/GitHub
- **Optional Enhancement:** MLflow for experiment tracking (future)

### 6.2 Project Structure
```
AI_TitanicSurvivalPrediction/
├── data/
│   ├── raw/                    # Original Kaggle data
│   └── processed/              # Cleaned data
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_model_evaluation.ipynb
├── src/                        # (Optional) Refactored code
│   ├── data/
│   ├── features/
│   └── models/
├── models/                     # Saved model artifacts
├── reports/
│   └── figures/               # Generated visualizations
├── docs/
│   └── PRD.md
├── config/
│   └── config.yaml            # (Optional) Parameters
├── requirements.txt
└── README.md
```

### 6.3 Data Pipeline (Notebook-Based)
1. **Notebook 01:** Data loading, exploration, and visualization
2. **Notebook 02:** Data cleaning, missing value handling, outlier detection
3. **Notebook 03:** Feature engineering and transformation experiments
4. **Notebook 04:** Model training, hyperparameter tuning, cross-validation
5. **Notebook 05:** Model evaluation, comparison, and interpretation
6. **(Optional Phase 2)** Refactor working code into reusable Python modules

---

## 7. Data Requirements
### 7.1 Data Sources
- **Source:** [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- **Training data:** `data/raw/train.csv` (891 records)
- **Test data:** `data/raw/test.csv` (418 records)
- **Data dictionary:** Available on Kaggle

### 7.2 Data Quality
- **Missing Data:**
  - Age: ~20% missing
  - Cabin: ~77% missing
  - Embarked: ~0.2% missing
  - Fare: 1 missing value in test set
- **Strategy:** Imputation for Age/Fare, drop or encode Cabin, mode for Embarked
- **Outliers:** Detect in Fare and Age, decide on handling strategy

### 7.3 Data Privacy & Security
- Public dataset, no privacy concerns
- No personally identifiable information (historical data)

---

## 8. Model Development Plan
### 8.1 Baseline Model
- **Model:** Logistic Regression with minimal preprocessing
- **Purpose:** Establish baseline performance (~75-78% accuracy expected)
- **Features:** Basic features without engineering

### 8.2 Advanced Models
**Models to Evaluate:**
1. **Logistic Regression** (with feature engineering)
2. **Random Forest Classifier**
3. **Gradient Boosting** (XGBoost or LightGBM)
4. **Optional:** Support Vector Machine, Naive Bayes

**Hyperparameter Tuning:**
- Use GridSearchCV or RandomizedSearchCV
- Focus on key parameters (n_estimators, max_depth, learning_rate)
- Tune based on cross-validation performance

### 8.3 Experiment Tracking
**Track in Notebooks:**
- Model parameters and configuration
- Training/validation/test accuracy
- Cross-validation scores
- Feature importance
- Confusion matrices

**Optional (Future):** Integrate MLflow for systematic experiment tracking

---

## 9. Testing & Validation
### 9.1 Model Testing
- **Data Validation:** Check for data leakage, correct train/test split
- **Performance Testing:** Verify metrics calculation correctness
- **Sanity Checks:** Predictions make logical sense

### 9.2 Validation Strategy
- **Cross-Validation:** 5-fold stratified CV on training set
- **Holdout Test Set:** 20% of training data held out
- **Final Evaluation:** Kaggle test set (if submitting)

---

## 10. Deployment (Future Phase - Optional)
### 10.1 Deployment Strategy
Not in scope for initial learning phase. Potential future enhancements:
- Create simple Flask/Streamlit web app
- Accept passenger details and return survival prediction
- Deploy to Heroku or similar platform

### 10.2 Monitoring
Not applicable for learning project

### 10.3 Model Updates
Not applicable for learning project

---

## 11. Timeline and Milestones
| Phase | Task | Duration | Target Date |
|-------|------|----------|-------------|
| Phase 1 | Data exploration & visualization | 1-2 days | Week 1 |
| Phase 2 | Data preprocessing & cleaning | 1-2 days | Week 1 |
| Phase 3 | Feature engineering experiments | 2-3 days | Week 1-2 |
| Phase 4 | Model training & hyperparameter tuning | 2-3 days | Week 2 |
| Phase 5 | Model evaluation & comparison | 1-2 days | Week 2 |
| Phase 6 | Documentation & final report | 1 day | Week 2-3 |
| **Total** | | **~2 weeks** | |

**Note:** Timeline is flexible based on learning pace and depth of exploration

---

## 12. Risks and Mitigation
| Risk | Impact | Probability | Mitigation Strategy |
|------|--------|-------------|---------------------|
| Poor model performance | Medium | Low | Try multiple algorithms, extensive feature engineering, hyperparameter tuning |
| Overfitting | High | Medium | Use cross-validation, regularization, ensemble methods |
| Data quality issues | Medium | Low | Thorough EDA, robust preprocessing pipeline |
| Scope creep (over-engineering) | Low | Medium | Stay focused on learning objectives, keep initial implementation simple |
| Missing domain knowledge | Low | Medium | Research Titanic history, consult Kaggle notebooks for insights |

---

## 13. Dependencies
- [ ] Python 3.9+ installed
- [ ] Jupyter Notebook or JupyterLab
- [ ] Required libraries: pandas, numpy, scikit-learn, matplotlib, seaborn
- [ ] Kaggle account for dataset download
- [ ] Git/GitHub for version control

---

## 14. Open Questions
- [ ] Should we implement deep learning models or stick to classical ML?
- [ ] How much time to spend on feature engineering vs model tuning?
- [ ] Should we create a simple web app for demonstration?
- [ ] Will we submit to Kaggle competition for public leaderboard score?

---

## 15. Appendix
### 15.1 References
- [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
- Recommended Kaggle notebooks for inspiration (research after initial attempt)

### 15.2 Glossary
- **EDA:** Exploratory Data Analysis
- **Cross-Validation:** Technique to assess model generalization
- **Feature Engineering:** Creating new features from existing data
- **Overfitting:** Model performs well on training data but poorly on unseen data
- **AUC-ROC:** Area Under the Receiver Operating Characteristic curve
- **Stratification:** Maintaining class distribution in train/test splits

---

## Document History
| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | Jan 18, 2026 | Rick Hodder | Initial draft with learning-focused approach |
