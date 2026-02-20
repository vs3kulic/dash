# Transaction Categorization ML Project - TODO

Following the **Basic Data Analysis Workflow**

---

## 1. Question/Task + Data ✅
- [x] **Question**: Can we automatically categorize transactions based on subject text?
- [x] **Data Collected**: 254 labeled transactions (202 train, 52 validation)
- [x] **Target**: 13 transaction categories
- [x] **Input Features**: Subject text, amount, dates

---

## 2. Preprocessing ✅
- [x] Load raw CSV files (German format: semicolon separator, comma decimals)
- [x] Handle different date formats (DD.MM.YYYY vs DD.MM.YY)
- [x] Clean and standardize text (subject field; minimal, will refine during feature engineering if needed)
- [x] Extract counterparty information
- [x] Map to categories using alias_mapping.json
- [x] Split into train/validation sets
- [x] Save processed data

### Prior Knowledge Applied:
- [x] Domain expertise for category mappings
- [x] Alias dictionary for counterparty normalization
- [x] Business rules for category assignment

---

## 3. Choose Features (In Progress) 🔄
- [x] **Exploratory Data Analysis**
  - [x] Class distribution analysis → Severe imbalance found
  - [x] Location measures computed (mean, median, outliers)
  - [x] Text pattern analysis incl. common keywords per category
  - [ ] Correlation Analysis

- [ ] **Feature Engineering**
  - [ ] Text vectorization (TF-IDF)
    - [ ] Choose n-gram range (unigrams, bigrams, trigrams)
    - [ ] Set max_features parameter
    - [ ] Handle German text specifics
  - [ ] Numerical features
    - [ ] Transaction amount (normalized)
    - [ ] Amount sign (income vs expense)
    - [ ] Temporal features (day, month, day of week)
  - [ ] Feature scaling/normalization
  - [ ] Create feature matrix (X) and labels (y)

---

## 4. Choose Model Class 📋
- [ ] **Baseline Model**
  - [ ] Evaluate current rule-based mapping accuracy
  - [ ] Establish performance floor

- [ ] **Primary Candidate: Logistic Regression**
  - Reasons: Good for multiclass text classification, interpretable, handles high-dimensional sparse features (TF-IDF)
  - Plan: Use `class_weight='balanced'` for imbalance

- [ ] **Alternative Models to Compare**
  - [ ] Naive Bayes (good baseline for text)
  - [ ] Random Forest (handles imbalance well)
  - [ ] Linear SVM (alternative to LogReg)
  - [ ] XGBoost (if tree-based works better)

---

## 5. Train Model 🎯
- [ ] **Initial Training**
  - [ ] Implement Logistic Regression with class_weight='balanced'
  - [ ] Fit on training data (202 samples)
  - [ ] Save trained model

- [ ] **Configuration**
  - [ ] Set regularization parameter (C)
  - [ ] Choose solver (lbfgs, saga, etc.)
  - [ ] Set max_iter for convergence

---

## 6. Evaluate Model 📊

### New Insights from Evaluation:
- [ ] **Performance Metrics**
  - [ ] Overall accuracy
  - [ ] Per-class precision, recall, F1-score
  - [ ] Macro vs weighted averages
  - [ ] Confusion matrix visualization

- [ ] **Validation Set Evaluation**
  - [ ] Test on 52 validation samples
  - [ ] Identify misclassified transactions
  - [ ] Analyze which categories perform poorly

- [ ] **Error Analysis**
  - [ ] Review misclassifications
  - [ ] Identify patterns in errors
  - [ ] Check if small classes are underperforming

### Iterate:
- [ ] If performance insufficient → Return to **Choose Features** (add more features, adjust TF-IDF)
- [ ] If class imbalance causes issues → Return to **Choose Features** (try SMOTE, adjust class weights)
- [ ] If model underfits → Return to **Choose Model Class** (try more complex models)
- [ ] If model overfits → Return to **Train Model** (increase regularization)

---

## 7. Final Model + Answer ✨
- [ ] **Model Selection**
  - [ ] Choose best performing model
  - [ ] Document final hyperparameters
  - [ ] Record performance metrics

- [ ] **Production Integration**
  - [ ] Save final model (pickle/joblib)
  - [ ] Create prediction pipeline
  - [ ] Integrate with data_processing.py
  - [ ] Handle edge cases (unknown categories, low confidence)

- [ ] **Answer the Question**
  - [ ] Can we auto-categorize? (Yes/No + confidence level)
  - [ ] Which categories work well vs poorly?
  - [ ] Recommended threshold for auto-categorization
  - [ ] When to flag for manual review

- [ ] **Documentation**
  - [ ] Model performance summary
  - [ ] Feature importance (if available)
  - [ ] Usage examples
  - [ ] Limitations and assumptions

---

## 8. Monitoring & Maintenance 🔄
- [ ] Track model performance on new data
- [ ] Retrain periodically with new labeled transactions
- [ ] Update category mappings based on new patterns
- [ ] Version control for models
- [ ] Log predictions and confidence scores

---

## Technical Setup Checklist
- [ ] Add scikit-learn to requirements.txt
- [ ] Create src/feature_engineering.py
- [ ] Create src/model.py
- [ ] Create models/ directory for saved models
- [ ] Set up experiment logging
