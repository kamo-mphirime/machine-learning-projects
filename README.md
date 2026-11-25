# Machine Learning Projects

A collection of end-to-end machine learning workflows, feature engineering pipelines, and model benchmarks built with Python, scikit-learn, and XGBoost.

---

## Projects

### 1. Adult Census Income Classification (`adult_income_classification.ipynb`)
Binary classification pipeline predicting whether an individual's annual income exceeds **$50K** using the [UCI Adult Census Income Dataset](https://archive.ics.uci.edu/dataset/2/adult).

#### Data Preprocessing & Feature Engineering
- **Contextual Missing-Value Imputation:** Custom `conditional_mode_impute` function filling missing (`?`) values in `workclass`, `occupation`, and `native-country` using conditional group modes (`occupation` ↔ `workclass`, and `race` → `native-country`) with global mode fallbacks.
- **Pipeline Architecture (`ColumnTransformer`):**
  - **Numerical features (`age`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`):** Median imputation + `StandardScaler`.
  - **Ordinal features (`education`):** Ordered encoding from `Preschool` through `Doctorate`.
  - **Binary features (`sex`):** Binary ordinal encoding (`0`/`1`).
  - **Nominal categorical features (`workclass`, `marital-status`, `occupation`, `relationship`, `race`, `native-country`):** Most-frequent imputation + `OneHotEncoder`.

#### Model Benchmark Results (Test Set, $n = 9,525$)

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost** | **67.84%** | **71.75%** | 55.14% | 62.36% |
| **Logistic Regression** | 65.94% | 65.77% | **61.51%** | **63.57%** |
| **KNN ($k=50$)** | 65.69% | 69.05% | 52.51% | 59.65% |
| **SVM** | 65.48% | 66.44% | 57.66% | 61.74% |
| **Random Forest** | 62.99% | 63.14% | 56.18% | 59.46% |
| **Decision Tree** | 60.01% | 59.17% | 55.53% | 57.29% |

- **Key Takeaway:** **XGBoost** achieved the highest overall accuracy (**67.84%**) and precision (**71.75%**), while **Logistic Regression** delivered the strongest recall (**61.51%**) and F1-score (**63.57%**).

---

## Tech Stack
- **Data Manipulation:** `pandas`, `numpy`, `ucimlrepo`
- **Machine Learning:** `scikit-learn`, `xgboost`
- **Visualization:** `matplotlib`
