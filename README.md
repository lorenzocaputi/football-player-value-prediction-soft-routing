# A Soft-Routing Pipeline to Predict the Value of Football Players

A machine learning approach for predicting professional football player market values using a novel soft-routing ensemble methodology that addresses distributional outliers and multimodal feature distributions.

## Project Overview

This project tackles the challenge of predicting football player market values from a high-dimensional dataset containing demographic, contractual, and performance features. The key innovation is a **soft-routing pipeline** that explicitly handles the structural differences between typical players and high-value outliers, which behave as distributional anomalies in the feature space.

### Problem Statement

Predicting football player market values presents unique challenges:
- **Highly skewed target distribution** with extreme outliers among top-valued players
- **Multimodal feature distributions** reflecting different player roles (goalkeepers vs. outfield players)
- **Nonlinear relationships** between features and market value
- **High variance** in model performance across cross-validation folds due to uneven distribution of outliers

### Solution: Soft-Routing Pipeline

Our approach uses a three-component ensemble:
1. **Binary Classifier**: Identifies players in the top 5% of market value
2. **Base Regressor**: Trained on the complete dataset
3. **Top Regressor**: Specialized model trained exclusively on high-value players

Final predictions are computed as a weighted combination of the two regressors, with weights derived from the classifier's probability estimates.

## Key Findings

### Data Insights
- Market values follow a highly right-skewed distribution (mean: €2.8M, std: €7.5M)
- Strong multicollinearity exists between technical attributes, particularly within player roles
- Top-valued players are not just statistical outliers but **distributional anomalies** in feature space
- Prediction errors are heavily concentrated in the top quantile of player values

### Model Performance
- **Best Configuration**: Random Forest (classifier + base regressor) + CatBoost (top regressor)
- **Validation RMSE**: €609,265 (5-fold cross-validation)
- **Improvement**: Outperforms standard tree-based methods while reducing variance across folds
- **Error Reduction**: Removing top 20% of players reduces RMSE by 90% to ~€70k, demonstrating outlier impact

### Methodological Contributions
- Error analysis reveals that high-value players dominate loss function despite being <5% of data
- Autoencoder visualization confirms that top players occupy low-density regions in feature space
- Soft-routing approach provides both improved accuracy and increased stability across validation folds

## Repository Structure

```
├── eda.ipynb                     # Exploratory Data Analysis
├── preprocessing.ipynb           # Data preprocessing pipeline
├── model_selection.ipynb         # Comprehensive model evaluation
├── autoencoders.ipynb           # Dimensionality reduction and visualization
├── error_analysis.ipynb         # Analysis of prediction errors by value quantiles
├── generate_predictions.ipynb   # Final model training and test predictions
├── REPORT.docx.pdf              # Detailed technical report
├── submission.csv               # Final test set predictions
├── results_seed_100_q0.95_2025-05-16_12-05-45.csv  # Model selection results
└── requirements.txt             # Python dependencies
```

## Methodology

### Data Preprocessing
1. **Missing Value Strategy**: Zero imputation with structural missing flags for informative nulls
2. **Feature Engineering**: 
   - Position groups (Goalkeeper, Defender, Midfielder, Attacker)
   - BMI, contract duration, years at club
   - Work rate ordinal encoding
3. **Categorical Encoding**: One-hot for low cardinality, 5-fold cross-validated target encoding for high cardinality
4. **Nonlinear Transformations**: Log transforms for skewed features, squared age term
5. **Standardization**: Z-score normalization for model flexibility

### Model Architecture

**Pipeline Components:**
- **Classifier**: Random Forest (300 estimators, max_features=0.5)
- **Base Regressor**: Random Forest (300 estimators, max_features=0.5)
- **Top Regressor**: CatBoost (1500 iterations, depth=3, learning_rate=0.1)

**Prediction Formula:**
```
final_prediction = (1 - p_top) × base_prediction + p_top × top_prediction
```
where `p_top` is the classifier's estimated probability of being a top player.

### Feature Selection
Key features include:
- **Contract Terms**: `wage_eur`, `release_clause_eur`, `log_wage_eur`, `log_release_clause_eur`
- **Player Attributes**: `overall`, `potential`, `international_reputation`, `age`
- **Technical Skills**: `pace`, `shooting`, `passing`, `dribbling`, `defending`, `physic`
- **Positional**: One-hot encoded position groups
- **Physical**: `height_cm`, `weight_kg`
- **Club Information**: `club_name_te` (target encoded)

## Results

### Model Comparison (5-fold Cross-Validation RMSE)
| Model Type | RMSE (€) | Std Dev (€) |
|------------|----------|-------------|
| Linear Models | ~1,355,000 | High |
| K-NN (k=4) | 1,560,000 | High |
| Gradient Boosting | 687,281 | 114,170 |
| **Soft-Routing Pipeline** | **609,265** | **Reduced** |

### Error Analysis
- **Leave-one-bin-out analysis** shows errors concentrated in top value quintile
- **Normalized RMSE** remains high for top players even after accounting for scale
- **Fold stability** improved through explicit handling of distributional outliers

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model
```python
# 1. Run preprocessing
python preprocessing.ipynb

# 2. Perform model selection
python model_selection.ipynb

# 3. Generate final predictions
python generate_predictions.ipynb
```

### Key Parameters
- **Quantile Threshold**: 0.95 (top 5% players)
- **Random State**: 100 (for reproducibility)
- **CV Folds**: 5 (stratified by top player flag)

## Future Work

1. **Meta-Learning Ensemble**: Replace heuristic averaging with learnable meta-model for dynamic weight computation
2. **External Data Integration**: Incorporate injury records, recent performance metrics, and transfer history
3. **Temporal Modeling**: Account for time-varying factors affecting player valuations
4. **Position-Specific Models**: Develop specialized regressors for each position group

## Authors

Lorenzo Caputi, Andrea Procopio, Filippo Strub, Andrea Pettenon, Paride Lauretti

## License

This project is for academic research purposes. Please cite if you use this methodology in your work.

---

*This project demonstrates the effectiveness of structure-aware modeling in heterogeneous regression settings, with particular relevance for domains characterized by extreme outliers and multimodal feature distributions.*
