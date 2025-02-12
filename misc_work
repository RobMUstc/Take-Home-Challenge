from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV



# Performing grid search for tuning hyperparameters of Random Forest Classifier
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train_resampled, y_train_resampled)

print("Best parameters found: ", grid_search.best_params_)

best_model = grid_search.best_estimator_



# Mapping question type and module for correlation testing with target grade classes
# Question module
module_mapping = {'FaultFinding': 1, 'SwitchesBulbsParallel': 2, 'SwitchesBulbsSeries': 3}
train_df['module_encoded'] = train_df['question_module'].map(module_mapping)

# Question type
question_mapping = {'PREDICT': 1, 'EVALUATE': 2, 'QUESTION': 3}
train_df['question_encoded'] = train_df['question_stype'].map(question_mapping)

X = train_df[features]
y = train_df[target]

mi = mutual_info_classif(X, y)



# Perform Recursive Feature Elimination (RFE) with Random Forest Classifier
selector = RFE(rf, n_features_to_select=3)  # Select the top 3 features, for example
selector = selector.fit(X_train, y_train)

# Get the ranking of features (1 is the most important, larger numbers are less important)
feature_ranking = selector.ranking_

# Get the selected features
selected_features = X_train.columns[selector.support_]

print("Feature Ranking:", feature_ranking)
print("Selected Features:", selected_features)
