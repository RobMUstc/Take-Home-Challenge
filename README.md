# Take-Home Challenge

Alongside this README, the repo contains the following scripts:

- _model_run_ : Code to execute the workflow. Imports preprocessed data and chosen features, trains and evaluates the model.
- _feature_extraction_ : Extracts features from the preprocessed data.
- _data_preprocessing_ : Preprocesses the data. 
- _misc_work_ : Not to be run. Contains miscellaneous experimentation code for different stages of the  workflow. 

## Objectives

The objective of this task was to build, train and evaluate a machine learning (ML) model for automatic short anser grading (ASAG).  

Two datasets were provided (accessible from https://huggingface.co/datasets/Atomi/semeval_2013_task_7_beetle_5way): 

- A training data set containing over 10,000 examples of questions, reference answers and student answers triples graded by domain experts.
- A test data set containing unseen questions and unseen answers.

In this task, I aimed to achieve these objectives:

- Select an appropriate ML model for the problem.
- Follow ML best practices (e.g. appropriate data preparation, extracting meaningful features, proper model evaluation).
- Deliver a reasonable baseline solution given a four hour development time constraint.
- Document and communicate my workflow clearly and concisely.


## Methodology

### ML Approaches 

The target of this task was to accurately predict the grade classes of student answers in the test dataset. As such, this was considered a classification problem.

Multiple approaches were considered to assess the class of student answers, including a lexical approach, a semantic approach and a syntactic approach. Deep learning approaches were considered (after a brief review of some literature), but were left for future exploration given the time constraints of the task. 

Ultimately, an ensemble approach was chosen, in which key features were extracted from the training data, focusing on a comparison between the reference and student answers. A Random Forest Classifier (RFC) was trained with the chosen features. The classifier was used to predict the grade class.



### Chosen Features

The mix of chosen features aimed to assess the relevance, completeness and domain-specificity of the student answers. After some experimentation, the following features were extracted to train the RFC:

- **Cosine similarity** between student and reference answers. This gives a semantic measure of similarity between the answers.
- **Part of speech (POS) tags overlap** between student and reference answers. This gives a measure of structure similarity between the answers.
- **Length difference** between student and reference answers. This gives a measure of completeness of the student's answer. 
- **Keyword overlap** (from the reference answer) present in the student's answer. This gives a measure of the student's answer addressing key elements in the domain. 


## Workflow / Results

### Data Preprocessing

Various preprocessing steps were performed to clean the data for feature extraction and proper model evaluation. The corresponding source code can be found in the _data_preprocessing_ script.

The text in the reference and student answers were cleaned using the following simple techniques:

- Removing whitespaces
- Removing punctuation
- Lowercasing

A special token <BULB_A> was created for the circumstance when the letter 'a' followed the word 'bulb' or 'bulbs'. Exploration of the data showed in the 'SwitchesBulbsParallel' module, 'a' often referred to a 'bulb' labelled 'a'. The token was created to differentiate between this use and the use of 'a' in common language. No other special tokens were created, although a deeper analysis of the data with more time could reveal the need for another. 

For the reference answers in the training dataset, those classed as 'MINIMAL' were replaced with the 'BEST' answer for the corresponding question. Given all reference answers in the test dataset were classed as 'BEST', we want to train the model using the 'BEST' for meaningful feature extraction and proper evaluation. 

Spelling typos in the student answers were also corrected, although the method used proved to be computationally costly. It was assumed the 'BEST' reference answers contained no typos.

The five grade classes were numerically encoded, which was useful for assessing correlation between possible features and the grade classes. 

Other preprocessing techniques were considered, but not committed to the presented solution:

- Stop word removal. Given the brevity of many answers, this was not applied in the preprocessing stage. It did have targeted use in extracting the shared keyword feature. 
- Stemming / lemmatisation. Judged as overkill for this task. 
- Synonym matching. Judged as a useful addition and a future improvement to overcome a linguistic challenge common to NLP tasks. Not applied due to time constraints.


### Dealing with Data Imbalance

Various techniques were considered to deal with the imbalance in the grade classes in the training dataset. For proper evaluation, we need to train the model on a dataset balanced in the target feature (grade class). 

Oversampling using the Synthetic Minority Oversampling Technique (SMOTE) was tested. It was not applied due to problems with overfitting the training dataset. 

Undersampling by taking a subset of the training data was also tested. It was not applied as the remaining subset was judged too small for model training. 

Ultimately, balanced class weights were calculated and used as a parameter in the RFC.

Analysis was also done to assess the imbalance in grades across different question types and modules. These differences were found to be negligible. 



### Model Training 

Before fitting the RFC, a validation dataset was split from the training dataset (20% validation, 80% training). This allowed an assessment of whether the model overfitting the training dataset. 

Some possible features were extracted, including cosine similarity, Jaccard similarity, POS tags overlap, length difference, keyword overlap, contradiction score (see details and source code in the _feature_extraction_ script). Feature correlations with the grade classes were computed to get an initial sense of importance. Feature correlations with each other were also computed to flag possible overlap / overfitting. It was found that cosine similarity and Jaccard similarity were strongly correlated (no surprise...) and so only one would be chosen as a feature.  

The RFC was fitted with different combinations of these features. Feature importance scores were analysed and recursive feature elimination was performed. The resulting choice of features to train the Random Forest Classifier was - **Cosine similarity**, **Part of speech (POS) tags overlap**, **Length difference**, **Keyword overlap**.  

The model was tested against a validation dataset. A grid search was performed to tune the hyperparameters. The default hyperparameters were adopted in the model presented (find
source code for the grid search in the _misc work_ script).

The tuned model was then used against the unseen test dataset.


### Key Results 

Key results are captured in the classification reports for the test sets.

Classification report for Unseen Answers:


|  | Precision | Recall | F1 score |
| ------------- | ------------- | ------------- | ------------- |
| contradictory  | 0.33  | 0.28  | 0.30  |
| correct | 0.59  | 0.74  | 0.66  |
| irrelevant | 0.00  | 0.00  | 0.00  |
| non_domain | 0.61  | 0.59  | 0.60  |
| partially_correct_incomplete  | 0.40 | 0.34  | 0.37  |
|   |  |   |   |
| accuracy  |  |   | 0.48  |
| weighted avg  | 0.46 | 0.48  | 0.46  |


Classification report for Unseen Questions:


|  | Precision | Recall | F1 score |
| ------------- | ------------- | ------------- | ------------- |
| contradictory  | 0.40  | 0.34  | 0.36  |
| correct | 0.61  | 0.55  | 0.58  |
| irrelevant | 0.23  | 0.12  | 0.15  |
| non_domain | 0.42  | 0.61  | 0.50  |
| partially_correct_incomplete  | 0.30 | 0.41  | 0.35  |
|   |  |   |   |
| accuracy  |  |   | 0.45 |
| weighted avg  | 0.46 | 0.45 | 0.45 |


### Error Analysis 
- Pearson correlation test

## Concluding Remarks 
