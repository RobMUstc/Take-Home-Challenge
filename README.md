# Take-Home-Challenge
ML model for ASAG

## Objectives

The objective of this task was to build, train and evaluate a machine learning (ML) model for automatic short anser grading (ASAG). The model aimed to achieve high grading accuracy and show robustness to linguistic challenges common to NLP tasks. 

Two datasets were provided (accessible from https://huggingface.co/datasets/Atomi/semeval_2013_task_7_beetle_5way): 

- A training data set containing over 10,000 examples of questions, reference answers and student answers triples graded by domain experts
- A test data set, split into unseen questions and unseen answers

In this task, I aimed to achieve these objectives:

- Select an appropriate ML for the problem
- Follow ML best practices (e.g. appropriate data preparation, extracting meaningful features, proper model evaluation)
- Deliver a reasonable baseline solution given a four hour development time constraint
- Document and communicate my workflow clearly and concisely 


## Methodology

### ML Approaches 

Multiple approaches were considered for this task, including purely lexical, semantic approach and syntactic approaches. Ultimately, an ensemble approach was chosen, in which key features were extracted from the training data (specifically the reference and student answers). The correlation of these features was assessed against the target feature (grade of the student answer - the label_5way grade in the dataset). A Random Forest Classifier was trained using the chosen features. 

The model was tested against a validation dataset (a subset of given training dataset, which was not involved in the training of the model). A grid search was performed to tune the hyperparameters.



### Chosen Features
- Presence of specific key words
- Similarity to reference answers
- Length difference

## Workflow / Results

### Data Preprocessing
- Stop words - do not remove



### Model Training 
- Dealing with data imbalance: SMOTE oversampling, balancing class weights
- Feature to train on: Cosine similarity, jacard similarity, dice coefficient


### Key Results 

### Error Analysis 
- Pearson correlation test

## Concluding Remarks 
