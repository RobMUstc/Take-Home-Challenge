# Take-Home-Challenge
ML model for ASAG

## Objectives

The objective of this task was to build, train and evaluate a machine learning (ML) model for automatic short anser grading (ASAG). The model aimed to achieve high grading accuracy and show robustness to linguistic challenges common to NLP tasks. 

Two datasets were provided (accessible from https://huggingface.co/datasets/Atomi/semeval_2013_task_7_beetle_5way): 

- A training data set containing over 10,000 examples of questions, reference answers and student answers triples graded by domain experts
- A test data set, split into unseen questions and unseen answers

In this task, I aimed to achieve these objectives:

- Select an appropriate ML model for the problem
- Follow ML best practices (e.g. appropriate data preparation, extracting meaningful features, proper model evaluation)
- Deliver a reasonable baseline solution given a four hour development time constraint
- Document and communicate my workflow clearly and concisely 


## Methodology

### ML Approaches 

Multiple approaches were considered for this task, including a lexical approach, a semantic approach and a syntactic approach. Deep learning approaches were considered (after a brief review of some literature), but were left for future exploration given the time constraints of the task. 

Ultimately, an ensemble approach was chosen, in which key features were extracted from the training data (specifically the reference and student answers). A Random Forest Classifier was trained using the chosen features. 





### Chosen Features

The mix of chosen features aimed to assess the relevance, completeness and domain-specificity of the student answers. After some experimentation, the following features were extracted to train the Random Forest Classifier:

- Cosine similarity between student and reference answers. This gives a semantic measure of similarity between both answers.
- Length difference between student and reference answers. This gives a measure of completeness of the student's answer. 
- Percentage of specific key words (from the question and reference answer) present in the student's answer. This gives a measure of the student's answer addressing key elements in the domain. 


## Workflow / Results

### Data Preprocessing

Various preprocessing steps were performed to clean the data for feature extraction and proper model evaluation. 

The text in the reference and student answers were cleaned using the following simple techniques:

- Removing whitespaces
- Removing punctuation
- Lowercasing

A special token <BULB_A> was created for the circumstance when the letter 'a' followed the word 'bulb' or 'bulbs'. Exploration of the data showed in the 'SwitchesBulbsParallel' module, 'a' often referred to a 'bulb' labelled 'a'. The token was created to differentiate between this use and the use of 'a' in common language. No other special tokens were created, although a deeper analysis of the data with more time could reveal the need for another. 

For the reference answers in the training dataset, those classed as 'MINIMAL' were replaced with the 'BEST' answer for the corresponding question. Given all reference answers in the test dataset were classed as 'BEST', we want to train the model using the 'BEST' for meaningful feature extraction and proper evaluation. 

Spelling typos in the student answers were also corrected, although the method used proved to be computationally costly. It was assumed the 'BEST' reference answers contained no typos.

The five grade classes were numerically encoded, which was useful for assessing correlation between possible features and the grade classes. 



- Stop words - do not remove
- Handling negations, stemming?
- Linguistic challenges not addressed: synonyms
- Ignore variation in question type/module



### Model Training 
- Dealing with data imbalance: SMOTE oversampling, balancing class weights
- Feature to train on: Cosine similarity, jacard similarity, dice coefficient
  
The correlation of these features was assessed against the target feature (grade of the student answer - the label_5way grade in the dataset)

The model was tested against a validation dataset (a subset of given training dataset, which was not involved in the training of the model). A grid search was performed to tune the hyperparameters.

### Key Results 



### Error Analysis 
- Pearson correlation test

## Concluding Remarks 
