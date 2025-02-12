import pandas as pd
import re
from textblob import TextBlob
import string
from sklearn.feature_extraction.text import TfidfVectorizer


# TEXT CLEANING ANSWERS


# Removing whitespaces
def remove_whitespaces(text):
    text = re.sub(r'\s+', ' ', text).strip()  # Replace multiple spaces with one, and strip leading/trailing spaces
    return text


# Lowercase text
def lowercase(text):
    return text.lower()


# Remove punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))


# Treat the character 'a' differently if it follows the word 'bulb' or 'bulbs'
def bulb_a_special(text):
    # Define a regex pattern to find 'bulb' or 'bulbs' followed by 'A' or 'a'
    pattern = r'\b(bulb|bulbs)\s+([Aa])\b'  # 'bulb' or 'bulbs' followed by 'A' or 'a'

    # Replace 'A' or 'a' after 'bulb' or 'bulbs' with a special token "<BULB_A>"
    special_token_text = re.sub(pattern, r'\1 <BULB_A>', text)

    return special_token_text


def clean_answers(text):
    text = remove_whitespaces(text)

    text = lowercase(text)

    text = remove_punctuation(text)

    text = bulb_a_special(text)

    return text


#  Fixing typos
def fix_typos(text):
    blob = TextBlob(text)
    return str(blob.correct())


def load_preprocess_data():

    # Import data from hugging face
    splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
    train_df = pd.read_parquet("hf://datasets/Atomi/semeval_2013_task_7_beetle_5way/" + splits["train"])
    test_df = pd.read_parquet("hf://datasets/Atomi/semeval_2013_task_7_beetle_5way/" + splits["test"])

    # Clean the text in the answers
    train_df['student_answer'] = train_df['student_answer'].apply(clean_answers)
    train_df['reference_answer'] = train_df['reference_answer'].apply(clean_answers)
    test_df['student_answer'] = test_df['student_answer'].apply(clean_answers)
    test_df['reference_answer'] = test_df['reference_answer'].apply(clean_answers)

    # Fix typos in the student answers - computationally costly, omitted by default
    #train_df['student_answer'] = train_df['student_answer'].apply(fix_typos)
    #test_df['student_answer'] = test_df['student_answer'].apply(fix_typos)

    # Replace 'MINIMAL' answers with their 'BEST' answer for the same question (only relevant to training dataset)
    answer_map = train_df[train_df['reference_answer_quality'] == 'BEST'].set_index('question')['reference_answer'].to_dict()
    train_df.loc[train_df['reference_answer_quality'] == 'MINIMAL', 'reference_answer'] \
        = train_df.loc[train_df['reference_answer_quality'] == 'MINIMAL', 'question'].map(answer_map)
    train_df['reference_answer_quality'].replace('MINIMAL', 'BEST')

    #  Vectorise the answers and add as colum to existing dataframe
    vectoriser = TfidfVectorizer()
    combined_train_answers = train_df['reference_answer'].tolist() + train_df['student_answer'].tolist()
    vectoriser.fit(combined_train_answers)

    train_ref_tfidf = vectoriser.transform(train_df['reference_answer'])
    train_stud_tfidf = vectoriser.transform(train_df['student_answer'])

    test_ref_tfidf = vectoriser.transform(test_df['reference_answer'])
    test_stud_tfidf = vectoriser.transform(test_df['student_answer'])

    train_ref_tfidf_df = pd.DataFrame(train_ref_tfidf.toarray(),
                                      columns=[f'ref_feature_{i}' for i in range(train_ref_tfidf.shape[1])])
    train_stud_tfidf_df = pd.DataFrame(train_stud_tfidf.toarray(),
                                       columns=[f'stud_feature_{i}' for i in range(train_stud_tfidf.shape[1])])

    test_ref_tfidf_df = pd.DataFrame(test_ref_tfidf.toarray(),
                                     columns=[f'ref_feature_{i}' for i in range(test_ref_tfidf.shape[1])])
    test_stud_tfidf_df = pd.DataFrame(test_stud_tfidf.toarray(),
                                      columns=[f'stud_feature_{i}' for i in range(test_stud_tfidf.shape[1])])

    train_df = pd.concat([train_df, train_ref_tfidf_df, train_stud_tfidf_df], axis=1)
    test_df = pd.concat([test_df, test_ref_tfidf_df, test_stud_tfidf_df], axis=1)

    # Numerically encode the grade classes
    grade_mapping = {'correct': 5, 'partially_correct_incomplete': 4, 'irrelevant': 3, 'contradictory': 2, 'non_domain': 1}
    train_df['grade_encoded'] = train_df['label_5way'].map(grade_mapping)

    # Drop any duplicates
    train_df.drop_duplicates(inplace=True)
    test_df.drop_duplicates(inplace=True)

    return train_df, test_df, vectoriser
