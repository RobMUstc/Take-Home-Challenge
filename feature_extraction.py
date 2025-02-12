from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import load_preprocess_data


# Function to compute Cosine Similarity between two texts
def cos_similarity(reference_answer, student_answer, vectoriser):
    tdidf_matrix = vectoriser.fit_transform([reference_answer, student_answer])
    cosine_sim = cosine_similarity(tdidf_matrix[0], tdidf_matrix[1])
    return cosine_sim[0][0]


# Function to compute Jaccard Similarity between two texts
def jacc_similarity(reference_answer, student_answer):
    ref_set = set(reference_answer.split())
    stud_set = set(student_answer.split())
    intersection = len(ref_set.intersection(stud_set))
    union = len(ref_set.union(stud_set))
    return intersection / union  # Jaccard similarity


# Function to compute part of speech tagging information
def pos_similarity(reference_answer, student_answer):
    ref_pos = pos_tag(word_tokenize(reference_answer))
    stud_pos = pos_tag(word_tokenize(student_answer))

    # Calculate POS tag overlap
    ref_tags = set(tag for word, tag in ref_pos)
    stud_tags = set(tag for word, tag in stud_pos)
    intersection = len(ref_tags.intersection(stud_tags))
    union = len(ref_tags.union(stud_tags))
    return intersection / union


# Function to compute text length (in words)
def length_diff(reference_answer, student_answer):
    ref_tokens = word_tokenize(reference_answer)
    stud_tokens = word_tokenize(student_answer)
    ref_length = len(ref_tokens)
    stud_length = len(stud_tokens)
    return abs(ref_length - stud_length)


# Add keyword overlap feature - remove stopwords first to focus on keywords
def remove_stopwords(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in ENGLISH_STOP_WORDS]
    return tokens


def keyword_ol(reference_answer, student_answer):
    ref_tokens = remove_stopwords(reference_answer)
    stud_tokens = remove_stopwords(student_answer)
    common_keywords = set(ref_tokens).intersection(set(stud_tokens))
    return len(common_keywords)


# Feature to handle contradictions between student and reference answers
negation_words = ['not', 'no', 'never', 'none', 'nothing', 'nowhere', 'neither', 'don\'t', 'doesn\'t' 'can\'t', 'won\'t']


def detect_negation(tokens):
    return any(neg_word in tokens for neg_word in negation_words)


def contra_score(reference_answer, student_answer):
    ref_tokens = nltk.word_tokenize(reference_answer)
    stud_tokens = nltk.word_tokenize(student_answer)

    ref_has_negation = detect_negation(ref_tokens)
    student_has_negation = detect_negation(stud_tokens)

    if ref_has_negation != student_has_negation:
        return True
    else:
        return False


# Load and preprocess the data
train_df, test_df, vectoriser = load_preprocess_data()


def select_features(vec):
    # Cosine similarity
    train_df['cosine_similarity'] = train_df.apply(
        lambda row: cos_similarity(row['reference_answer'], row['student_answer'], vec), axis=1)
    test_df['cosine_similarity'] = test_df.apply(
        lambda row: cos_similarity(row['reference_answer'], row['student_answer'], vec), axis=1)

    # Jaccard similarity
    train_df['jaccard_similarity'] = train_df.apply(
        lambda row: jacc_similarity(row['reference_answer'], row['student_answer']), axis=1)
    test_df['jaccard_similarity'] = test_df.apply(
        lambda row: jacc_similarity(row['reference_answer'], row['student_answer']), axis=1)

    # POS similarity
    train_df['pos_similarity'] = train_df.apply(
        lambda row: pos_similarity(row['reference_answer'], row['student_answer']), axis=1)
    test_df['pos_similarity'] = test_df.apply(
        lambda row: pos_similarity(row['reference_answer'], row['student_answer']), axis=1)

    # Length difference
    train_df['length_difference'] = train_df.apply(
        lambda row: length_diff(row['reference_answer'], row['student_answer']), axis=1)
    test_df['length_difference'] = test_df.apply(
        lambda row: length_diff(row['reference_answer'], row['student_answer']), axis=1)

    # Keyword overlap
    train_df['keyword_overlap'] = train_df.apply(
        lambda row: keyword_ol(row['reference_answer'], row['student_answer']), axis=1)
    test_df['keyword_overlap'] = test_df.apply(
        lambda row: keyword_ol(row['reference_answer'], row['student_answer']), axis=1)

    # Contradiction score
    train_df['contradiction_score'] = train_df.apply(
        lambda row: contra_score(row['reference_answer'], row['student_answer']), axis=1)
    test_df['contradiction_score'] = test_df.apply(
        lambda row: contra_score(row['reference_answer'], row['student_answer']), axis=1)

    fts = ['cosine_similarity', 'pos_similarity', 'length_difference', 'keyword_overlap']

    return train_df, test_df, fts


def feature_correlation(df, fts):
    X = df[fts]
    correlation_matrix = X.corr()

    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Matrix")
    plt.show()


# Assess feature correlation
# train_df, test_df, features = select_features(vectoriser)
# feature_correlation(train_df, features)
