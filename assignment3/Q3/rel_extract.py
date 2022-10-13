from itertools import combinations
import json
import pandas as pd
import numpy as np
import spacy
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from random import choices
import tqdm


# read in the data
train_data = json.load(open("sents_parsed_train.json", "r"))
test_data = json.load(open("sents_parsed_test.json", "r"))


# load spacy stopwords
nlp = spacy.load("en_core_web_sm")


def print_example(data, index):
    """Prints a single example from the dataset. Provided only
    as a way of showing how to access various fields in the
    training and testing data.

    Args:
        data (list(dict)): A list of dictionaries containing the examples 
        index (int): The index of the example to print out.
    """
    # NOTE: You may Delete this function if you wish, it is only provided as 
    #   an example of how to access the data.
    
    # print the sentence (as a list of tokens)
    print("Tokens:")
    print(data[index]["tokens"])

    # print the entities (position in the sentence and type of entity)
    print("Entities:")
    for entity in data[index]["entities"]:
        print("%d %d %s" % (entity["start"], entity["end"], entity["label"]))
    
    # print the relation in the sentence if this is the training data
    if "relation" in data[index]:
        print("Relation:")
        relation = data[index]["relation"]
        print("%d:%s %s %d:%s" % (relation["a_start"], relation["a"],
            relation["relation"], relation["b_start"], relation["b"]))
    else:
        print("Test examples do not have ground truth relations.")

def write_output_file(relations, filename = "q3.csv"):
    """The list of relations into a csv file for the evaluation script

    Args:
        relations (list(tuple(str, str))): a list of the relations to write
            the first element of the tuple is the PERSON, the second is the
            GeoPolitical Entity
        filename (str, optional): Where to write the output file. Defaults to "q3.csv".
    """
    out = []
    for person, gpe in relations:
        out.append({"PERSON": person, "GPE": gpe})
    df = pd.DataFrame(out)
    df.to_csv(filename, index=False)

# print a single training example
print("Training example:")
print_example(train_data, 1)

print("---------------")
print("Testing example:")
# print a single testing example
# the testing example does not have a ground
# truth relation
print_example(test_data, 2)

#TODO: build a training/validation/testing pipeline for relation extraction
#       then write the list of relations extracted from the *test set* to "q3.csv"
#       using the write_output_file function.

def train_preprocess(data_dict):
    feature_sents = []
    labels = []
    nationality = '/people/person/nationality'

    for sent in tqdm.tqdm(data_dict):
        # extract relation indices
        relation_a, relation_b = sent["relation"]["a"], sent["relation"]["b"]
        for entity in sent["entities"]:
            start = entity["start"]
            end = entity["end"]
            entity_str = sent["tokens"][start: end]
            if relation_a == entity_str:
                x = entity["start"], entity["end"]
            if relation_b == entity_str:
                y = entity["start"], entity["end"]

        # preprocessing labels
        if sent["relation"]["relation"] == nationality:
            labels.append(1)
        else:
            labels.append(0)

        # preprocessing features
        # NOTE: inspired from online resources, see report
        start = max(min(x[0], y[0]) - 3, 0)
        end = min(max(x[1], y[1]) + 3, len(sent["tokens"]))
        entity_indices = list(range(x[0], x[1])) + list(range(y[0], y[1]))
        sequence = []
        for index in range(start, end):
            if index not in entity_indices and not sent["isstop"][index] and sent["isalpha"][index]:
                sequence.append(sent["tokens"][index].lower())
        # processing the sequence
        processed = nlp(" ".join(sequence))
        lemmatized = []
        for token in processed:
            lemmatized.append(token.lemma_)
        feature_sents.append(" ".join(lemmatized))
    
    return feature_sents, labels


def test_preprocess(data_dict):
    pass


train_x, train_y = train_preprocess(train_data)
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.15)

cv = CountVectorizer()
mat = cv.fit_transform(train_x)
lr = LogisticRegression(class_weight="balanced")
lr.fit(mat, train_y)


# validation
pred_val = lr.predict(cv.transform(val_x))
precision = precision_score(val_y, pred_val)
acu = accuracy_score(val_y, pred_val)
recall = recall_score(val_y, pred_val, average="macro")
score = f1_score(val_y, pred_val)
print("precision %f "%precision)
print("recall %f "%recall)
print("auccuacy %f "%acu)
print("score %f "%score)


# Example only: write out some relations to the output file
# normally you would use the list of relations output by your model
# as an example we have hard coded some relations from the training set to write to the output file
# TODO: remove this and write out the relations you extracted (obviously don't hard code them)
relations = [
    ('Hokusai', 'Japan'), 
    ('Hans Christian Andersen', 'Denmark')
    ]
write_output_file(relations)
