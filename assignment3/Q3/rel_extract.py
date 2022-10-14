from itertools import product
import json
import pandas as pd
import numpy as np
import spacy
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import tqdm


# read in the data
# train_data = json.load(open("sents_parsed_train.json", "r"))
# test_data = json.load(open("sents_parsed_test.json", "r"))
train_data = json.load(open("train.json", "r"))
test_data = json.load(open("val.json", "r"))

with open("country_names.txt", "r") as f:
    countries = f.read()
    countries = countries.split("\n")[:-1]

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


# function for preprocess feature from sentences
def sentence_processing(sent, start, end):
    sequence = []
    for index in range(start, end):
        # only if the token is not stopword, and only contain alphabetic characters
        if not sent["isstop"][index] and sent["isalpha"][index]:
            sequence.append(sent["tokens"][index].lower())
    spacy_sequence = nlp(" ".join(sequence))
    rtn = []
    for token in spacy_sequence:
        rtn.append(token.lemma_)
    return rtn


# training set preprocessing
def train_preprocess(data_dict):
    feature_sents = []
    labels = []
    distance = []
    nationality = '/people/person/nationality'

    for sent in tqdm.tqdm(data_dict):
        # extract relation indices
        relation_a, relation_b = sent["relation"]["a"], sent["relation"]["b"]

        for entity in sent["entities"]:
            # range from first proper noun to last proper noun
            entity_tokens = sent["tokens"][entity["start"]: entity["end"]]
            if entity_tokens == relation_a:
                entity_idx_a = entity["start"], entity["end"]
            if entity_tokens == relation_b:
                entity_idx_b = entity["start"], entity["end"]

        # NOTE: inspired from online resources, see report
        # we allow the relation word to extend the relation word with length 3
        # Features: words in between entities, entities themselves, words in particular positions
        start = max(min(entity_idx_a[0], entity_idx_b[0]) - 3, 0)
        end = min(max(entity_idx_a[1], entity_idx_b[1]) + 3, len(sent["tokens"]))

        # preprocessing features
        processed_sequence = sentence_processing(sent, start, end)
        feature_sents.append(" ".join(processed_sequence))
        distance.append(end - start)

        # preprocessing labels
        if sent["relation"]["relation"] == nationality:
            labels.append(1)
        else:
            labels.append(0)
    
    cv = CountVectorizer()
    train_features = cv.fit_transform(feature_sents).toarray()
    embedded_features = np.hstack((train_features, np.array(distance).reshape(-1, 1)))

    return embedded_features, cv, labels


def test_preprocess(data_dict, cv):
    indices_original_sent = []
    feature_sents = []
    entities = []
    distance = []

    for i, sent in enumerate(tqdm.tqdm(data_dict)):
        # we only extract GPE and PERSON as they are the only chance for nationality
        gpes_indices = []
        persons_indices = []
        for entity in sent["entities"]:
            x = sent["tokens"][entity["start"]: entity["end"]]
            if entity["label"] == "GPE" and " ".join(sent["tokens"][entity["start"]: entity["end"]]) in countries:
                # check if the entity is in country list
                gpes_indices.append((entity["start"], entity["end"]))
            elif entity["label"] == "PERSON":
                persons_indices.append((entity["start"], entity["end"]))

        entity_indices_combinations = product(persons_indices, gpes_indices)
        
        # extract features for each entity pairs in the given list
        for a_idx, b_idx in entity_indices_combinations:
            # allow a extension of window 3
            start = max(min(a_idx[0], b_idx[0]) - 3, 0)
            end = min(max(a_idx[1], b_idx[1]) + 3, len(sent["tokens"]))

            # preprocessing features
            processed_sequence = sentence_processing(sent, start, end)

            # append the corresponding sentence index
            indices_original_sent.append(i)
            # append features to the list
            feature_sents.append(" ".join(processed_sequence))
            # append the entity words used
            entities.append((sent["tokens"][a_idx[0]: a_idx[1]], sent["tokens"][b_idx[0]: b_idx[1]]))
            # add distance (# of words) between entities
            distance.append(end - start)

    features = cv.transform(feature_sents).toarray()
    embedded_features = np.hstack((features, np.array(distance).reshape(-1, 1)))

    return indices_original_sent, embedded_features, entities


def test_postprocess(predictions, entities):
    relations = []
    for i, prediction in enumerate(predictions):
        if prediction == 1:
            relations.append((" ".join(entities[i][0]), " ".join(entities[i][1])))
    write_output_file(relations=relations)


train_x, cv, train_y = train_preprocess(train_data)
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2)

lr = LogisticRegression(max_iter=500, class_weight="balanced")
lr.fit(train_x, train_y)

# training data
pred_val = lr.predict(train_x)
precision = precision_score(train_y, pred_val)
acu = accuracy_score(train_y, pred_val)
recall = recall_score(train_y, pred_val, average="macro")
score = f1_score(train_y, pred_val)
print("precision %f "%precision)
print("recall %f "%recall)
print("auccuacy %f "%acu)
print("score %f "%score)
print()

# validation
pred_val = lr.predict(val_x)
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
test_sent_indices, test_features, entities = test_preprocess(test_data, cv)
test_predictions = lr.predict(test_features)
test_postprocess(test_predictions, entities)
