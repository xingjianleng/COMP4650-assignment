from itertools import combinations
import json
import pandas as pd
import numpy as np
import spacy
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer


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

def preprocess(data_dict, train=True):
    feature_sents = []
    labels = []
    nationality = '/people/person/nationality'

    for sent in data_dict:
        entities = []
        for entity in sent['entities']:
            entities.append((entity["start"], entity["end"]))
        entity_index_combination = list(combinations(entities, r=2))

        # preprocessing labels
        if train:
            # generate negative samples, inspired from assignment instruction
            all_entity_combination = [
                (sent["tokens"][x[0]: x[1]], sent["tokens"][y[0]: y[1]]) for x, y in entity_index_combination
            ]
            if sent["relation"]["relation"] == nationality:
                relation_a, relation_b = sent["relation"]["a"], sent["relation"]["b"]
                relations_combination = [(relation_a, relation_b), (relation_b, relation_a)]
                labels.extend(list(map(int, map(lambda x: x in relations_combination, all_entity_combination))))
            else:
                labels.extend([0] * len(all_entity_combination))

        # preprocessing features
        for x, y in entity_index_combination:
            # NOTE: inspired from online resources, see report
            start = max(min(x[0], y[0]) - 3, 0)
            end = min(max(x[1], y[1]) + 3, len(sent["tokens"]))
            entity_indicies = list(range(x[0], x[1])) + list(range(y[0], y[1]))
            sequence = []
            for index in range(start, end):
                if index not in entity_indicies and not sent["isstop"][index] and sent["isalpha"][index]:
                    sequence.append(sent["tokens"][index].lower())
            # processing the sequence
            processed = nlp(" ".join(sequence))
            lemmatized = []
            for token in processed:
                lemmatized.append(token.lemma_)
            feature_sents.append(lemmatized)
    
    if train:
        # there will be too many negative examples generated, randomly drop some of examples

        return feature_sents, labels
    else:
        return feature_sents


cv = CountVectorizer()
cv.fit_transform()

lr = LogisticRegression()


# Example only: write out some relations to the output file
# normally you would use the list of relations output by your model
# as an example we have hard coded some relations from the training set to write to the output file
# TODO: remove this and write out the relations you extracted (obviously don't hard code them)
relations = [
    ('Hokusai', 'Japan'), 
    ('Hans Christian Andersen', 'Denmark')
    ]
write_output_file(relations)

