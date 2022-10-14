from itertools import product
import json
import pandas as pd
import spacy
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
import multiprocessing
import tqdm


# read in the data
train_data = json.load(open("sents_parsed_train.json", "r"))
test_data = json.load(open("sents_parsed_test.json", "r"))

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

# TODO: build a training/validation/testing pipeline for relation extraction
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
    nationality = '/people/person/nationality'

    for sent in tqdm.tqdm(data_dict):
        # extract relation indices
        relation_a, relation_b = sent["relation"]["a"], sent["relation"]["b"]
        idx_a = sent["relation"]["a_start"], sent["relation"]["a_start"] + len(relation_a)
        idx_b = sent["relation"]["b_start"], sent["relation"]["b_start"] + len(relation_b)
        
        # Features: words in between entities, entities themselves, words in particular positions
        start = max(min(idx_a[0], idx_b[0]) - 1, 0)
        end = min(max(idx_a[1], idx_b[1]) + 1, len(sent["tokens"]))

        # preprocessing bag of word
        processed_sequence = sentence_processing(sent, start, end)
        # Features: headword and the concatenation of them
        head_a = relation_a[0]
        head_b = relation_b[0]
        concatenation = f"{head_a}-{head_b}"
        processed_sequence.extend([head_a, head_b, concatenation])

        feature_sents.append(" ".join(processed_sequence))

        # preprocessing labels
        if sent["relation"]["relation"] == nationality:
            labels.append(1)
        else:
            labels.append(0)
    
    cv = CountVectorizer(ngram_range=(1, 2))
    train_features = cv.fit_transform(feature_sents)

    return train_features, cv, labels


def test_preprocess(data_dict, cv):
    feature_sents = []
    entities = []
    entity_distance = []

    for sent in tqdm.tqdm(data_dict):
        # we only extract GPE and PERSON as they are the only chance for nationality
        gpes_indices = []
        persons_indices = []
        for entity in sent["entities"]:
            # could be the case that it have stopwords so doesn't belong to countries
            if entity["label"] == "GPE":
                # check if the entity is in country list
                entity_word = " ".join([sent["tokens"][index]
                        for index in range(entity["start"], entity["end"]) if not sent["isstop"][index]])
                if entity_word in countries:
                    gpes_indices.append((entity["start"], entity["end"]))
            elif entity["label"] == "PERSON":
                # person names
                persons_indices.append((entity["start"], entity["end"]))

        entity_indices_combinations = product(persons_indices, gpes_indices)
        
        # extract features for each entity pairs in the given list
        for idx_a, idx_b in entity_indices_combinations:
            # allow a extension of window 3
            start = max(min(idx_a[0], idx_b[0]) - 1, 0)
            end = min(max(idx_a[1], idx_b[1]) + 1, len(sent["tokens"]))

            # preprocessing features
            processed_sequence = sentence_processing(sent, start, end)
            # we also add headword and the concatenation of them
            head_a = sent["tokens"][idx_a[0]]
            head_b = sent["tokens"][idx_b[0]]
            concatenation = f"{head_a}-{head_b}"

            processed_sequence.extend([head_a, head_b, concatenation])

            # append features to the list
            feature_sents.append(" ".join(processed_sequence))
            # append the entity words used, we need to skip stopwords in entities
            entities.append(
                (
                    " ".join([sent["tokens"][index]
                        for index in range(idx_a[0], idx_a[1]) if not sent["isstop"][index]]),
                    " ".join([sent["tokens"][index]
                        for index in range(idx_b[0], idx_b[1]) if not sent["isstop"][index]])
                )
            )
            # append distance between entities
            entity_distance.append(end - start)

    features = cv.transform(feature_sents)

    return features, entities, entity_distance


def test_postprocess(prediction_prob, entities, entity_distance):
    relations = []
    name_probs = {}
    for i, prob in enumerate(prediction_prob):
        name = entities[i][0]
        confidence = prob[1]
        if (name not in name_probs or confidence > name_probs[name][0]) and confidence > 0.5 and entity_distance[i] < 30:
            # if name not in dict or higher confidence, update attribute
            name_probs[name] = (confidence, entities[i][1])
    for key, value in name_probs.items():
        relations.append((key, value[1]))
    write_output_file(relations=relations)


train_x, cv, train_y = train_preprocess(train_data)

lr = LogisticRegression()
gs = GridSearchCV(
    estimator=lr,
    param_grid={"C": (0.1, 1, 100, 1000, 10000), "max_iter": (100, 1000, 3000), "class_weight": (None, "balanced")},
    scoring="f1_macro",
    n_jobs=multiprocessing.cpu_count(),
    refit=True
)
gs.fit(train_x, train_y)
print(f"Used parameters:\n{gs.best_params_}")

# Example only: write out some relations to the output file
# normally you would use the list of relations output by your model
# as an example we have hard coded some relations from the training set to write to the output file
# TODO: remove this and write out the relations you extracted (obviously don't hard code them)
test_features, entities, entity_distance = test_preprocess(test_data, cv)
prediction_prob = gs.predict_proba(test_features)
test_postprocess(prediction_prob, entities, entity_distance)
