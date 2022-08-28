import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# load the training data
train_data = json.load(open("genre_train.json", "r"))
X = train_data['X']
Y = train_data['Y']  # id mapping: 0 - Horror, 1 - Science Fiction, 2 - Humour, 3 - Crime Fiction
docid = train_data['docid']  # these are the ids of the books which each training example came from

# load the test data
# the test data does not have labels, our model needs to generate these
test_data = json.load(open("genre_test.json", "r"))
Xt = test_data['X']


# This is a very poor model which looks for keywords and if none are found it predicts
# randomly according to the class distribution in the training set
class KeywordModel(object):
    def __init__(self):
        self.counts = None

    def fit(self, X, Y):
        # fit the model
        # normally you would want to use X the training data but this simple model doesn't need it
        self.counts = np.array(np.bincount(Y), dtype=np.float32)
        self.counts /= np.sum(self.counts)

    def predict(self, Xin):
        Y_test_pred = []
        for x in Xin:
            # split into words
            xs = x.lower().split()

            # check if for our keywords
            if "scary" in xs or "spooky" in xs or "raven" in xs:  # horror book
                Y_test_pred.append(0)
            elif "science" in xs or "space" in xs:  # science fiction book
                Y_test_pred.append(1)
            elif "funny" in xs or "embarrassed" in xs:  # humor book
                Y_test_pred.append(2)
            elif "police" in xs or "murder" in xs or "crime" in xs:  # crime fiction book
                Y_test_pred.append(3)
            else:
                Y_test_pred.append(np.random.choice(len(self.counts), p=self.counts))  # predict randomly
        return Y_test_pred


# customized dataset for loading
class GenreDataset(Dataset):
    def __init__(self, input_file, transform=None):
        raw_data = json.load(open(input_file, "r"))
        self.docs = raw_data["X"]  # TODO: preprocessing here into vector
        self.labels = raw_data["Y"]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        doc = self.docs[idx]
        label = self.labels[idx]
        if self.transform:
            doc = self.transform(doc)
        return doc, label


# fit the model on the training data
model = KeywordModel()
model.fit(X, Y)

# predict on the test data
Y_test_pred = model.predict(Xt)

# write out the csv file
# first column is the id, it is the index to the list of test examples
# second column is the prediction as an integer
fout = open("out.csv", "w")
fout.write("Id,Predicted\n")
for i, line in enumerate(Y_test_pred):  # Y_test_pred is in the same order as the test data
    fout.write("%d,%d\n" % (i, line))
fout.close()
