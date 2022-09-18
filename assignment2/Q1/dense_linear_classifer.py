import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk

# read the data
df = pd.read_csv("data/labelled_movie_reviews.csv")

# shuffle the rows
df = df.sample(frac=1, random_state=123).reset_index(drop=True)

# get the train, val, test splits
train_frac, val_frac, test_frac = 0.7, 0.1, 0.2
Xr = df["text"].tolist()
Yr = df["label"].tolist()
train_end = int(train_frac * len(Xr))
val_end = int((train_frac + val_frac) * len(Xr))
X_train = Xr[0:train_end]
Y_train = Yr[0:train_end]
X_val = Xr[train_end:val_end]
Y_val = Yr[train_end:val_end]
X_test = Xr[val_end:]
Y_test = Yr[val_end:]

data = dict(np.load("data/word_vectors.npz"))
w2v = {w: v for w, v in zip(data["words"], data["vectors"])}

# initialize tokenizer
tokenizer = nltk.tokenize.TreebankWordTokenizer()


# convert a document into a vector
def document_to_vector(doc):
    """Takes a string document and turns it into a vector
    by aggregating its word vectors.

    Args:
        doc (str): The document as a string

    Returns:
        np.array: The word vector this will be 300 dimensional.
    """
    # tokenize the input document
    tokens = tokenizer.tokenize(doc)
    matrix = []  # initialize the document matrix
    for token in tokens:
        if token not in w2v.keys():
            continue
        else:
            matrix.append(w2v[token])

    # aggregate the vectors of words in the input document using mean
    vec = np.mean(np.array(matrix), axis=0)

    return vec


# fit a linear model
def fit_model(Xtr, Ytr, C):
    """Given a training dataset and a regularization parameter
        return a linear model fit to this data.

    Args:
        Xtr (list(str)): The input training examples. Each example is a
            document as a string.
        Ytr (list(str)): The list of class labels, each element of the 
            list is either 'neg' or 'pos'.
        C (float): Regularization parameter C for LogisticRegression

    Returns:
        LogisticRegression: The trained logistic regression model.
    """
    # convert each of the training documents into a vector
    x_vec = np.array([document_to_vector(document) for document in Xtr])

    # train the logistic regression classifier
    model = LogisticRegression(max_iter=1000, C=C)
    model.fit(x_vec, Ytr)

    return model


# fit a linear model
def test_model(model, Xtst, Ytst):
    """Given a model already fit to the data return the accuracy
        on the provided dataset.

    Args:
        model (LogisticRegression): The previously trained model.
        Xtst (list(str)): The input examples. Each example
            is a document as a string.
        Ytst (list(str)): The input class labels, each element
            of the list is either 'neg' or 'pos'.

    Returns:
        float: The accuracy of the model on the data.
    """
    # convert each of the testing documents into a vector
    x_vec = np.array([document_to_vector(document) for document in Xtst])

    # test the logistic regression classifier and calculate the accuracy
    score = accuracy_score(Ytst, model.predict(x_vec))

    return score


# search for the best C parameter using the validation set
c_options = []
acc = []
cs = np.arange(59, 71, 0.25)
for c in cs:
    val_acc = test_model(fit_model(X_train, Y_train, c), X_val, Y_val)
    c_options.append(c)
    acc.append(val_acc)

plt.figure(figsize=(12, 8))
plt.plot(c_options, acc)
plt.xlabel("C")
plt.ylabel("Val acc")
plt.title("The plot of validation accuracy over regularization constant")
plt.show()

best_c = c_options[acc.index(max(acc))]
print(f"Best c chosen {best_c}")

# fit the model to the concatenated training and validation set
#   test on the test set and print the result
X_concat = Xr[0:val_end]
Y_concat = Yr[0:val_end]
print(f"The test accuracy of highest {test_model(fit_model(X_concat, Y_concat, C=best_c), X_test, Y_test)}")
