# -*- coding: utf-8 -*-

# imports and define device
from collections import Counter
import json

from datasets import load_metric, Dataset
import matplotlib.pyplot as plt
import numpy as np
from ray import tune
from sklearn.model_selection import train_test_split
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, TrainingArguments, Trainer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# preprocessing training and validation set
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


# process the training data
all_data = json.load(open("genre_train.json", "r"))
X = all_data['X'] # the titles in the training data
y = all_data['Y'] # the entity type of each training data title
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

train_dataset = Dataset.from_dict({"text": X_train, "labels": y_train})
val_dataset = Dataset.from_dict({"text": X_val, "labels": y_val})

processed_train = train_dataset.map(tokenize_function, batched=True)
processed_val = val_dataset.map(tokenize_function, batched=True)


# calculate weights for CrossEntropyLoss
train_label_dist = Counter(y_train)
class_weights = torch.zeros(4)
for key in range(4):
    class_weights[key] = train_label_dist[key]
class_weights = torch.reciprocal(class_weights)
class_weights /= torch.sum(class_weights)

# uncomment to plot imbalance training labels
# plt.bar(train_label_dist.keys(), train_label_dist.values())
# plt.xticks(list(train_label_dist.keys()))
# plt.title("Training set label distribution")
# plt.xlabel("Label")
# plt.ylabel("Count")
# plt.show()


# Custom trainer class for allowing different loss weight
class CustomTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights  # should be a tensor with size 4
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights).to(device)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# define the metrics
metric = load_metric("f1")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = torch.tensor(logits)
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average="macro")


# function for model initialization
def model_init():
    return DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=4)


# define the hyperparameter search space (uncomment for hyperparameter search)
# def hp_space(trial):
#     return {
#         "learning_rate": tune.choice([1e-4, 2e-5, 5e-5]),
#         "num_train_epochs": tune.choice([3, 5, 10]),
#         "per_device_train_batch_size": tune.choice([8, 16, 32]),
#     }


# default training arguments
training_args = TrainingArguments(
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    warmup_ratio=0.1,
    weight_decay=0.01,
    report_to="none",
    output_dir="checkpoints",
    evaluation_strategy="epoch",
    metric_for_best_model="f1",
)

trainer = CustomTrainer(
    class_weights=class_weights,
    model_init=model_init,
    args=training_args,
    train_dataset=processed_train,
    eval_dataset=processed_val,
    compute_metrics=compute_metrics,
)

# hyperparameter search with 10 trials (uncomment for hyperparameter search)
# best_hyperparameter = trainer.hyperparameter_search(
#     hp_space=hp_space,
#     direction="maximize", 
#     backend="ray", 
#     n_trials=10,
# )

# load the best hyperparameter (uncomment for hyperparameter search)
# for n, v in best_hyperparameter.hyperparameters.items():
#     setattr(trainer.args, n, v)

# train the network with given argument
trainer.train()

# evaluate the chosen model on the validation set
trainer.evaluate(processed_val)

# loading testing data
test_data = json.load(open("genre_test.json", "r"))
test_X = test_data["X"]

# process test data
test_dataset = Dataset.from_dict({"text": test_X})
processed_test = test_dataset.map(tokenize_function, batched=True)

# generate predictions using fine-tuned model
predictions = trainer.predict(processed_test)
prediction_prob = predictions[0]
results = np.argmax(prediction_prob, axis=-1)

# write predictions to csv file
fout = open("pretrained_transformer.csv", "w")
fout.write("Id,Predicted\n")
for i, line in enumerate(results):
    fout.write("%d,%d\n" % (i, line))
fout.close()
