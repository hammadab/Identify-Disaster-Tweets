import pandas as pd
from sklearn import naive_bayes, tree
import re
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split, cross_val_score

# Load Data
data = pd.read_csv('./socialmedia-disaster-tweets-DFE.csv', encoding='ISO-8859-1')[['choose_one', 'text']]
data = data[data.choose_one != 'Can\'t Decide']
data = pd.DataFrame(data.values.tolist())

text = data[1].values
labels = [str(x) for x in data[0].values]

# labels = []
# for i in range(0, len(data[0].values)):
#     if data[0].values[i] == 'Relevant':
#         labels.append(1)
#     else:
#         labels.append(0)
# labels = np.array(labels)

# Clean Data
for i in range(0, len(text)):
    text[i] = re.compile(r'https?://\S+|www\.\S+').sub(r'', text[i])
    text[i] = re.compile(r'<.*?>').sub(r'', text[i])
    text[i] = re.compile("[" u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF" u"\U0001F680-\U0001F6FF" u"\U0001F1E0"
                         u"-\U0001F1FF" u"\U00002702-\U000027B0" u"\U000024C2-\U0001F251" "]+",
                         flags=re.UNICODE).sub(r'', text[i])

tknzr = Tokenizer()
tknzr.fit_on_texts(text)
text = tknzr.texts_to_matrix(text)

X_train, X_test, y_train, y_test = train_test_split(text, labels, test_size=0.2, random_state=0)

model = naive_bayes.GaussianNB()
print("GaussianNB: 10-fold cross validation accuracy:")
print(100 * cross_val_score(model, X_train, y_train, cv=10))
model.fit(X_train, y_train)
print('Accuracy on test set =', '%.2f' % (100 * sum(model.predict(X_test) == y_test) / len(y_test)), '%')

model = naive_bayes.MultinomialNB()
print("MultinomialNB: 10-fold cross validation accuracy:")
print(100 * cross_val_score(model, X_train, y_train, cv=10))
model.fit(X_train, y_train)
print('Accuracy on test set =', '%.2f' % (100 * sum(model.predict(X_test) == y_test) / len(y_test)), '%')

model = naive_bayes.ComplementNB()
print("ComplementNB: 10-fold cross validation accuracy:")
print(100 * cross_val_score(model, X_train, y_train, cv=10))
model.fit(X_train, y_train)
print('Accuracy on test set =', '%.2f' % (100 * sum(model.predict(X_test) == y_test) / len(y_test)), '%')

model = naive_bayes.BernoulliNB()
print("BernoulliNB: 10-fold cross validation accuracy:")
print(100 * cross_val_score(model, X_train, y_train, cv=10))
model.fit(X_train, y_train)
print('Accuracy on test set =', '%.2f' % (100 * sum(model.predict(X_test) == y_test) / len(y_test)), '%')

model = naive_bayes.CategoricalNB()
print("CategoricalNB: 10-fold cross validation accuracy:")
print(100 * cross_val_score(model, X_train, y_train, cv=10))
model.fit(X_train, y_train)
print('Accuracy on test set =', '%.2f' % (100 * sum(model.predict(X_test) == y_test) / len(y_test)), '%')

model = tree.DecisionTreeClassifier()
print("DecisionTreeClassifier: 10-fold cross validation accuracy:")
print(100 * cross_val_score(model, X_train, y_train, cv=10))
model.fit(X_train, y_train)
print('Accuracy on test set =', '%.2f' % (100 * sum(model.predict(X_test) == y_test) / len(y_test)), '%')


## Neural Networks
import re
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

data = pd.read_csv('./socialmedia-disaster-tweets-DFE.csv', encoding='ISO-8859-1')[['choose_one', 'text']]
data = data[data.choose_one != 'Can\'t Decide']
data = pd.DataFrame(data.values.tolist())

text = data[1].values
labels = [str(x) for x in data[0].values]

labels = []
for i in range(0, len(data[0].values)):
    if data[0].values[i] == 'Relevant':
        labels.append(1)
    else:
        labels.append(0)
labels = np.array(labels).reshape((-1, 1))

# # Clean Data
# for i in range(0, len(text)):
#     # text[i] = text[i].lower()
#     text[i] = re.compile(r'https?://\S+|www\.\S+').sub(r'', text[i])
#     text[i] = re.compile(r'<.*?>').sub(r'', text[i])
#     text[i] = re.compile("[" u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF" u"\U0001F680-\U0001F6FF" u"\U0001F1E0"
#                          u"-\U0001F1FF" u"\U00002702-\U000027B0" u"\U000024C2-\U0001F251" "]+",
#                          flags=re.UNICODE).sub(r'', text[i])
#     # text[i] = text[i].translate(str.maketrans('', '', string.punctuation))
#     # text[i] = text[i].split()
#     # text[i] = [x for x in text[i] if len(x)>1]
#     # text[i] = str(text[i])
# # text = np.array(text)
#
# tknzr = Tokenizer()
# tknzr.fit_on_texts(text)
# # text = tknzr.texts_to_matrix(text)


# text = tknzr.texts_to_sequences(text)
# text = tf.keras.preprocessing.sequence.pad_sequences(text, padding='post')

X_train, X_test, y_train, y_test = train_test_split(text, labels, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1 / 8, random_state=0)

# train_set = tf.data.Dataset.from_tensor_slices((X_train, y_train))
# val_set = tf.data.Dataset.from_tensor_slices((X_val, y_val))
# test_set = tf.data.Dataset.from_tensor_slices((X_test, y_test))

model = tf.keras.Sequential()
model.add(hub.KerasLayer("https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1", input_shape=[], dtype=tf.string,
                         trainable=True))
# model.add(tf.keras.layers.Embedding(22739, 128))
model.add(tf.keras.layers.Dense(64, activation='sigmoid'))
model.add(tf.keras.layers.Dense(32, activation='sigmoid'))
model.add(tf.keras.layers.Dense(16, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.summary()

with tf.device('/GPU:0'):
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])
    history = model.fit(X_train, y_train, batch_size=128, epochs=20, verbose=1, validation_data=(X_val, y_val),
                        shuffle=True)

model.evaluate((X_test, y_test))
