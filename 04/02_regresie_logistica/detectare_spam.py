import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

data = pd.read_csv('dataset_spam_2.csv')

print(type(data['Label'][0]))

train_data, test_data, train_labels, test_labels = train_test_split(
    data['Text'], data['Label'],
    test_size=0.2,
    random_state=42)

vectorized = TfidfVectorizer(max_features=5000)
X_train = vectorized.fit_transform(train_data)

X_test = vectorized.transform(test_data)

model = LogisticRegression()
model.fit(X_train, train_labels)

predictions = model.predict(X_test)

"""smote = SMOTE()
train_data_resampled, train_labels_resampled = smote.fit_resample(
    train_data.to_frame(), train_labels
)

pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(max_features=5000)),
    ('classifier', LogisticRegression())
])

pipeline.fit(train_data_resampled['Text'], train_labels_resampled)

predictions = pipeline.predict(test_data)"""

acc = accuracy_score(test_labels, predictions)

conf_mat = confusion_matrix(test_labels, predictions)
rap = classification_report(test_labels, predictions)

print(acc, conf_mat, '\n', rap)




