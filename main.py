import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Define functions to read email data
def read_spam():
    category = 'spam'
    directory = './training_data/spam'
    return read_category(category, directory)

def read_ham():
    category = 'ham'
    directory = './training_data/ham'
    return read_category(category, directory)

def read_category(category, directory):
    emails = []
    for filename in os.listdir(directory):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(directory, filename), 'r', errors='ignore') as fp:
            try:
                content = fp.read()
                emails.append({'name': filename, 'content': content, 'category': category})
            except:
                print(f'Skipped {filename}')
    return emails

# Read and prepare data
ham = read_ham()
spam = read_spam()
df = pd.concat([pd.DataFrame.from_records(ham), pd.DataFrame.from_records(spam)], ignore_index=True)

# Vectorization
def preprocessor(e):
    return e  # Modify if preprocessing is needed

vectorizer = CountVectorizer(preprocessor=preprocessor)
X = df['content']
y = df['category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vect, y_train)

# Validate model
y_pred = model.predict(X_test_vect)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Feature importance
feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]
feature_importance = sorted(zip(feature_names, coefficients), key=lambda x: abs(x[1]), reverse=True)
top_positive_features = [feat for feat in feature_importance if feat[1] > 0][:10]
top_negative_features = [feat for feat in feature_importance if feat[1] < 0][:10]

print("Top 10 Positive Features:")
for feature, importance in top_positive_features:
    print(f"{feature}: {importance}")

print("Top 10 Negative Features:")
for feature, importance in top_negative_features:
    print(f"{feature}: {importance}")
