
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix
import shap
import pandas as pd
import io
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
shap.initjs()
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

'''PATHWIN = "C:/Users/hanna/OneDrive/Dokumente/#Florida/#Spring Semester/A8-Big-Data-Submission/compdata/win/0_0-win.csv"
data = pd.read_csv(PATHWIN)'''

PATHDFT = "C:/Users/hanna/OneDrive/Dokumente/#Florida/#Spring Semester/A8-Big-Data-Submission/compdata/dft/7_0-dft.csv"
data = pd.read_csv(PATHDFT)

print(data.shape)
print(data.head())


'''#calculate new lable
selected_columns = ['5000', '10001', '15002', '20003', '25004']
data_labels = data[selected_columns]
print(data_labels)
data['lable'] = round(data[selected_columns].mean(axis=1))
data = data.drop(['5000', '10001', '15002', '20003', '25004'], axis=1)
print(data.head())'''

print("testsplit")
#train test split
X = data.drop('5000', axis=1)
y = data['5000']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("oversampling")
#oversampling
'''smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)'''

ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)


print("train rf")
# Random forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_resampled, y_train_resampled)

y_pred = rf_classifier.predict(X_test)


# Accuracy
print("accuracy")
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# Print classification report
print("Classification Report:")
report = classification_report(y_test, y_pred, target_names=["Paradoxical Sleep", "Slow-Wave Sleep", "Wakefulness"])
print(report)
df_report = pd.read_csv(io.StringIO(report), sep='\s+')
plt.figure(figsize=(8, 6))
sns.heatmap(df_report.iloc[:-3, 1:].astype(float), annot=True, cmap='Blues', fmt=".2f", linewidths=.5)
plt.title('Classification Report Heatmap')
plt.show()
plt.savefig('Classification_Report')


# Print confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
# Compute confusion matrix
disp = plot_confusion_matrix(conf_matrix, X_test, y_test, cmap=plt.cm.Blues, normalize='true')
disp.ax_.set_title('Confusion Matrix')
plt.show()

'''
explainer = shap.Explainer(rf_classifier)

shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)

shap.summary_plot(shap_values[0], X_test)'''