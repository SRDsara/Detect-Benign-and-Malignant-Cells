import matplotlib.pyplot as plt
import pandas as pd 
from sklearn import metrics, preprocessing, svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#1- import dataset
df = pd.read_csv("F:\\tutorials\machine learning\machine_learning_with_python_jadi-main\cell_samples.csv")

#2- clean data: handle missing data
df = df[pd.to_numeric(df["BareNuc"], errors="coerce").notnull()]
df = df.astype({"BareNuc":'int64'})

#3- plotting
ax = df[df["Class"]==4].plot(kind="scatter", color="red", x="MargAdh", y="Mit", label="malignant")
df[df["Class"]==2].plot(kind="scatter", color="green", x="MargAdh", y="Mit", label="malignant", ax=ax)
plt.show()

# define X(features) and y(label)
X=  df[["Clump","UnifSize","UnifShape","MargAdh","SingEpiSize","BareNuc","BlandChrom","NormNucl","Mit"]]
y = df[["Class"]]

#4- normalize
scale = preprocessing.StandardScaler().fit(X)
X = scale.fit_transform(X)

#5- train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#6- modeling
clf = svm.SVC(kernel="sigmoid")
clf.fit(X_train, y_train.values.ravel()) 

#7- predicting
y_pred = clf.predict(X_test)

#8- confusiosn matrix
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [2, 4])
cm_display.plot()
plt.show()

#9- evaluation
print(accuracy_score(y_test, y_pred))

