from sklearn.datasets import load_iris
iris= load_iris()
X= iris.data

y= iris.target
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=12)
knn_clf=knn.fit(X, y)
import joblib
joblib.dump(knn_clf, "Knn_Classifier.pkl")