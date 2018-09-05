from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# load data
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)


print("len(X_train[0]):", len(X_train[0]))
print("len(X_train):", len(X_train))
print("len(y_train):", len(y_train))
print("len(X_test:)", len(X_test))
print("len(y_test:)", len(y_test))
print("len(X_train)/len(X_test):", len(X_train)/len(X_test))
print(y[:10])
print((y*5)[:10])
