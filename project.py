from tkinter.font import names
import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
from io import StringIO
from IPython.display import Image
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import export_graphviz
import pydotplus


df = pd.read_csv('data2.csv')

features = ["PassengerId","Pclass","Sex","Age","SibSp","Parch"]

X = df[features]
Y = df["Survived"]


x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=1)

clf = DecisionTreeClassifier(max_depth=)
clf = clf.fit(x_train,y_train)
from tkinter.font import names
import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
from io import StringIO
from IPython.display import Image
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import export_graphviz
import pydotplus


df = pd.read_csv('data2.csv')

features = ["PassengerId","Pclass","Sex","Age","SibSp","Parch"]

X = df[features]
Y = df["Survived"]


x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=1)

clf = DecisionTreeClassifier(max_depth = 3)
clf = clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)

dot_data = StringIO()

export_graphviz(clf , out_file= dot_data , filled= True , special_characters= True , rounded= True  , feature_names= features , class_names= ['0','1'])
# print(doc_data.getvalue())

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

graph.write_png('project.png')

Image(graph.create_png())
y_pred = clf.predict(x_test)

dot_data = StringIO()

export_graphviz(clf , out_file= dot_data , filled= True , special_characters= True , rounded= True  , feature_names= features , class_names= ['0','1'])
# print(doc_data.getvalue())

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

graph.write_png('project.png')

Image(graph.create_png())