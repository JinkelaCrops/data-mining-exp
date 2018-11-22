# 在环境变量中加入安装的Graphviz路径
import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'

from sklearn import tree
from sklearn.datasets import load_iris

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = tree.ExtraTreeClassifier()
clf = clf.fit(iris.data, iris.target)

import pydotplus

dot_data = tree.export_graphviz(clf, out_file=None, feature_names=iris.feature_names, class_names=iris.target_names,
                                filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
with open("try.png", "wb") as f:
    f.write(graph.create_png())

