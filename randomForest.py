
# code and the understandind of the code
from sklearn.ensemble import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# here Nestimators Gives the no  of trees to be made 
# max_leaf_nodes difien the depth of tree 
# n_job is used for to use available cpu in computer
x=data 
y=data
rnd_clf=RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
rnd_clf.fit(x,y)
# extra randomness in the data 
# for better overall model prediction 
from sklearn.ensemble import BaggingClassifier 
bag_clf = BaggingClassifier(
DecisionTreeClassifier(splitter="random", max_leaf_nodes=16),
n_estimators=500, max_samples=1.0, bootstrap=True, n_jobs=-1)