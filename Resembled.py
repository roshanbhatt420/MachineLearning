from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
data=load_breast_cancer()
x,y=data.data,data.target
xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=42,test_size=0.2)

log_clf=LogisticRegression()
rnd_clf=RandomForestClassifier()
svm_clf=SVC()
voting_clf=VotingClassifier(
    estimators=[('lr',log_clf),('rf',rnd_clf),('svc',svm_clf)],voting='hard'
)
voting_clf.fit(xtrain,ytrain)

from sklearn.metrics import accuracy_score  
for clf in (log_clf,rnd_clf,svm_clf,voting_clf):
    clf.fit(xtrain,ytrain)
    y_pred=clf.predict(xtest)
    print(clf.__class__.__name__,accuracy_score(ytest,y_pred))

