from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from Resembled import xtrain, xtest,ytrain,ytest
 
bgl_clf=BaggingClassifier(RandomForestClassifier(),n_estimators=500,max_samples=100,bootstrap=False,n_jobs=-1)
# n_jobs=-1 tells cpu to use remains cpu 
# sample size is 100
bgl_clf.fit(xtrain,ytrain)
predictions=bgl_clf.predict(xtest)