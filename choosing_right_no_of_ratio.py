# anther very usefull information  is explained varience ratio
# for each principal 
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
data=load_iris()
x=data.data
y=data.target

xtrain,xtest,ytrain, ytest=train_test_split(x,y,random_state=42,test_size=0.2)
from sklearn.decomposition import PCA
pca.explained_varience_ratio_
# choosing the right no of dimensions 
from sklearn.decomposition import PCA
pca=PCA()
pca.fit(xtrain)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1
