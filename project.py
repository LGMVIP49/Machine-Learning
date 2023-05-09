import pandas as pd
# Reading the CSV file
df = pd.read_csv('https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv')
# Printing top 5 rows
df.head()

df.info()

df.describe()

df.isnull().sum()

data = df.drop_duplicates(subset ="variety",)
data

df.value_counts("variety")

# importing packages
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='variety', data=df, )
plt.show()

# importing packages
import seaborn as sns
import matplotlib.pyplot as plt
sns.scatterplot(x='sepal.length', y='sepal.width',
				hue='variety', data=df, )
# Placing Legend outside the Figure
plt.legend(bbox_to_anchor=(1, 1), loc=2)
plt.show()

# importing packages
import seaborn as sns
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 2, figsize=(10,10))
axes[0,0].set_title("Sepal Length")
axes[0,0].hist(df['sepal.length'], bins=7)
axes[0,1].set_title("Sepal Width")
axes[0,1].hist(df['sepal.width'], bins=5);
axes[1,0].set_title("Petal Length")
axes[1,0].hist(df['petal.length'], bins=6);
axes[1,1].set_title("Petal Width")
axes[1,1].hist(df['petal.width'], bins=6);

data.corr(method='pearson')

# importing packages
import seaborn as sns
import matplotlib.pyplot as plt
def graph(y):
	sns.boxplot(x="variety", y=y, data=df)
plt.figure(figsize=(10,10))
# Adding the subplot at the specified
# grid position
plt.subplot(221)
graph('sepal.length')
plt.subplot(222)
graph('sepal.width')
plt.subplot(223)
graph('petal.length')
plt.subplot(224)
graph('petal.width')
plt.show()

# importing packages
import seaborn as sns
import matplotlib.pyplot as plt
# Load the dataset
df = pd.read_csv('https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv')
sns.boxplot(x='sepal.width', data=df)

# Importing
import sklearn
from sklearn.datasets import load_boston
import pandas as pd
import seaborn as sns
import numpy as np
# Load the dataset
df = pd.read_csv('https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv')
# IQR
Q1 = np.percentile(df['sepal.width'], 25,
				interpolation = 'midpoint')
Q3 = np.percentile(df['sepal.width'], 75,
				interpolation = 'midpoint')
IQR = Q3 - Q1
print("Old Shape: ", df.shape)
# Upper bound
upper = np.where(df['sepal.width'] >= (Q3+1.5*IQR))
# Lower bound
lower = np.where(df['sepal.width'] <= (Q1-1.5*IQR))
# Removing the Outliers
df.drop(upper[0], inplace = True)
df.drop(lower[0], inplace = True)
print("New Shape: ", df.shape)
sns.boxplot(x='sepal.width', data=df)

#load the Iris dataset
from  sklearn import  datasets
iris=datasets.load_iris()

x=iris.data
y=iris.target

#Splitting the dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.5)

from sklearn import tree
classifier=tree.DecisionTreeClassifier()

from sklearn import neighbors
classifier=neighbors.KNeighborsClassifier()

#Train the Model.
classifier.fit(x_train,y_train)

#Predictions can be done with predict function
predictions=classifier.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))
