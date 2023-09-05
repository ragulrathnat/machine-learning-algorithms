import pandas as pd
import numpy as ny
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


titanic_data = fetch_openml('titanic',version = 1, as_frame= True)

df = titanic_data['data']

df['survied'] = titanic_data['target']


sns.countplot(x='survied',data=df)
#plt.show()
sns.countplot(x='survied',hue='sex',data=df)
#plt.show()
sns.countplot(x='survied',hue='pclass',data=df)
#plt.show()
df['age'].plot.hist(title='age precentage')
#plt.show()

#print(df.isnull().sum())


miss_val = pd.DataFrame(df.isnull().sum()/len(df) * 100)

miss_val.plot(kind='bar',title='missing value percentage')
#plt.show()

df['family'] = df['sibsp'] + df['parch']
df.loc[df['family'] > 0, 'traveling_alone'] = 0
df.loc[df['family'] == 0, 'traveling_alone'] = 1

df.drop(['sibsp','parch'],axis=1,inplace=True)
sns.countplot(x='traveling_alone',data=df)
#plt.show()


df.drop(['name','ticket','home.dest'], axis=1, inplace=True)

sex = pd.get_dummies(df['sex'],drop_first=True)

df['sex'] = sex

imp_mean = SimpleImputer(strategy='mean')

df['age'] = imp_mean.fit_transform(df[['age']])
df['fare'] = imp_mean.fit_transform(df[['fare']])

# imp_freq = SimpleImputer(strategy='most_frequent')

# df['embarked'] = imp_freq.fit_transform(df[['embarked']])
y = df['survied']
df.drop(['embarked','cabin','boat','body','survied'],axis=1, inplace=True)

x = df

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=43)


model = LogisticRegression()
model.fit(x_train,y_train)
ypred = model.predict(x_test)

print("accuracy score",accuracy_score(y_test,ypred))