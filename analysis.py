import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

df = pd.read_csv("heart disease.csv")

print("shape of dataset :")
print(df.shape)
print("First 5 rows of the dataset :")
print(df.head())
print("Info of the dataset :")
print(df.info())
print("Null values :")
print(df.isna().sum())
print("Duplicate values :")
print(df.duplicated())
print("Columns :" )
print(df.columns)

print("Detail of Age :")
print(df['Age'].describe())

print("Detail of chestpain :")
print(df['ChestPain'].describe())

print("Detail of sex:")
print(df['Sex'].describe())

print("Detail of RestBP :")
print(df['RestBP'].describe())

print("Detail of cholestrol :")
print(df['Chol'].describe())

print("Detail of fasting blood suger level :")
print(df['Fbs'].describe())

print("Detail of Rest ECG :")
print(df['RestECG'].describe())

print("Detail of MAX HEART RATE :")
print(df['MaxHR'].describe())

print("Detail of EX Angina :")
print(df['ExAng'].describe())

print("Detail of ST depression during exercise :")
print(df['Oldpeak'].describe())

print("Detail of slope of ST curve  :")
print(df['Slope'].describe())

print("Detail of ca :")
print(df['Ca'].describe())

print("Detail of Thalassemia : ")
print(df['Thal'].describe())

print("Detail of target :")
print(df['Target'].describe())

# visualization 
sns.histplot(x=df['Age'] , bins=4 , kde=True)
plt.show()

Sex_in_cat = df['Sex'].map({0:'female',1:'male'})
sns.countplot(x=Sex_in_cat ,data=df , palette='pastel')
plt.show()

sns.countplot(x='ChestPain',data=df,palette="pastel")
plt.show()

sns.histplot(x=df["RestBP"] , bins=4 , kde=True , palette="rainbow")
plt.show()

sns.histplot(x=df['Chol'], bins=5 , kde=True)
plt.show()

sns.histplot(x=df['Fbs'] , bins=3 , kde=True)
plt.show()

sns.histplot(x=df['RestECG'] , bins=4, kde=True)
plt.show()

sns.histplot(x=df['MaxHR'] , bins=4 , kde=True)
plt.show()

Ang_in_cat=df["ExAng"].map({0:'No pain',1:'pain'})
sns.countplot(x=Ang_in_cat , data=df , palette="pastel")
plt.show()

sns.histplot(x=df['Oldpeak'] , bins=4 , kde=True)
plt.show()

sns.histplot(x=df['Slope'] , bins=3 , kde=True)
plt.show()

sns.histplot(x=df['Ca'] , bins=3 , kde=True)
plt.show()

sns.countplot(x='Thal',data=df, palette='pastel')
plt.show()

Target_in_cat=df["Target"].map({0:'NO Disease',1:'Disease'})
sns.countplot(x=Target_in_cat,data=df,palette="pastel")
plt.show()

pd.crosstab(df['Age'],df['Target'], normalize='index').plot(kind='bar',stacked=True)
plt.title("Percentage of disease by Age")
plt.ylabel("Target by  using Percentage")
plt.show()

sns.countplot( x=Sex_in_cat , hue='Target',data=df , palette='Set2')
plt.title("Sex vs Heart Disease")
plt.legend(title="Heart Disease",labels=['No Disease','Disease'])
plt.show()

pd.crosstab(df['ChestPain'],df['Target'],normalize='index').plot(kind='bar')
plt.title("Percentage of Disease by ChestPain")
plt.ylabel("Target by  using percentage ")
plt.show()

sns.boxplot(x='Target',y='RestBP',data=df,palette='pastel')
plt.title("Disease Prediction by RestBP")
plt.legend(labels=['No Disease','Disease'])
plt.show()

sns.boxplot(x='Target',y='Chol',data=df,palette='pastel')
plt.title("Disease Prediction by Cholestrol")
plt.legend(labels=['No Disease','Disease'])
plt.show()

sns.countplot(x='Fbs' , hue='Target' , data=df, palette='pastel')
plt.title("Fbs vs Heart Disease")
plt.legend(title="Heart Disease",labels=['No Disease','Disease'])
plt.show()

sns.countplot(x='RestECG',hue='Target',data=df , palette="rainbow")
plt.title("RestECG vs Heart Disease")
plt.legend(title="Heart Disease",labels=['No Disease','Disease'])
plt.show()

pd.crosstab(df['MaxHR'],df['Target'],normalize='index').plot(kind='hist')
plt.title("Percentage of  Heart Disease by MaxHR")
plt.ylabel("Target by using percentage")
plt.show()

sns.countplot(x='ExAng',hue='Target',data=df , palette="rainbow")
plt.title("ExAng vs Heart Disease")
plt.legend(title="Heart Disease",labels=['No Disease','Disease'])
plt.show()

sns.boxplot(y='Oldpeak',x='Target',data=df , palette="pastel")
plt.title("Oldpeak vs Heart Disease")
plt.legend(title="Heart Disease",labels=['No Disease','Disease'])
plt.show()


sns.countplot(x='Slope',hue='Target',data=df , palette="rainbow")
plt.title("Slope vs Heart Disease")
plt.legend(title="Heart Disease",labels=['No Disease','Disease'])
plt.show()

sns.countplot(x='Ca',hue='Target',data=df , palette="rainbow")
plt.title("Ca vs Heart Disease")
plt.legend(title="Heart Disease",labels=['No Disease','Disease'])
plt.show()

sns.countplot(x='Thal',hue='Target',data=df , palette="rainbow")
plt.title("Thal vs Heart Disease")
plt.legend(title="Heart Disease",labels=['No Disease','Disease'])
plt.show()


# Correlation of data
numeric_data=df[['Age','Sex','RestBP','Chol','Fbs','RestECG','MaxHR','ExAng','Oldpeak','Slope','Ca','Target']]
sns.heatmap(numeric_data.corr(),annot=True)
plt.show()


# outlier detection 

outlier_detection = ['RestBP','Chol','MaxHR','Oldpeak']

for col in outlier_detection:
    Q1=df[col].quantile(0.25)
    Q3=df[col].quantile(0.75)
    IQR=Q3-Q1
    lower=Q1 - 1.5 * IQR
    upper=Q3 + 1.5 * IQR
    outlier = df[(df[col] < lower) | (df[col] > upper)]
    print(col, " -> " , len(outlier) ,"outliers ")
    print(col," outlier are :",outlier[col].value_counts().sort_index())

df['Thal']=df['Thal'].fillna(df['Thal'].max())
print(df['Thal'].isnull().sum())

