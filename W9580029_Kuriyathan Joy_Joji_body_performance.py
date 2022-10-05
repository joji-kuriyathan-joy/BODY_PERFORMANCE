# %% [markdown]
# ## Importing well-known Python libraries that will be used in the development of this project.

# %%
#import libraries
import os
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split

from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tsf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical

from sklearn import metrics
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_recall_fscore_support as score

import warnings
warnings.filterwarnings('ignore') 

# %% [markdown]
# ## Body performance multi class classification:
# **Objective**:The objective of this project is to classify the body performance and grading into body type catagory, the grade of performance of body is dependent upon  age and some exercise performance data.
# 
# The dataset is taken from dataset repository of Kaggle
# https://www.kaggle.com/datasets/kukuroo3/body-performance-data

# %%
#load dataset
__file__ = 'bodyPerformance.csv'
df=pd.read_csv(os.path.join(os.getcwd(), __file__))
df.head()

# %%
#Checking the size of the given dataset.
print("The given data is of shape ",df.shape)

# %%
#checking information of dataset
df.info()

# %% [markdown]
# **Experiment** :
# 
# The above cell method prints the information about the DataFrame. The information contains the number of columns, column labels, column data types, memory usage, range index, and the number of cells in each column (non-null values).
# 
# **Observation**:
# 
# Gender and class columns are of object type that need encoding to get converted into numerical form.

# %%
#function to convert object type to numeric type
def object_to_numeric(data):
  s={}
  index=0
  for i in df[data]:
    if i not in s:
      s[i]=index
      index=index+1
  df[data]=df[data].map(s)

#object type features in the data
obj_list=['gender','class']  

#function is called to covert object datatype to numeric datatype
for j in obj_list:
    object_to_numeric(j)

# %% [markdown]
# **Experiment**:
# 
# The above function maps the object type to integer type.
# you can see the difference in the below cell since all object type columns are converted to float and int type
# 

# %%
#checking the information of datatype after conversion
df.info()

# %% [markdown]
# ##Exploratory Data Analysis

# %%
#gender vs age
sns.set(rc = {'figure.figsize':(10,5)})
sns.barplot(x='gender',y='age',data=df,hue='class',palette="colorblind")

# %% [markdown]
# Barplot between age and gender and are differentiated by target classes
# 
# **Observation**:
# 
# 1)Male age is between 30-40 
# 
# 2)Female age is above 30
# 
# 3)clear observation of age of different classes with respect to gender can be seen.
# 
# 

# %%
#gender vs height
sns.set(rc = {'figure.figsize':(10,5)})
sns.barplot(x='gender',y='height_cm',data=df,hue='class', palette="colorblind",dodge=True)

# %% [markdown]
# Barplot between height and gender and are differentiated by target classes
# 
# **Observation**:
# 
# 1) Male height is above 150 and below 175 centimeter 
# 
# 2) female height is  below 160 centimeter
# 
# 3)Clear observation of height of different classes with respect to height in cm can be seen.
# 

# %%
#gender vs weight
sns.set(rc = {'figure.figsize':(10,5)})
sns.barplot(x='gender',y='weight_kg',data=df,hue='class',palette="colorblind")

# %% [markdown]
# Barplot between weight in kg and gender and are differentiated by target classes
# 
# **Observation**:
# 
# 1)male weight is above 70kg
# 
# 2)female weight is  below 60kg
# 
# 3)Clear observation of weight of different classes with respect to weight in kg can be seen.
# 

# %%
#gender vs body fat
sns.set(rc = {'figure.figsize':(10,5)})
sns.barplot(x='gender',y='body fat_%',data=df,hue='class',palette="colorblind")

# %% [markdown]
# Barplot between body and gender and are differentiated by target classes
# 
# **Observation**:
# 
# 1) Male bodyfat is less than 25 percent
# 
# 2) female bodyfat is greater than equal to 25 percent
# 
# 3)Clear observation of bodyfat of different classes with respect to fat in % can be seen.
# 

# %%
#count plot of fetal health
sns.set(rc = {'figure.figsize':(5,5)})
plt.title("countplot")
plt.xlabel("class")
plt.ylabel("count")
one,two,three,four=df['class'].value_counts()
sns.barplot(x=["one","two","three","four"],y=[one,two,three,four],palette="colorblind")

# %% [markdown]
# 
# The countplot above shows number of values of each label in the column. The main objective behind the plot is to check whether the data is balanced or imbalanced
# 
# **Observation**:
# 
#  The obseravtion shows count of each label are equal, hence dataset is balanced.

# %%
#Observing frequency of features
hist=df.hist(figsize=[15,20],color="purple")

# %% [markdown]
# Histogram represents the distribution of a continuous variable over a given interval or period of time by this we can figure out the values within the bins in each column for comparitive study
# 
# **Observation**
# 
# 1)sit-ups counts,broad jump_cm,height_cm,weight_kg columns has  Binomial distribution 
# 
# 2)Age has Exponential distribution
# 
# 3)Gripforce has Probability Distributions

# %%
#check for null values
sns.set(rc = {'figure.figsize':(10,5)})
sns.heatmap(df.isnull(),cmap="Blues")

# %% [markdown]
# Heatmap plotted above represents that if null values present or not, if there null value are present then we have to resolve it by either droping or applying statistical approach.
# 
# **Observation:**
# 
# No nan values present in the dataset since no noise can be encountered.
# 
# 
# 
# 

# %%
#check correlation of dataset
sns.set(rc = {'figure.figsize':(15,8)})
sns.heatmap(df.corr(),annot=True,cmap = 'ocean')

# %% [markdown]
# Correlation between all the features have been shown using heatmap. The lighter the color the more it is correlated.
# 
# **Observation**
# 
# Correlation matrix represented using heatmap shows correlation between dependent and independent variables also among each other. Variables are not much correlated as can be observed through their magnitude hence, it would not cause complexity in the model.

# %%
#printing all values which are highly correlated
list_column=list(df.columns)
new_list=[]
for i in list_column:
  if(df['class'].corr(df[i])>0):
    new_list.append(i)
new_list 

# %% [markdown]
# **Objective**
# 
# Printing all the features that are highly correlated with the with other features so that we can check the type of relation they have and decide whether to ignore the collinearity or remove it.
# 

# %%
#Pair-wise plotting features
sns.set(font_scale=3)
sns.set(rc = {'figure.figsize':(10,8)})
g = sns.pairplot(df, hue="class", height=5, aspect=1.5, palette='magma')
g.fig.subplots_adjust(top=0.95)
g.fig.suptitle('Relation ship', fontsize=40)
plt.show()

# %% [markdown]
# The pattern of scatter plot indicates the relationship between variables.The relationship can be linear or non-linear.
# Scatter plot shows the relationship between two variable but does not indicates the strength of relationship
# 
# **Observation**:
# 
# 1)the above plot tells the correlation among all the independent features, checking correlation will help in using most significant features to avoid high dimentionality, many of features have less correlation for example, correlation between systolic and weight_kg,diastolic and weight_kg.
# 
# 2)we are using all features since all features are relevant
# 
# 3) this plot also helps in determining the outliers if any.

# %%
#checking for outliers, if any
df.boxplot(figsize=[30,5])

# %% [markdown]
# **Observation**
# 
# from the given plot we can observe that there are 8 columns(height_cm,weigh_cm,body_fat,diastolic,sit and bend forward,broad jump) that contain outliers but analysis shows that the outliers are not too far from the quantiles therefore, these may not reply on model performance.

# %%
#selecting the dependent and in dependent features
df=df.astype(int)
x=df.iloc[:,:11]
y=df['class']

# %%
#Standard Scaling
Scaler = StandardScaler()
x_scaled = Scaler.fit_transform(x)
x_scaled=pd.DataFrame(x_scaled)

# %% [markdown]
# Standard scalar is used to standardize the feature values upto a certain range.
# 
# **Result**
# 
# The data is scaled/standardize and outliers have been scaled too.

# %%
#split the data into train and test
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.1,random_state=42,shuffle=True)

# %% [markdown]
# ##Model Fitting

# %% [markdown]
# **Support Vector Machine**

# %%
#Support vector machine classifier training
svc = svm.SVC(gamma=0.1 ,C=1,random_state=42)
svc.fit(x_train,y_train)

#accuracy 
svc_acc=accuracy_score(y_test,svc.predict(x_test))

# %%
#accuracy on test data
svc_acc=accuracy_score(y_test,svc.predict(x_test))
print(svc_acc*100,"%")

# %%
#Confusion matrix
cm = confusion_matrix(y_test,svc.predict(x_test), labels=svc.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svc.classes_)
sns.set(rc = {'figure.figsize':(10,5)})
disp.plot()
plt.title('SVM')
plt.grid(False)
plt.show()

# %%
#classfication report
print("REPORT FOR SVM\n")
print(classification_report(y_test,svc.predict(x_test)))

#precision,recall and F-1 score 
svc_precision = metrics.precision_score(y_test,svc.predict(x_test), average='weighted', labels=np.unique(svc.predict(x_test)))
print("svc_precision", svc_precision)
svc_recall = metrics.recall_score(y_test,svc.predict(x_test), average='weighted', labels=np.unique(svc.predict(x_test)))
print("svc_recall:",svc_recall)
f1_svc = metrics.f1_score(y_test,svc.predict(x_test), average='weighted', labels=np.unique(svc.predict(x_test)))
print("svc_f1_score: ",f1_svc)

# %% [markdown]
# **Hyperparameter tunnning of decesion tree**
# 
# 
# 
# 

# %%
for max_d in range(1,21):

  model = DecisionTreeClassifier(max_depth=max_d, random_state=42)
  model.fit(x_train,y_train)
  print('The Training Accuracy for max_depth {} is:'.format(max_d), accuracy_score(y_train,model.predict(x_train)))
  print('The Validation Accuracy for max_depth {} is:'.format(max_d), accuracy_score(y_test,model.predict(x_test)))
  print('')

# %% [markdown]
# **Objective**:
# 
# To get best parameters for decision tree model to improve model performance

# %% [markdown]
# **Decision Tree**

# %%
#creating decision tree object
dcf= DecisionTreeClassifier(max_depth=10, random_state=42)

#train the decision tree model
dcf.fit(x_train,y_train)

#checking test accuracy
dcf_acc=accuracy_score(y_test,dcf.predict(x_test))
print(dcf_acc*100,"%")

# %%
#plot confusion matrix
cm = confusion_matrix(y_test,dcf.predict(x_test), labels=dcf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dcf.classes_)
sns.set(rc = {'figure.figsize':(10,5)})
disp.plot()
plt.title('decision tree')
plt.grid(False)
plt.show()

# %%
#classfication report
print("REPORT FOR Decesiontree\n")
print(classification_report(y_test,dcf.predict(x_test)))

#precision,recall and F-1 score for rf
dcf_precision = metrics.precision_score(y_test,dcf.predict(x_test), average='weighted', labels=np.unique(dcf.predict(x_test)))
print("dcf_precision:", dcf_precision)
dcf_recall = metrics.recall_score(y_test,dcf.predict(x_test), average='weighted', labels=np.unique(dcf.predict(x_test)))
print("dcf_recall:",dcf_recall)
dcf_f1 = metrics.f1_score(y_test,dcf.predict(x_test), average='weighted', labels=np.unique(dcf.predict(x_test)))
print("F1_score_dcf: ",dcf_f1)

# %% [markdown]
# **Random Forest Classifier**

# %%
#random forest object
rf=RandomForestClassifier()
rf.fit(x_train,y_train)

#accuracy on test data
rf_acc=accuracy_score(y_test,rf.predict(x_test))
print(rf_acc*100,"%")

# %%
#plot confusion matrix
cm = confusion_matrix(y_test,rf.predict(x_test), labels=rf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf.classes_)
sns.set(rc = {'figure.figsize':(10,5)})
disp.plot()
plt.title('Random forest')
plt.grid(False)
plt.show()

# %%
#classfication report
print("REPORT FOR RandomForestClassifier\n")
print(classification_report(y_test,rf.predict(x_test)))

#precision,recall and F-1 score for rf
rf_precision = metrics.precision_score(y_test,rf.predict(x_test), average='weighted', labels=np.unique(rf.predict(x_test)))
print("rf_precision:", rf_precision)
rf_recall = metrics.recall_score(y_test,rf.predict(x_test), average='weighted', labels=np.unique(rf.predict(x_test)))
print("rf_recall:",rf_recall)
rf_f1 = metrics.f1_score(y_test,rf.predict(x_test), average='weighted', labels=np.unique(rf.predict(x_test)))
print("F1_score_rf: ",rf_f1)

# %% [markdown]
# **Artificial neural networks**

# %%
#one hot encoding 
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#neural network object
model=Sequential()

#building neural network
model.add(Dense(10,activation='relu',input_dim=11,kernel_initializer='glorot_uniform'))
model.add(Dense(7,activation='relu',kernel_initializer='glorot_uniform'))
model.add(Dense(5,activation='relu'))
model.add(Dense(3,activation='tanh'))
model.add(Dense(4,activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#training neural network
history = model.fit(x_train,y_train,epochs=100,batch_size=100)


# %%
#check accuracy score for neural network model on test data.
loss,accuracy=model.evaluate(x_test, y_test)

# %%
#plotting accuracy and loss to observe how it is changing with increasing epochs.
plt.plot(history.history['accuracy'], label='Training accuracy', c='r')
plt.plot(history.history['loss'],  label='Training loss', c='g')
plt.legend()
plt.show()

# %% [markdown]
# ## RESULTS

# %%
#comparative accuracy barplot for all models
plt.title("Accuracy")
sns.barplot(x=["SVM","Decision Tree","Random forest","Artificial neural network"],y=[svc_acc,dcf_acc,rf_acc,accuracy],palette="colorblind")

# %% [markdown]
# The accuracy of RF can be seen outperforming other models. ANN can also been seen coming up with greater score.

# %%
#comparative precision barplot for all models
plt.title("precision")
sns.barplot(x=["SVM","Decision Tree","Random forest",],y=[svc_precision ,dcf_precision,rf_precision ],palette="colorblind")

# %% [markdown]
# **observation**:
# 
# Precision of Random forest is highest. SVM has least precision score which means it is lagging predicitng positive responses.
# 

# %%
#comparative recall barplot for all models
plt.title("recall")
sns.barplot(x=["SVM","Decision Tree","Random forest",],y=[svc_recall ,dcf_recall,rf_recall ],palette="colorblind")

# %% [markdown]
# **observation**:
# 
# recall of Random forest and decision tree are highest  whereas SVM is again lagging in terms of prediciting class correclty.
# 

# %%
#comparative f1-score barplot for all models
plt.title("F1-score")
sns.barplot(x=["SVM","Decision Tree","Random forest",],y=[f1_svc ,dcf_f1,rf_f1],palette="colorblind")

# %% [markdown]
# **observation**:
# 
# F1-score of Random forest i greatest followed with decision tree. SVM is still least active in predicting accurately.

# %% [markdown]
# ### **Result and Observation**:
# 
# 1)Random Forest Classifer is the best model for this classification with accuracy around 76 percent.
# 
# 2)Neural Network Model came out to be as the second best model for this classification with accuracy 75 percent.
# 
# As the batches are increased in neural network the accuracy increases and log loss decreases accordingly. Hence, it is concluded that efficiency of neural network model can be increased by working on its layers and epoch and batch variations.
# 
# Hence, for such dataset or related datasets, Random Forest and Neural Network seem to be more active and accurate.

# %%



