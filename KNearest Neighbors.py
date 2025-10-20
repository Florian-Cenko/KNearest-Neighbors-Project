import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import  train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix

# 1. Get the Data and do Analysis
df = pd.read_csv("KNN_Project_Data")
print(df.head())

sns.pairplot(df,hue='TARGET CLASS')
plt.show()

# 2. Standardize the Variables
scaler = StandardScaler()
# a) Fit scaler to the features.
scaler.fit(df.drop('TARGET CLASS',axis=1))

# b) Use the .transform() method to transform the features to a scaled version.
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))

# c) Convert the scaled features to a dataframe and check the head of this dataframe to make sure the scaling worked.
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
print(df_feat.head())

# 2. Using of Train and Test Split
X = df_feat
y = df['TARGET CLASS']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)

# 3. Now we can use KNN
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)

# 4. We can predict and Evaluate the Model
pred = knn.predict(X_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

# 5. Now is the time to choose a K Value with Elbow Method help
error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6))
# For better figure
plt.plot(range(1,40),error_rate,color='blue',linestyle='--',marker='o',markerfacecolor='red',markersize=10)
plt.title('Error Rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()

# 6. Retrain your model with the best K value (up to you to decide what you want) and re-do the classification report and the confusion matrix.
knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(X_train,y_train)

pred = knn.predict(X_test)
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))
