#!/usr/bin/env python
# coding: utf-8

# # Problem 1

# In[313]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from math import sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from random import sample

data = pd.read_csv("data.csv")
df = pd.DataFrame(data)
df = df.drop(['id'],axis =1)
train_data = np.array(df.drop(['diagnosis'],axis =1))
train_data = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(train_data)
train_data = StandardScaler().fit_transform(train_data)
test = np.array(df['diagnosis'])

def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)
        

def k_means(data,k,itera,centroids):
    
    i,j = sample(range(len(data)),2)
    centroids[0] = data[i]
    centroids[1] = data[j]
    
    for i in range(itera):
        classifications = {}
    
        for i in range(k):
            classifications[i] = []

        for row in data:
            distance0 = euclidean_distance(row,centroids[0])
            distance1 = euclidean_distance(row,centroids[1])
            if(distance0<distance1):
                classifications[0].append(row)
            else:
                classifications[1].append(row)
        
        prev_centroids = dict(centroids)
        
        for classification in classifications:
                centroids[classification] = np.average(classifications[classification],axis=0)
       
        flag = False
        for c in centroids:
                original_centroid = prev_centroids[c]
                current_centroid = centroids[c]
                if(original_centroid.all() == current_centroid.all()):
                    flag = True
        if flag:
            break

def predict(row,centroids):
    distance0 = euclidean_distance(row,centroids[0])
    distance1 = euclidean_distance(row,centroids[1])
    if(distance0<distance1):
        return 0
    else:
        return 1


# In[314]:


centroids = {}
k_means(train_data,2,200,centroids)
correct1 = 0
correct2 = 0
test_trans1 = []
test_trans2 = []
for i in range(len(test)):
    if (test[i] == 'M'):
        test_trans1.append(0)
        test_trans2.append(1)
    elif (test[i] == 'B'):
        test_trans1.append(1)
        test_trans2.append(0)
        
for i in range(len(train_data)):

    predict_me = train_data[i]
    prediction = predict(predict_me,centroids)
    #print(prediction,test[i])
    if prediction == test_trans1[i]:
        correct1 += 1
    elif prediction == test_trans2[i]:
        correct2 += 1


# Note: Since the cluster 0 can be either 'B' or 'M' checking both the clusters

# In[315]:


if(correct1>correct2):
    print("accuracy of correct prediction",correct1/len(train_data)*100)
    print("false prediction",correct2/len(train_data)*100)
else:
    print("accuracy of correct prediction",correct2/len(train_data)*100)
    print("false prediction",correct1/len(train_data)*100)


# As we are choosing centroid randomly  the accuracy varies depending on the centroids choosen

# In[312]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

X_train, X_test, y_train, y_test = train_test_split(train_data, test, random_state=0)
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
print("Test score: {:.2f}".format(logreg.score(X_test, y_test)))


# A supervised algorithm predicts more accurately than our model 

# # Problem 3

# In[26]:


import numpy as np
import pandas as pd
import nltk
import math

boys = pd.read_csv("boy_names.csv")
girls = pd.read_csv("girl_names.csv")
test = pd.read_csv("test_names.csv")

boys = pd.DataFrame(boys)
girls = pd.DataFrame(girls)
test = pd.DataFrame(test)

bnames = np.array(boys['x'])
gnames = np.array(girls['x'])
testnames = np.array(test['x'])

bdict = []
f_countb = 0
for i in range(len(bnames)-1):
    features = nltk.ngrams(bnames[i], 3)
    features = list(features)
    bdict.append(features)
    f_countb+= len(features)

gdict = []
f_countg = 0
for i in range(len(gnames)-1):
    features = nltk.ngrams(gnames[i], 3)
    features = list(features)
    gdict.append(features)
    f_countg+= len(features)


def nb_classifier(name):
    
    features = nltk.ngrams(name, 3)
    features = list(features)
    prob_mlb = 1
    prob_mlg = 1
    for f in features:
        count = 1
        for i in range(len(bdict)-1):
            if f in bdict[i]:
                count+= 1
        #Computing P(X/Y)        
        prob_mlb *= count/(f_countb+f_countg)
        
        count = 1
        for i in range(len(gdict)-1):
            if f in gdict[i]:
                count+= 1
        #Computing P(X/Y) 
        prob_mlg *= count/(f_countb+f_countg)
        
    #Computing P(Y)
    prob_bf = f_countb/(f_countb+f_countg)
    prob_gf = f_countg/(f_countb+f_countg)
    
    #Computing log P(Y=1/X)/P(Y=-1/X)
    logp = math.log((prob_mlg*prob_gf)/(prob_mlb*prob_bf))
    
    # P(Y=1/X) > P(Y=-1/X) so returning +1 for girl
    if(logp > 0):
        return "+1"
    # P(Y=1/X) < P(Y=-1/X) so returning -1 for boy
    elif(logp < 0):
        return "-1"
    else:
        return "0"

    
def predict(names):
    dct = {'x':names,'classification':[]}
    count = 0
    count1 = 0
    for i in names:
        dct['classification'].append(nb_classifier(i))
    return dct


# In[30]:


data = predict(testnames)
df = pd.DataFrame(data, columns= ['x', 'classification'])
df.to_csv('results.csv')
print(df)


# # Problem 2

# In[90]:


f1 = lambda x,y: (x-2)**2 + (y-3)**2
f2 = lambda x,y: (1-(y-3))**2 + 20*(((x+3)-(y-3)**2)**2)

d1x = lambda x,y: 2*(x-2)
d1y = lambda x,y: 2*(y-3)

d2x = lambda x,y: 40*(x+3) - 40*((y-3)**2)
d2y = lambda x,y: 2*(y-4) + 80*((y-3)**3) - 80*(x+3)*(y-3)


def grad_descent(func,dx,dy,lr,itera,threshold):
    x=0
    y=0
    fhist = []
    fhist.append(func(x,y))   
    for i in range(itera):
        newx = x - lr*dx(x,y)
        newy = y - lr*dy(x,y)
        x = newx
        y = newy
        fhist.append(func(x,y))
        if abs(fhist[-1])<threshold:
            print("Found optimal solution x,y" )
            print(x,y)
            return fhist
        if fhist[-2]*1000<fhist[-1]:
            print("Diverged too far ending")
            return fhist
    print("Solution after max iterations")
    return fhist
            


# In[91]:


g1 = grad_descent(f1,d1x,d1y,0.5,10,0.0000000001)
print(g1[-1])

g2 = grad_descent(f2,d2x,d2y,0.5,100,0.0000000001)
print(g2[-1])


#0.00210985161208 learning rate by trail and error
g2t = grad_descent(f2,d2x,d2y,0.00210985,100,0.0000000001)
print(g2t[-1])


# In[92]:


d1x2 = 2
d1y2 = lambda x,y: 2
d2x2 = 40
d2y2 = lambda x,y: 2 + 240*((y-3)**2) - 80*(x+3)


def newton_method(func,dx,dy,dx2,dy2,x0,y0,itera,epsilon,gamma = 1):
    x = x0
    y = y0
    fhist = []
    fhist.append(func(x,y))    
    for i in range(itera):
        if abs(fhist[-1])<epsilon:
                print("Optimal solution found")
                print(x,y)
                return fhist
        if(dx2 != 0 and dy2(x,y) != 0):
            newx = x - gamma*(dx(x,y)/dx2)
            newy = y - gamma*(dy(x,y)/dy2(x,y))
            x = newx
            y = newy
            fhist.append(func(x,y))
    print("After max iterations")
    return fhist
    


# In[93]:


n1 = newton_method(f1,d1x,d1y,d1x2,d1y2,0,0,10,0.0000000001)
print(n1[-1])

n2 = newton_method(f2,d2x,d2y,d2x2,d2y2,0,0,100,0.0000000001,0.1)
print(n2[-1])


# # Problem 4

# 1. The equality doesnt hold when the features are dependent on each other the left side value will be larger than right side if any of the feature has zero probability consider the example of spam filtering if the spam has any new word that has been not classified as spam
#     The right side value will be greater if the joint probability of features is less than individual features we can take the same example of spam filtering if car and medicine appeared in an email the joint probability of car and medicine in a spam email is very low but if we take individual probabilities the word car in spam mail can be high and the word medicine is also high in this case the left side dominates
# 
# 2. P(x/y=c) = P(x,y=c)/P(y=c)
#    P(x,y) can be further expanded by chain rule
#    P(x1,x2,x3.....xD,y=c) = P(xd/xd-1,.....x3,x2,x1,y=c)*P(xd-1,.....x3,x2,x1,y=c)
#                             P(xd/xd-1,.....x3,x2,x1,y=c)*P(xd-1/xd-2.....x3,x1,y=c)*P(xd-2.....x2,x1,y=c)
#                             P(xd/xd-1,.....x3,x2,x1,y=c)*P(xd-1/xd-2.....x3,x1,y=c).......*P(x1/y=c)*P(y=c)
# 3. For the fixed features D by our assumption of independent features in naive bayes we can compute all the the probabilities easily with less training data and predict accuarately where as the full model needs lots of training data to compute all the joint probabilities and doesnt perform very well with less training data. So naive bayes gives the lower test set error
#  

# In[ ]:




