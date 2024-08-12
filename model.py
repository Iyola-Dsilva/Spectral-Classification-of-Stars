#the ML Model
import sklearn as sk
from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
stars = pd.read_csv('stars.csv')
stars["Star color"] = stars["Star color"].astype('category') 
stars["color"] = stars["Star color"].cat.codes #converting the string to category integer codes
c="Star color"
st="Star type"
cl="color"
target="Spectral Class" #target variable
X = np.array(stars.drop([target,c,st,cl],axis=1)) #the dataset has 7 attributes here drop 4 including target
Y = np.array(stars[target])
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2,random_state=10)
model = svm.SVC(kernel="linear").fit(x_train,y_train)
pickle.dump(model,open('star.pkl','wb')) #store the model as .pkl 

