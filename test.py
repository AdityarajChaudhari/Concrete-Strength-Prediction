import pandas as pd
import pickle

model = pickle.load(open('./ModelSaving/model.pkl','rb'))
scalar = pickle.load(open('./DataScaler/Scaler.pkl','rb'))

s = scalar.transform([[475,0,0,228,0,932,574,0,0,0,0,0,0]])
print(s)

a = model.predict(s)
print(a)