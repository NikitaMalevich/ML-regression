import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn import datasets,linear_model,metrics
from sklearn.model_selection import train_test_split

# pd.set_option('display.max_columns',None)
sns.set(style="darkgrid")

data = pd.read_csv('E:\\Python\\ML\\house_prices\\train.csv')
data = data.drop(columns=['Id'])

y = data['SalePrice']
X = data.drop(columns=['SalePrice'])

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=10)
# sns.displot(np.log(y_test))

numeric_data = X_train.select_dtypes([np.number])
numeric_features = numeric_data.columns

X_train = X_train[numeric_features]
X_test = X_test[numeric_features]

# fig,axs = plt.subplots(figsize=(16,5),ncols=3)
# for i,feature in enumerate(['GrLivArea','GarageArea','TotalBsmtSF']):
#     axs[i].scatter(X_train[feature],y_train,alpha=0.2)
#     axs[i].set_xlabel(feature)
#     axs[i].set_ylabel('SalePrice')
# plt.tight_layout()

# print(len(X_train),len(y_train))
#
# model = Ridge() #initialization model
# model.fit(X_train,y_train) #teach model
# y_predict = model.predict(X_test) #predict Xtest
# y_train_predict = model.predict(X_train)#predict Xtrain
#
# print(f'Test RMSE = {mean_squared_error(y_test,y_predict,squared=False)}')
# print(f'Train RMSE = {mean_squared_error(y_train,y_train_predict,squared=False)}')



# plt.show()