import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf

df = pd.read_csv('advertising.txt',delimiter = "\t")
print(df.head(),'\n')

# create a fitted model in one line
lm_TV = smf.ols(formula='Sales ~ TV', data=df).fit()

# print the coefficients
##print(lm.params)
X_new = pd.DataFrame({'TV': [df.TV.min(),df.TV.max()]})

preds = lm_TV.predict(X_new)
print('\n',preds)

df.plot(kind='scatter', x='TV', y='Sales')
plt.plot(X_new, preds, c='red', linewidth=2)
plt.show()


