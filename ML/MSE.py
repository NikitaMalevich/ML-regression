import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf

df = pd.read_csv('advertising.txt',delimiter = "\t")
df['ln_Sales'] = df['Sales'].apply(np.log)
df['ln_TV'] = df['TV'].apply(np.log)

print(df)

##fig,axs = plt.subplots(1,3, sharey =True)
##df.plot(kind = 'scatter', x = 'TV',y='Sales',ax=axs[0],figsize=(15,5))
##df.plot(kind = 'scatter', x = 'Radio',y='Sales',ax=axs[1])
##df.plot(kind = 'scatter', x = 'Newspaper',y='Sales',ax=axs[2])
##plt.show()

#create a fitted model in one line
lm = smf.ols(formula='ln_Sales ~ ln_TV',data=df)
res_TV = lm.fit()

#print the coefficients
print('\n',res_TV.params)
preds = res_TV.predict(df['ln_TV'])
print(preds)

##b_0  = res_TV.params['Intercept']
##b_1 = res_TV.params['TV']
##
##x = np.linspace(1,300,300)
##y = b_0 + b_1*x

##plt.plot(x,y,'r')
##plt.scatter(df['TV'],df['Sales'])
##plt.xlabel('TV')
##plt.ylabel('Sales')
##plt.title('TV sales')

##x_new = pd.DataFrame({'TV':[50]})
##preds = res_TV.predict(x_new)
##print('\n',preds.iloc[0])

df.plot(x='TV',y='Sales',kind='scatter')
##plt.plot(x_new,preds)
plt.show()



#______Ln data______
##ln_lm = smf.ols(formula='ln_Sales ~ ln_TV',data=df)
##res_ln_TV = ln_lm.fit()
##print('\n',res_ln_TV.params)

##x_new_ln = np.linspace(1,300,300)
##preds_ln = res_TV.predict(df['ln_TV'])
##print(preds_ln)

##b_0_ln  = res_ln_TV.params['Intercept']
##b_1_ln = res_ln_TV.params['ln_TV']
##
##x_ln = np.log(np.linspace(1,300,300))
##y_ln = b_0_ln + b_1_ln*x
##
##plt.plot(x_ln,y_ln,'r')
##plt.scatter(df['ln_TV'],df['ln_Sales'])
##plt.xlabel('ln_TV')
##plt.ylabel('ln_Sales')
##plt.title('logarifm TV sales')
##





