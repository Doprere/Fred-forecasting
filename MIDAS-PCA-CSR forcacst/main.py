
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from numpy.linalg import svd
from itertools import combinations
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)





# Step 1: Data Loading

fredM_data = pd.read_csv('C:/Users/dopre/OneDrive/桌面/碩1/上學期/經濟預測/fredM_data.csv', index_col=0, parse_dates=True)
fredQ_data = pd.read_csv('C:/Users/dopre/OneDrive/桌面/碩1/上學期/經濟預測/fredQ_data.csv', index_col=0, parse_dates=True)

# Select quarterly CPI and compute percentage change
CPI = fredQ_data.loc[:, 'CPIAUCSL'].pct_change(4)
    
# Normalize monthly data
fredM_data = fredM_data.dropna(axis=1)
fredM_data = (fredM_data - fredM_data.mean(axis=0)) / fredM_data.std(axis=0)
fredM_data.mean(axis=0)
fredM_data.std(axis=0)

#Select variables with high correlation to CPI
X_q = fredM_data.loc[CPI.dropna().index, :]
correlations = X_q.apply(lambda col: col.corr(CPI.dropna()))
fred_data_subset = fredM_data.loc[:, np.abs(correlations) > 0.4]



# In[112]:


# Step 2: 權重
def weight_function(theta1, theta2, p):
    k = np.arange(0, p)
    weights = np.exp(theta1 * k + theta2 * k**2)
    normalized_weights = weights / np.sum(weights)
    return normalized_weights


# In[113]:


def lag_v(x,lag,max_lag): 
    n = x.shape[0]
    y = x[max_lag-lag:n-lag,:]
    return y


# In[114]:


# Step 3: NLS 
def nls_objective(params, y_t, x_t, p):
    theta1, theta2, beta, alpha = params
    weights = weight_function(theta1, theta2, p)
    y_hat = np.zeros(len(y_t))

    for tt in range(np.int32(p/3), len(y_t)):
        lagged_values = x_t[tt * 3 - p:tt * 3, :][::-1]
        weighted_sum = weights[:len(lagged_values)] @ lagged_values
        y_hat[tt] = alpha + beta * weighted_sum[0]
    residuals = y_t - y_hat
    rss = np.sum(residuals**2)
    return rss


# In[115]:


#Step 4: 建構不同的變數組合  CSR?
combinations_list = list(combinations(fred_data_subset.columns, 3))
print(combinations_list)
pca1 = pd.DataFrame(index = fred_data_subset.index ,columns=list(range(len(combinations_list))))
for jj in range(len(combinations_list)):
    svd_results = svd(fred_data_subset[list(combinations_list[jj])])
    # 只擷取第一個pca組成
    pca1.iloc[:,jj] = svd_results[0][:,0]
print(pca1)
p = 12 
y_m = CPI.dropna().resample('ME').bfill()
yhat_m = pd.DataFrame(np.zeros((len(y_m),len(combinations_list))),index = y_m.index)





# Step 5: Nowcasting CPI
result = [None]*2
rss = np.zeros(2)
for ii in range(len(combinations_list)):
    nn = 0
    X = pca1.iloc[:,ii:ii+1] 
    name = X.columns[0]
    y_t = np.array(CPI.dropna()) 
    x_t = np.array(X.iloc[1:,:]) 
    while nn<2: 
        initial_params =np.random.randn(4) 
        res = minimize(nls_objective, initial_params, args=(y_t, x_t, p), method='L-BFGS-B')
        if res.success == True: # 收斂才會把結果存下來
            result[nn] = res
            rss[nn] = res.fun
            nn = nn+1
            
    ex_theta1, ex_theta2, ex_beta, ex_alpha = result[np.argmin(rss)].x
    ex_weights = weight_function(ex_theta1, ex_theta2, p)

    X_lags = np.hstack([lag_v(x_t, lag, p-1) for lag in range(0,p)])
    X_lags = pd.DataFrame(X_lags,index = pd.date_range(start=CPI.dropna().index[0],periods=len(X_lags),freq='ME'))
    yhat_m.iloc[:,ii] = pd.DataFrame(ex_beta*(X_lags@ex_weights)+ex_alpha)




yhat_mean = np.mean(yhat_m,axis=1)
print(yhat_mean)




# Step 6: Visualization
combined = pd.concat((y_m, yhat_mean), axis=1)
combined.columns = ['Actual','Predicted_Mean']
combined.loc['2022-1-1':,:]
combined.plot(figsize=(12, 6), linewidth=2)
plt.title('CPI Nowcasting: Actual vs Predicted(mean)')
plt.xlabel('Year')
plt.ylabel('CPI Change Rate')
plt.legend()
plt.show()


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 將 NaN 去除（因為前期可能無法預測）
actual = combined['Actual'].dropna()
predicted = combined['Predicted_Mean'].loc[actual.index]

rmse = np.sqrt(mean_squared_error(actual, predicted))
mae = mean_absolute_error(actual, predicted)
r2 = r2_score(actual, predicted)

print(f'RMSE: {rmse:.4f}')
print(f'MAE : {mae:.4f}')
print(f'R²   : {r2:.4f}')





residual = actual - predicted
plt.figure(figsize=(10, 4))
plt.plot(residual)
plt.title('Prediction Residual (Actual - Predicted)')
plt.xlabel('Date')
plt.ylabel('Residual')
plt.axhline(0, color='black', linestyle='--')
plt.show()