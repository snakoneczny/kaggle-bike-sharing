import pandas as pd


# Read submissions
rf_sbm = pd.read_csv('submissions/rf_extended.csv')
xgb_sbm = pd.read_csv('submissions/xgb_extended.csv')
keras_sbm = pd.read_csv('submissions/keras_neuralnet.csv')

# Mean
df_mean = rf_sbm.copy()
df_mean['count'] = (rf_sbm['count'] + xgb_sbm['count'] + keras_sbm['count']) / 3.0
df_mean.to_csv('submissions/rf_xgb_keras_mean.csv', index=False)

# Weighted mean
df_wighted = rf_sbm.copy()
rf_w = 1.0 / 0.313
xgb_w = 1.0 / 0.277
keras_w = 1.0 / 0.284
s = rf_w + xgb_w + keras_w
df_wighted['count'] = rf_w / s * rf_sbm['count'] + xgb_w / s * xgb_sbm['count'] + keras_w / s * keras_sbm['count']
df_wighted.to_csv('submissions/rf_xgb_keras_wighted.csv', index=False)
