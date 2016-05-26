import pandas as pd


# Read submissions
rf_sbm = pd.read_csv('submissions/rf_extended.csv')
xgb_sbm = pd.read_csv('submissions/xgb_extended.csv')

# Mean
df_mean = rf_sbm.copy()
df_mean['count'] = 0.5 * rf_sbm['count'] + 0.5 * xgb_sbm['count']
df_mean.to_csv('submissions/rf_xgb_extended_mean.csv', index=False)

# Weighted mean
df_wighted = rf_sbm.copy()
rf_w = 1.0 / 0.313
xgb_w = 1.0 / 0.277
s = rf_w + xgb_w
df_wighted['count'] = rf_w / s * rf_sbm['count'] + xgb_w / s * xgb_sbm['count']
df_wighted.to_csv('submissions/rf_xgb_extended_wighted.csv', index=False)
