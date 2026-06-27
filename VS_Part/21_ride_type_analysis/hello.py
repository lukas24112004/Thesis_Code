# import pandas as pd
# 
# df = pd.read_excel(r"C:\Users\Gebruiker\Desktop\Econ\Econometrics_dataset.xlsx")
# df['Date'] = pd.to_datetime(df['Date'])
# df = df.sort_values('Date').reset_index(drop=True)
# 
# import numpy as np
# df['ln_LMT'] = np.log(df['LMT'])
# df['ln_RTX'] = np.log(df['RTX'])
# df['ln_NOC'] = np.log(df['NOC'])
# df['ln_GD']  = np.log(df['GD'])
# df['ln_SPY'] = np.log(df['SPY'])
# 
# df[['Date','ln_LMT','ln_RTX','ln_NOC','ln_GD','ln_SPY']].to_excel(
#     r"C:\Users\Gebruiker\Desktop\Econ\log_prices.xlsx", index=False
# )
# print("Done")

import pandas as pd

df = pd.read_excel(r"C:\Users\Gebruiker\Desktop\Econ\log_prices.xlsx")
df.to_csv(r"C:\Users\Gebruiker\Desktop\Econ\log_prices.csv", index=False)
print("Done")