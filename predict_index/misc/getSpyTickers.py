import pandas as pd
import numpy as np

table=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
df = table[0]
dfNP = df.to_numpy()
dfNP = np.transpose(dfNP)
print(dfNP[0])
