import pandas as pd
import numpy as np

df1 = pd.DataFrame(np.random.randn(4,4),columns=list('abcd'))
df2 = pd.DataFrame(np.random.randn(4,4),columns=list('dbca'))

print(df1)
print(df2)
df3 = pd.concat([df1,df2], ignore_index=True)
print(df3)