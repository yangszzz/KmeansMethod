import pandas as pd

import numpy as np

# a= np.array([1,2])
# b= np.array([[3,4]])
# # c = np.hstack((a,b))
# a = a[None]
# print(a)

a  = pd.DataFrame(
    [[1,2,3],
    [4,5,6],
    [7,8,9]]
)
b  = pd.DataFrame(
    [[1,2,3],
    [4,5,6],
    [7,8,9]]
)
c = pd.concat([a,b])
print(c)
print(a['2'].shape)
