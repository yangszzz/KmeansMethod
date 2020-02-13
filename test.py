import numpy as np
d = np.array(
    [
        [1,2,3,4],
        [2,3,4,0],
        [1,2,3,1],
    ]
)
print(d[np.where(d[:,-1]==0)])