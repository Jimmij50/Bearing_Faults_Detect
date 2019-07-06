#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 22:51:55 2019

@author: jim
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 22:49:22 2019

@author: jim
"""
import pandas as pd
import numpy as np
My='submmit_MLP.csv'
#My='submmit_feature.csv'
#My='submmit.csv'
Xu='xujiansubmmit.csv'
mm=pd.read_csv(My)
xx=pd.read_csv(Xu)
mm=np.array(mm)
xx=np.array(xx)
mmm=mm[:,1:]
xxx=xx[:,1:]
print(mmm.shape)
mmm=np.array(mmm)
xxx=np.array(xxx)
aaa=(mmm==xxx)
c=0
index=0
for i in aaa:
    index=index+1
    if i==True:
        c=c+1;
    else:
        print(index)
print(c,index)
print(My)
print('accuracy-{}%'.format((c/index)*100))
        
