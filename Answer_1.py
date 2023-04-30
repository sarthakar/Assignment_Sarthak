import numpy as np
data = np.array([[1,1],[-1,-1],[0,0.5],[0.1,0.5],[0.2,0.2],[0.9,0.5]]) 
cc = 0
y=[1,-1,-1,-1,1,1]
w = [1,1] 
w = np.append(w,1) 
print(w)
print("starting..")
steps = 0
while (cc != len(data)):
  steps = steps +1
  for sample in range(len(data)):
    x = np.append(data[sample,0:2],1)
    if y[sample]==1: 
      if np.dot(np.transpose(w),x)>=0: 
        cc=cc+1    
      else: 
        w=w+x
    else: 
      if np.dot(np.transpose(w),x)<0: 
        cc=cc+1  
      else: 
        w=w-x
  if(cc != len(data)):
    cc=0
print("steps: "+ str(steps-1))
print(w)
