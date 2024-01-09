import numpy as np 
x = np.array(([2,9],[1,5],[3,6]),dtype=float) 
y = np.array(([92],[86],[89]),dtype=float) 
x = x/np.amax(x,axis=0)
y = y/100 

def sigmoid(x): 
    return 1/(1+np.exp(-x)) 
def sigmoid_grad(x): 
    return x*(1-x) 

wh = np.random.uniform(size=(2,3)) 
wout = np.random.uniform(size=(3,1))
bh = np.random.uniform(size=(1,3)) 
bout = np.random.uniform(size=(1,1)) 

for i in range(1000):  
    h_ip = np.dot(x,wh)+bh 
    h_act = sigmoid(h_ip) 
    o_ip = np.dot(h_act,wout)+bout
    o_act = sigmoid(o_ip)
    hiddengrad = sigmoid_grad(h_act)
    outgrad = sigmoid_grad(o_act)
    hidden =(y-h_act)*hiddengrad 
    output = (y-o_act)*outgrad
    wh += o_act.T.dot(hidden)*0.2
    wout += h_act.T.dot(output)*0.2 
    
print("Normalized Input:\n"+str(x))
print("Actual Output:\n"+str(y))
print("Predicted output:\n",o_act)
