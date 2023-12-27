import numpy as np 
x = np.array(([2,9],[1,5],[3,6]),dtype=float) 
y = np.array(([92],[86],[89]),dtype=float) 
x = x/np.amax(x,axis=0)
y = y/100 

def sigmoid(x): 
    return 1/(1+np.exp(-x)) 
def sigmoid_grad(x): 
    return x*(1-x) 

output_neurons=1
input_neurons=2
hidden_neurons=3

wh = np.random.uniform(size=(input_neurons,hidden_neurons)) 
wout = np.random.uniform(size=(hidden_neurons,output_neurons))
bh = np.random.uniform(size=(1,hidden_neurons)) 
bout = np.random.uniform(size=(1,output_neurons)) 

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
