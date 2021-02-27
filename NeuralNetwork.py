
import random
import sys
import time
import numpy as np

def read_input_file():

    if len(sys.argv)>1:
        training_x_filename=sys.argv[1]
        training_labels_filename=sys.argv[2]
        testing_x_filename=sys.argv[3]
    else:
        training_x_filename="train_image.csv"
    
        training_labels_filename="train_label.csv"
     
    
        testing_x_filename="test_image.csv"
    
    return training_x_filename,training_labels_filename,testing_x_filename
   
def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e   
    
def format_input(training_x_filename,training_labels_filename,testing_x_filename):
   
    values = np.genfromtxt(training_x_filename,delimiter=',')

    training_inputs = [np.reshape((x/255), (784, 1)) for x in values]
   
    values = np.genfromtxt(training_labels_filename,delimiter=',')
    print(int(values[0]))
    training_results = [vectorized_result(int(y)) for y in values]
    training_data = zip(training_inputs, training_results)
    values = np.genfromtxt(testing_x_filename,delimiter=',')
    test_inputs = [np.reshape((x/255), (784, 1)) for x in values]
    
    return training_data,test_inputs
    
        
def write_output(predictions):
    with open("test_predictions.csv",'w') as f:
        for i in predictions:
            f.write(str(i))
            f.write("\n")
    f.close()
    return
    
def activation_sigmoid(z):
    
    return 1.0/(1.0+np.exp(-z))

def activation_softmax(z):

    return np.exp(z)/np.sum(np.exp(z), axis=0, keepdims=True)

def activation_sigmoid_derivative(z):
    
    return activation_sigmoid(z)*(1-activation_sigmoid(z))   

    
def cross_entropy_cost_forward(y_pred, y_true):

    logistic_loss=-y_true*np.log(y_pred)-(1-y_true)*np.log(1-y_pred)
    return np.sum(np.nan_to_num(logistic_loss))

    
def loss_backward(y_pred, y_true):

    return (y_pred-y_true)
    
def batch_sync(batch, derivative):

    return [i+j for i, j in zip(batch, derivative)]
    


class Network(object):

    def __init__(self, layer_count,neuron_count):
        
        self.num_layers = layer_count
        self.sizes = neuron_count
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        



    def forward_pass(self, input):
  
        activations = [input]  
        before_activation=[]
        count=0
        for b, w in zip(self.biases, self.weights):
            count=count+1
            
            z=np.dot(w, input)+b
            before_activation.append(z)
            if count<3:
                input = activation_sigmoid(z)
            else:
                input = activation_softmax(z)
            activations.append(input)
        return input,activations,before_activation
        
    def forward_pass_2(self, input):
      
        count=0
        for b, w in zip(self.biases, self.weights):
            count=count+1
            
            z=np.dot(w, input)+b
            if count<3:
                input = activation_sigmoid(z)
            else:
                input = activation_softmax(z)
           
           
        return input
            
   
       
    def backpropagation(self, x, y,activations,before_activation):
    
        derivative_w = [np.zeros(w.shape) for w in self.weights]
        derivative_b = [np.zeros(b.shape) for b in self.biases]
        
             
        for l in range(1, self.num_layers):
            if l ==1:
                loss = loss_backward(activations[-l], y)
                
                derivative_w[-l] = np.dot(loss, activations[-l-1].T)
                derivative_b[-l] = loss
            else:
                loss = np.dot(self.weights[-l+1].T, loss) * activation_sigmoid_derivative(before_activation[-l])
                derivative_w[-l] = np.dot(loss, activations[-l-1].T)
                derivative_b[-l] = loss
            
        return (derivative_b, derivative_w)


    def train(self, training_data,evaluation_data, epochs, batch_size,lr):
 
        training_data = list(training_data)
        n = len(training_data)
        

        predictions = []
       
        for epoch in range(epochs):
        
            randomize = np.arange(len(training_data))
            random.shuffle(training_data)
            np.random.shuffle(randomize)
            
            
            batches = [training_data[k:k+batch_size] for k in range(0, n, batch_size)]
            
            
            for batch in batches:
            
            
                batch_w = [np.zeros(w.shape) for w in self.weights]
                batch_b = [np.zeros(b.shape) for b in self.biases]
                
                
                for x, y in batch:
                
                    y_prediction,activations,before_activation=self.forward_pass(x)
                    
                    derivative_b, derivative_w = self.backpropagation(x, y,activations,before_activation)
                    
                    
                    z=np.array(derivative_w)+np.array(batch_w)
                    
                    
                    batch_b=batch_sync(batch_b,derivative_b)
                    batch_w=batch_sync(batch_w,derivative_w)
                    
                   
                self.weights = [w-(lr/len(batch))*dw for w, dw in zip(self.weights, batch_w)]
                self.biases  = [b-(lr/len(batch))*db for b, db in zip(self.biases, batch_b)]
            print("Epoch %s complete" % epoch)

           
       
        for x in evaluation_data:
            predictions.append(np.argmax(self.forward_pass_2(x)))
      
        
            
        return predictions  
 

start=time.time()

training_x_filename,training_labels_filename,testing_x_filename=read_input_file()
training_data,test_inputs=format_input(training_x_filename,training_labels_filename,testing_x_filename)

training_data = list(training_data)
test_data = list(test_inputs)
net = Network(4,[784, 200,32, 10])
predictions= net.train(training_data, test_data,15, 10, 0.2)
print(time.time()-start)
write_output(predictions)
print(time.time()-start)