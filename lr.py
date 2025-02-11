import numpy as np
import argparse


def extract_data(data):   #pull out the data
    with open(data,'r') as f_in:
        content=f_in.readlines()
        extracted_2dlst=[np.array(line.strip().split('\t')) for line in content]

    return extracted_2dlst

def process_data(data_lst):
    X = [arr[1:].astype(float) for arr in data_lst]  # Clean and convert X
    Y = [arr[0].astype(float) for arr in data_lst]   # Clean and convert Y
    return X, Y



def sigmoid(x : np.ndarray):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (np.ndarray): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(x)
    return e / (1 + e)   

def train_data_header(input_argument,num_epoch,learning_rate):
    data_str=extract_data(input_argument)
    train_processed=process_data(data_str)
    X=np.reshape(train_processed[0],(len(data_str),len(data_str[0])-1)) #dim: (N,D)
    Y=np.reshape(train_processed[1],(len(data_str),1)) # dim: (N,)
    theta=np.zeros((len(data_str[0])-1,1))  #dim: (D,)
    #print(X[0].shape)
    #print(theta.T.shape)
    #print(theta.T @ X[0] -Y[0])
    return train(theta,X,Y,num_epoch,learning_rate)


# X shape (N, D) where N is num of examples
# theta shape (D,) where D is feature dim
# Y shape (N,)
def train(theta, X, Y, num_epoch, learning_rate):
    bias = 0
    for i in range(num_epoch):
        for j in range(len(Y)):
            # Calculate the predicted value including the bias
            prediction = sigmoid(theta.T @ X[j] + bias)
            
            
            weight_gradient = X[j] * (prediction - Y[j])
            weight_gradient = np.reshape(weight_gradient, (theta.shape))
            
            
            bias_gradient = prediction - Y[j]  

            # Update theta and bias
            theta -= weight_gradient * learning_rate
            bias -= bias_gradient * learning_rate

            # Print shapes for debugging
            # print("Iteration:", i, "Example:", j)
            # print("Weight Gradient Shape:", weight_gradient.shape)

    return (theta, bias)






def predict_data_header(input_argument,parameters,output_file):
    data_str=extract_data(input_argument)
    train_processed=process_data(data_str)
    X=np.reshape(train_processed[0],(len(data_str),len(data_str[0])-1)) #dim: (N,D)
    theta=parameters[0]
    bias=parameters[1]
    return predict(theta,bias,X,output_file)

def predict(theta,bias,X,output_file):
    threshold=0.5
    prediction=[]
    for i in range(len(X)):
        evaluation=sigmoid(theta.T@X[i]+bias)
        if evaluation>threshold:
            prediction.append(1)
        else:
            prediction.append(0)
    
    returned_pred=prediction
    
    with open(output_file,'w') as f_out:
        for prediction in prediction:
            f_out.write(str(prediction)+'\n')
    
    return returned_pred

def compute_error_header(test_prediction,train_prediction,train_input,test_input,metrics_out):
    data_train_str=extract_data(train_input)
    data_test_str=extract_data(test_input)
    train_processed=process_data(data_train_str)
    train_true_labels=train_processed[1]
    test_processed=process_data(data_test_str)
    test_true_labels=test_processed[1]
    compute_error(train_prediction,train_true_labels,test_prediction,test_true_labels,metrics_out)
    return


def compute_error(train_pred,train_real,test_pred,test_real,metrics):
    train_count=0
    for i in range(len(train_pred)):
        if train_pred[i]!=train_real[i]:
            train_count+=1
    train_error_rate=train_count/len(train_pred)
    
    test_count=0
    for i in range(len(test_pred)):
        if test_pred[i]!=test_real[i]:
            test_count+=1
    test_error_rate=train_count/len(test_pred)
    test_error_rate=test_count/len(test_pred)
    with open(metrics, 'w') as f_out:
        f_out.write("error(train): " + str(train_error_rate) + '\n')
        f_out.write("error(test): " + str(test_error_rate) + '\n')
    
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to formatted training data')
    parser.add_argument("validation_input", type=str, help='path to formatted validation data')
    parser.add_argument("test_input", type=str, help='path to formatted test data')
    parser.add_argument("train_out", type=str, help='file to write train predictions to')
    parser.add_argument("test_out", type=str, help='file to write test predictions to')
    parser.add_argument("metrics_out", type=str, help='file to write metrics to')
    parser.add_argument("num_epoch", type=int, 
                        help='number of epochs of stochastic gradient descent to run')
    parser.add_argument("learning_rate", type=float,
                        help='learning rate for stochastic gradient descent')
    args = parser.parse_args()



trained_parameters=train_data_header(args.train_input,args.num_epoch,args.learning_rate)
test_prediction=predict_data_header(args.test_input,trained_parameters,args.test_out)
train_prediction=predict_data_header(args.train_input,trained_parameters,args.train_out)
compute_error_header(test_prediction,train_prediction,args.train_input,args.test_input,args.metrics_out)















