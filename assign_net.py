# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases   
    
    #自由選擇激活函數
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def create_new_model():
    
    xs = tf.placeholder(tf.float32, [None, 1])        
    l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)  
    l2 = add_layer(l1, 10, 5, activation_function=tf.nn.relu)  
    prediction = add_layer(l2, 5, 1, activation_function=None)  
   
    return xs,prediction
    
def create_old_model():
    
    xs = tf.placeholder(tf.float32, [None, 1])        
    l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)  
    prediction = add_layer(l1, 10, 1, activation_function=None)  
   
    return xs,prediction
    
"Step1:Get the weights which we trained before"

sess = tf.Session()

xs,prediction=create_old_model()

# train method
ys = tf.placeholder(tf.float32, [None, 1])
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


saver = tf.train.Saver()
saver.restore(sess, "my_networks/my-three-layer-networks-3000")

variable = tf.all_variables()
variable_list=sess.run(variable)

for i in range(0,len(variable_list),2):
    print "w",variable_list[i].shape,variable_list[i],    "\n"
    print "b",variable_list[i+1].shape,variable_list[i+1],"\n"
    

"Step2:Prepare to fill in the old weights to new model"
xs_new,prediction_new=create_new_model()

x_data = np.linspace(-1,1,300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

ys_new = tf.placeholder(tf.float32, [None, 1])
loss_new = tf.reduce_mean(tf.reduce_sum(tf.square(ys_new - prediction_new),reduction_indices=[1]))
train_step_new = tf.train.GradientDescentOptimizer(0.1).minimize(loss_new)
 
#全部設定好了之後 記得初始化喔
init = tf.initialize_all_variables()
sess2 = tf.Session()
sess2.run(init)
 
trainable_variables=tf.trainable_variables()
#sess2.run(train_step_new, feed_dict={xs_new: x_data, ys_new: y_data})
new_variable_list=sess2.run(trainable_variables)
#for i in range(0,len(new_variable_list),2):
#    print "w",new_variable_list[i].shape,new_variable_list[i],    "\n"
#    print "b",new_variable_list[i+1].shape,new_variable_list[i+1],"\n"
    
    
assing_op=[tf.assign(trainable_variables[0],variable_list[0]),tf.assign(trainable_variables[1],variable_list[1])]
sess2.run(assing_op)
new_variable_list=sess2.run(trainable_variables)
print "After assign:\n"
for i in range(0,len(new_variable_list),2):
    print "w",new_variable_list[i].shape,new_variable_list[i],    "\n"
    print "b",new_variable_list[i+1].shape,new_variable_list[i+1],"\n"

