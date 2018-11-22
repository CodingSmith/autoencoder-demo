import tensorflow as tf
#from utils import Mnist
import tensorflow.examples.tutorials.mnist.input_data as input_data
import os
import datetime


logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
# Parameter
learning_rate = 0.001
training_epochs = 500
batch_size = 256
display_step = 1
examples_to_show = 10
 
mnist = input_data.read_data_sets("/home/baofeng/project_code/GAN/Conditional-GAN/data/mnist", one_hot=True) 
# Network Parameters
n_input = 784  # MNIST data input (img shape: 28*28)
 
# hidden layer settings
n_hidden_1 = 256 # 1st layer num features
n_hidden_2 = 128 # 2nd layer num features
n_hidden_3 = 64  # 3rd layer num features
 
X = tf.placeholder(tf.float32, [None,n_input])
 
weights = {
    'encoder_h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
    'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2,n_hidden_3])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_3,n_hidden_2])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_2,n_hidden_1])),
    'decoder_h3': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
    }
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b2': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b3': tf.Variable(tf.random_normal([n_input])),
    }
 
# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                   biases['encoder_b3']))
    return layer_3
     
# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                   biases['decoder_b3']))
    return layer_3
 
# Construct model
encoder_op = encoder(X)             # 128 Features
decoder_op = decoder(encoder_op)    # 784 Features
 
# Prediction
y_pred = decoder_op    # After
# Targets (Labels) are the input data.
y_true = X            # Before
 
# Define loss and optimizer, minimize the squared error
print("y_true", y_true)
print("y_pred", y_pred)
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
cost_scalar = tf.summary.scalar("cost", cost)
merged_cost = tf.summary.merge([cost_scalar])

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
 
# Launch the graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(logdir, graph=sess.graph)
    total_batch = int(mnist.train.num_examples/batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # max(x) = 1, min(x) = 0
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, merged_cost], feed_dict={X: batch_xs})
        summary_writer.add_summary(c, epoch)
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1))
                #  "cost=", cost.eval())
#            print("Epoch:", (epoch+1))
            print("cost=", cost)
 
    print("Optimization Finished!")