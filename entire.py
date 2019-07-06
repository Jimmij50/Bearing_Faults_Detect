import pandas as pb
import tensorflow as tf
import numpy as np
import pandas as pd
import math
import  os


from cryptography.hazmat.primitives import padding
from tensorflow.python.framework import graph_util
MANIFEST_DIR='C:/Study/Professional Course/Pattern Recognition/DC competition/bearing_detection_by_conv1d-master/Bear_datatrain.csv'
pb.read_csv(MANIFEST_DIR)
Batch_size = 20
Long = 792
Lens = 640

def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

def xs_gen(path=MANIFEST_DIR,batch_size = Batch_size,train=True,Lens=Lens):

    img_list = pd.read_csv(path)
    if train:
        img_list = np.array(img_list)[:Lens]
        print("Found %s train items."%len(img_list))
        print("list 1 is",img_list[0,-1])
        steps = math.ceil(len(img_list) / batch_size)    # 确定每轮有多少个batch
    else:
        img_list = np.array(img_list)[Lens:]
        print("Found %s test items."%len(img_list))
        print("list 1 is",img_list[0,-1])
        steps = math.ceil(len(img_list) / batch_size)    # 确定每轮有多少个batch
    while True:
        for i in range(steps):

            batch_list = img_list[i * batch_size : i * batch_size + batch_size]
            np.random.shuffle(batch_list)
            batch_x = np.array([file for file in batch_list[:,1:-1]])
            batch_y = np.array([convert2oneHot(label,10) for label in batch_list[:,-1]])

            yield batch_x, batch_y

TEST_MANIFEST_DIR = "Bear_data/test_data.csv"

def ts_gen(path=TEST_MANIFEST_DIR,batch_size = Batch_size):

    img_list = pd.read_csv(path)

    img_list = np.array(img_list)[:Lens]
    print("Found %s train items."%len(img_list))
    print("list 1 is",img_list[0,-1])
    steps = math.ceil(len(img_list) / batch_size)    # 确定每轮有多少个batch
    while True:
        for i in range(steps):

            batch_list = img_list[i * batch_size : i * batch_size + batch_size]
            #np.random.shuffle(batch_list)
            batch_x = np.array([file for file in batch_list[:,1:]])
            #batch_y = np.array([convert2oneHot(label,10) for label in batch_list[:,-1]])

            yield batch_x



TIME_PERIODS = 6000
def convert2oneHot(index,Lens):
    hot = np.zeros((Lens,))
    hot[int(index)] = 1
    return(hot)

def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):#W 4 dimension
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def conv1d_2(x, W):# W 3 dimension 参数可能有问题 height stride layer
    return  tf.nn.con1d(x,W,2,padding='SAME')

def conv1d_1(x, W):# W 3 dimension 参数可能有问题 height  input_chanel output_channel
    return  tf.nn.con1d(x,W,2,padding='SAME')

def max_pool_1d(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='VALID')



def build_network(height,width):
    '''
    
    :param height: 输入的维度shape[0] 列
    :param width: 输入的维度shae=pe[1] 行
    :return: graph
    '''
    x=tf.placeholder(tf.float32,[None,TIME_PERIODS],name='input')
    y_placeholder=tf.placeholder(tf.float32,shape=[None,10],name='labels_placeholder')#输出的类别为10

    #第一组

    W_conv1=weight_variable([8,1,16])
    b_conv1=bias_variable([16])

    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)

    W_conv2 = weight_variable([8, 1, 16])
    b_conv2 = bias_variable([16])

    h_conv2 = tf.nn.relu(conv1d_2(h_conv1, W_conv2) + b_conv2)


    h_pool2 = max_pool_1d(h_conv2)

    # 第二组

    W_conv3 = weight_variable([4, 16, 64])
    b_conv3 = bias_variable([64])

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

    W_conv4 = weight_variable([4, 16, 64])
    b_conv4 = bias_variable([64])

    h_conv4 = tf.nn.relu(conv1d_2(h_conv3, W_conv4) + b_conv4)

    h_pool4 = max_pool_1d(h_conv4)


# 第三组logits = tf.matmul(h_pool4_flat, W_fc1) + b_fc1
    #
    #     sofmax_out = tf.nn.softmax(logits, name="out_softmax")

    W_conv5 = weight_variable([2, 64, 512])
    b_conv5 = bias_variable([512])

    h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)

    W_conv6 = weight_variable([2, 64, 512])
    b_conv6 = bias_variable([512])

    h_conv6 = tf.nn.relu(conv1d_1(h_conv5, W_conv6) + b_conv6)

    h_pool6 = max_pool_1d(h_conv6)
    #FC
    '''
    h_pool4_flat = tf.reshape(h_pool6, [-1, 7 * 7 * 512])
    W_fc1 = weight_variable([7 * 7 * 512, 2])
    b_fc1 = bias_variable([2])
    logits = tf.matmul(h_pool4_flat, W_fc1) + b_fc1

    sofmax_out = tf.nn.softmax(logits, name="out_softmax")
    '''

    dropout1 = tf.nn.dropout(h_pool6, 0.3)
    pool_shape = dropout1.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(dropout1, [pool_shape[0], nodes])



    fc1_w = weight_variable([nodes, 256])
    fc1_b = bias_variable([256])
    fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b)

    dropout2 = tf.nn.dropout(fc1, 0.3)

    '''
    还差一层Global Average Pooling
    '''
    # if train: fc1 = tf.nn.dropout(fc1, 0.5)
    fc2_w = weight_variable([256, 10])
    fc2_b = bias_variable([10])
    y = tf.matmul(fc1, fc2_w) + fc2_b
    sofmax_out = tf.nn.softmax(y, name="out_softmax")
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=sofmax_out, labels=y_placeholder))
    optimize = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

    prediction_labels = tf.argmax(sofmax_out, axis=1)
    real_labels = tf.argmax(y_placeholder, axis=1)

    correct_prediction = tf.equal(prediction_labels, real_labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 一个Batch中预测正确的次数
    correct_times_in_batch = tf.reduce_sum(tf.cast(correct_prediction, tf.int32))




    return dict(
        #keep_prob_placeholder=keep_prob_placeholder,
        x_placeholder=x,
        y_placeholder=y_placeholder,
        optimize=optimize,
        logits=sofmax_out,
        prediction_labels=prediction_labels,
        real_labels=real_labels,
        correct_prediction=correct_prediction,
        correct_times_in_batch=correct_times_in_batch,
        cost=cost,
        accuracy=accuracy,
    )
def train_test_split(ratio):
    data, label = read_img()
    #print(data.shape)
    print(data.shape)
    # data = np.array(data).reshape(-1, 200, 200,3);
    # print(data.shape)
    #打乱顺序
    num_example=data.shape[0]
    arr=np.arange(num_example)
    np.random.shuffle(arr)
    data=data[arr]
    label=label[arr]
    label=_dense_to_one_hot(label,2)

    #将所有数据分为训练集和验证集
    ratio=0.8
    s=np.int(num_example*ratio)
    x_train=data[:s]
    y_train=label[:s]
    x_test=data[s:]
    y_test=label[s:]
    return x_train,x_test,y_train,y_test



def train_network(graph,
                  batch_size,
                  num_epochs,
                  pb_file_path, ):
    """
    Function：训练网络。

    Parameters
    ----------
        graph: 一个dict,build_network函数的返回值。
        dataset: 数据集
        batch_size:
        num_epochs: 训练轮数。
        pb_file_path：要生成的pb文件的存放路径。
    """
    x_train,x_test,y_train,y_test=train_test_split(0.8)
    config = tf.ConfigProto(allow_soft_placement=True)
    tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())

        print("batch size:", batch_size)

        # 用于控制每epoch_delta轮在train set和test set上计算一下accuracy和cost
        epoch_delta = 2
        for epoch_index in range(num_epochs):

            #################################
            #    获取TRAIN set，开始训练网络
            #################################
            for (batch_xs, batch_ys) in minibatches(x_train,y_train,batch_size,shuffle=True):
                sess.run([graph['optimize']], feed_dict={
                    graph['x_placeholder']: batch_xs,
                    graph['y_placeholder']: batch_ys,
                    #graph['keep_prob_placeholder']: 0.5,
                })

            # 每epoch_delta轮在train set和test set上计算一下accuracy和cost
            if epoch_index % epoch_delta == 0:
                #################################
                #    开始在 train set上计算一下accuracy和cost
                #################################
                # 记录训练集中有多少个batch
                total_batches_in_train_set = 0
                # 记录在训练集中预测正确的次数
                total_correct_times_in_train_set = 0
                # 记录在训练集中的总cost
                total_cost_in_train_set = 0.
                for (train_batch_xs, train_batch_ys) in minibatches(x_train, y_train, batch_size, shuffle=True):
                    return_correct_times_in_batch = sess.run(graph['correct_times_in_batch'], feed_dict={
                        graph['x_placeholder']: train_batch_xs,
                        graph['y_placeholder']: train_batch_ys,
                        #graph['keep_prob_placeholder']: 1.0,
                    })
                    mean_cost_in_batch = sess.run(graph['cost'], feed_dict={
                        graph['x_placeholder']: train_batch_xs,
                        graph['y_placeholder']: train_batch_ys,
                        #graph['keep_prob_placeholder']: 1.0,
                    })

                    total_batches_in_train_set += 1
                    total_correct_times_in_train_set += return_correct_times_in_batch
                    total_cost_in_train_set += (mean_cost_in_batch * batch_size)

                #################################
                # 开始在 test set上计算一下accuracy和cost
                #################################
                # 记录测试集中有多少个batch
                total_batches_in_test_set = 0
                # 记录在测试集中预测正确的次数
                total_correct_times_in_test_set = 0
                # 记录在测试集中的总cost
                total_cost_in_test_set = 0.
                for (test_batch_xs, test_batch_ys) in minibatches(x_test, y_test, batch_size, shuffle=False):
                    return_correct_times_in_batch = sess.run(graph['correct_times_in_batch'], feed_dict={
                        graph['x_placeholder']: test_batch_xs,
                        graph['y_placeholder']: test_batch_ys,
                        #graph['keep_prob_placeholder']: 1.0,
                    })
                    mean_cost_in_batch = sess.run(graph['cost'], feed_dict={
                        graph['x_placeholder']: test_batch_xs,
                        graph['y_placeholder']: test_batch_ys,
                        #graph['keep_prob_placeholder']: 1.0,
                    })

                    total_batches_in_test_set += 1
                    total_correct_times_in_test_set += return_correct_times_in_batch
                    total_cost_in_test_set += (mean_cost_in_batch * batch_size)

                ### summary and print
                acy_on_test = total_correct_times_in_test_set / float(total_batches_in_test_set * batch_size)
                acy_on_train = total_correct_times_in_train_set / float(total_batches_in_train_set * batch_size)
                print(
                    'Epoch - {:2d} , acy_on_test:{:6.2f}%({}/{}),loss_on_test:{:6.2f}, acy_on_train:{:6.2f}%({}/{}),loss_on_train:{:6.2f}'.
                    format(epoch_index, acy_on_test * 100.0, total_correct_times_in_test_set,
                           total_batches_in_test_set * batch_size, total_cost_in_test_set, acy_on_train * 100.0,
                           total_correct_times_in_train_set, total_batches_in_train_set * batch_size,
                           total_cost_in_train_set))

                # 每轮训练完后就保存为pb文件
            constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["out_softmax"])  # out_softmax
            with tf.gfile.FastGFile(pb_file_path, mode='wb') as f:
                f.write(constant_graph.SerializeToString())
def main():
    batch_size = 20
    num_epochs = 10

    # pb文件保存路径
    pb_file_path = "/home/dyf/PycharmProjects/classifer_flower/output/model.pb"

    g = build_network(height=500, width=500)
    train_network(g, batch_size, num_epochs, pb_file_path)
    g = build_network(height=500, width=500)
    train_network(g, batch_size, num_epochs, pb_file_path)