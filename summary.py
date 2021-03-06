import tensorflow as tf
import  numpy as np
import pandas as pd
import  math
from tensorflow.python.framework import graph_util
CLASS=10
BATCH_SIZE=20
LENS=640
TRAIN_PATH='train.csv'
TEST_PATH='test_data.csv'
EPOCHS=20
TIME_PERIODS = 6000
LEARNING_RATE_BASE = 1e-3
LEARNING_RATE_DECAY = 0.1
REGULARIZER = 0.000001
STEPS_decay=2
STEPS = 50000
MOVING_AVERAGE_DECAY = 0.99
Data=pd.read_csv(TRAIN_PATH)
def convert2oneHot(index,Lens):
    hot = np.zeros((Lens,))
    hot[int(index)] = 1
    return(hot)

'''
tf.nn.conv1d
1.value：在注释中，value的格式为：[batch, in_width, in_channels]，batch为样本维，表示多少个样本，in_width为宽度维，表示样本的宽度，in_channels维通道维，表示样本有多少个通道。 
  事实上，也可以把格式看作如下:[batch, 行数, 列数]，把每一个样本看作一个平铺开的二维数组。这样的话可以方便理解。

2、filters：在注释中，filters的格式为：[filter_width, in_channels, out_channels]。按照value的第二种看法，filter_width可以看作每次与value进行卷积的行数，in_channels表示value一共有多少列（与value中的in_channels相对应）。out_channels表示输出通道，可以理解为一共有多少个卷积核，即卷积核的数目。

3、stride：一个整数，表示步长，每次（向下）移动的距离（TensorFlow中解释是向右移动的距离，这里可以看作向下移动的距离）。


'''
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    w = tf.Variable(initial)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(REGULARIZER)(w))
    return w
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d_1(x, W):#W 4 dimension
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def conv2d_2(x, W):#W 4 dimension
    return tf.nn.conv2d(x,W,strides=[1,1,2,1],padding='SAME')

def conv1d_2(x, W):# W 3 dimension 参数可能有问题
    return  tf.nn.conv1d(x,W,2,padding='SAME')

def conv1d_1(x, W):# W 3 dimension 参数可能有问题 [height  input_chanel output_channel]
    return  tf.nn.conv1d(x,W,1,padding='SAME')

def max_pool_1d(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='VALID')




def one_hot(label,classnum):
    height=label.shape[0]
    offset=np.arange(height)*classnum
    label=label.astype(int)
    onehot=np.zeros((height,classnum))
    onehot.flat[offset+label.ravel()]=1# 主要一定要吧label 变成整形
    return  onehot

# def one_hot(labels_dense, num_classes):
#   """Convert class labels from scalars to one-hot vectors."""
#   num_labels = labels_dense.shape[0]
#   labels_dense=labels_dense.astype(int)
#   print(labels_dense)
#   index_offset = np.arange(num_labels) * num_classes
#   labels_one_hot = np.zeros((num_labels, num_classes))
#   labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
#   return labels_one_hot

def train_data(train):
    data=Data
    data=np.array(data)

    if train:
        data=data[:LENS]
    else:
        data=data[LENS:]
   #print(data.shape)
    steps=math.ceil(len(data)/BATCH_SIZE)
    np.random.shuffle(data)
    #print(data.shape)

    for i in range(steps):
        Batch=data[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE]
        feature=Batch[:,1:-1]
        feature2 = []
        for j in feature:
            j=np.fft.fft(j)
            feature2.append(abs(j))
        label=Batch[:,-1]
        label_onehot=one_hot(label,10)
        feature2=np.reshape(feature2,(-1,1,TIME_PERIODS,1))
        #label_onehot=np.reshape(label_onehot,(BATCH_SIZE,1,CLASS,1))
        yield feature2,label_onehot
t=train_data(True)
for i in t:
    print(i)
    t.close()
def test_data():
    data=Data
    data=np.array(data)
    steps = math.ceil(len(data) / BATCH_SIZE)
    for i in range(steps):
        Batch=data[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE]
        feature=Batch[:,:-1]
        label=Batch[:,-1]
        label_onehot=one_hot(label,10)
        yield feature,label_onehot


def build_network():
    '''

    :param height: 输入的维度shape[0] 列
    :param width: 输入的维度shae=pe[1] 行
    :return: graph
    '''
    
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
    LEARNING_RATE_BASE,
    global_step,
    STEPS_decay,
    LEARNING_RATE_DECAY,
    staircase=True)
    # with tf.name_scope('learning_rate_decay'):
    #     global_step = tf.Variable(0, trainable=False)
    #     learning_rate = tf.train.exponential_decay(
    #     LEARNING_RATE_BASE,
    #     global_step,
    #     STEPS_decay,
    #     LEARNING_RATE_DECAY,
    #     staircase=True)
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None,1,6000,1], name='input')
        y_placeholder = tf.placeholder(tf.float32, shape=[None,10], name='labels_placeholder')  # 输出的类别为10
        keep_prob = tf.placeholder(tf.float32,name='keep_prob')
        tf.summary.audio('input',tf.reshape(x,[-1,1,6000]),1)
    # 第一组
    with tf.name_scope('layer1'):
    
        W_conv1 = weight_variable([1,8,1, 16])
        b_conv1 = bias_variable([16])
        tf.summary.histogram('weight',W_conv1)
        tf.summary.histogram('bias',b_conv1)
        tf.summary.scalar('bias',tf.reduce_mean(b_conv1))

        h_conv1 = tf.nn.relu(conv2d_2(x, W_conv1) + b_conv1)
    with tf.name_scope('layer2'):
        W_conv2 = weight_variable([1,8,16, 16])
        b_conv2 = bias_variable([16])
        tf.summary.histogram('weight',W_conv2)
        tf.summary.histogram('bias',b_conv2)

        h_conv2 = tf.nn.relu(conv2d_2(h_conv1, W_conv2) + b_conv2)

        h_pool2 = max_pool_1d(h_conv2)

    # 第二组
    with tf.name_scope('layer3'):
        W_conv3 = weight_variable([1,4, 16, 64])
        b_conv3 = bias_variable([64])

        h_conv3 = tf.nn.relu(conv2d_2(h_pool2, W_conv3) + b_conv3)
    with tf.name_scope('layer4'):
        W_conv4 = weight_variable([1,4, 64, 64])
        b_conv4 = bias_variable([64])

        h_conv4 = tf.nn.relu(conv2d_1(h_conv3, W_conv4) + b_conv4)

        h_pool4 = max_pool_1d(h_conv4)

    # 第三组logits = tf.matmul(h_pool4_flat, W_fc1) + b_fc1
    #
    #     sofmax_out = tf.nn.softmax(logits, name="out_softmax")
    with tf.name_scope('layer5'):
        W_conv5 = weight_variable([1,2, 64, 512])
        b_conv5 = bias_variable([512])

        h_conv5 = tf.nn.relu(conv2d_2(h_pool4, W_conv5) + b_conv5)

        W_conv6 = weight_variable([1,2, 512, 512])
        b_conv6 = bias_variable([512])
        tf.summary.histogram('weight',W_conv5)
        tf.summary.histogram('bias',b_conv5)
    with tf.name_scope('layer6'):
        h_conv6 = tf.nn.relu(conv2d_1(h_conv5, W_conv6) + b_conv6)

        h_pool6 = max_pool_1d(h_conv6)
    # FC
    '''
    h_pool4_flat = tf.reshape(h_pool6, [-1, 7 * 7 * 512])
    W_fc1 = weight_variable([7 * 7 * 512, 2])
    b_fc1 = bias_variable([2])
    logits = tf.matmul(h_pool4_flat, W_fc1) + b_fc1

    sofmax_out = tf.nn.softmax(logits, name="out_softmax")
    '''
    with tf.name_scope('dropout1'):
        dropout1 = tf.nn.dropout(h_pool6, keep_prob)
    pool_shape = dropout1.get_shape().as_list()
    nodes = pool_shape[1]*pool_shape[2] * pool_shape[3]
    '''
    shape[0] batch size
    shape[1]  width
    shape[2] channels 
    '''
    print(pool_shape)
    print(nodes)
    reshaped = tf.reshape(dropout1, [-1, nodes])
    with tf.name_scope('fc1'):
        fc1_w = weight_variable([nodes, 256])
        fc1_b = bias_variable([256])
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b)
    with tf.name_scope('dropout1'):
        dropout2 = tf.nn.dropout(fc1, keep_prob)

    '''
    还差一层Global Average Pooling
    '''
    # if train: fc1 = tf.nn.dropout(fc1, 0.5)
    with tf.name_scope('fc2'):
        fc2_w = weight_variable([256, 10])
        fc2_b = bias_variable([10])
        y = tf.matmul(dropout2, fc2_w) + fc2_b
    with tf.name_scope('out_softmax'):
        sofmax_out = tf.nn.softmax(y, name="out_softmax")
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_placeholder))
        cost=loss+tf.add_n(tf.get_collection('losses'))
        tf.summary.scalar('loss',cost)
    #global_step = tf.Variable(0, trainable=False)
    #learning_rate = tf.train.exponential_decay(
        #LEARNING_RATE_BASE,
        #global_step,
        #3,
        #LEARNING_RATE_DECAY,
        #staircase=True)

    #with tf.name_scope('train'):
    optimize = tf.train.AdamOptimizer(learning_rate).minimize(cost)


    with tf.name_scope('accuracy'):
        prediction_labels = tf.argmax(sofmax_out, axis=1)
        real_labels = tf.argmax(y_placeholder, axis=1)
        correct_prediction = tf.equal(prediction_labels, real_labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy',accuracy)
    with tf.name_scope('init'):
        init=tf.global_variables_initializer()
    # 一个Batch中预测正确的次数
    with tf.name_scope('correction_times'):
        correct_times_in_batch = tf.reduce_sum(tf.cast(correct_prediction, tf.int32))
    
    return dict(
        # keep_prob_placeholder=keep_prob_placeholder,
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
        keep_prob=keep_prob,
        init=init,
        global_step=global_step
    )

def train_network(graph,pb_file_path,batch_size=BATCH_SIZE,num_epochs=EPOCHS):
    # train=train_data(True)
    # test=train_data(False)
    saver=tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement=True)
    tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    config.gpu_options.allow_growth = True
    sesss = tf.Session()
    merged = tf.summary.merge_all() 
    writer = tf.summary.FileWriter("logs/", sesss.graph)
    with tf.Session(config=config) as sess:
        sess.run(graph['init'])
        epoch_delta = 2
        for index in range(num_epochs):
            # run_option=tf.RunOptions(
            #     trace_level=tf.RunOptions.FULL_TRACE
            # )
            # run_metadata=tf.RunMetadata()
            train = train_data(True)
            test = train_data(False)
            for (train_feature,train_label) in train:
                sess.run(graph['optimize']
                ,feed_dict={graph['x_placeholder']:train_feature,
                            graph['y_placeholder']:train_label,
                            graph['keep_prob']:0.75,
                            graph['global_step']:index
                            
                            })
                if index % epoch_delta==0:
                    total_correction_times_train=0
                    total_cost_train=0
                    Batch_times_train=0

                    total_correction_times_test = 0
                    total_cost_test = 0
                    Batch_times_test = 0
                    for(train_feature,train_label) in train:
                        correction_times=sess.run(graph['correct_times_in_batch']
                        ,feed_dict={graph['x_placeholder']: train_feature,
                            graph['y_placeholder']: train_label,
                                    graph['keep_prob']: 1
                            })
                        cost = sess.run(graph['cost']
                        ,feed_dict={graph['x_placeholder']: train_feature,
                            graph['y_placeholder']: train_label,
                                    graph['keep_prob']: 1
                            })
                        total_correction_times_train+=correction_times
                        total_cost_train+=(cost*BATCH_SIZE)
                        Batch_times_train+=1

                    for (test_feature, test_label) in test:
                        correction_times = sess.run(graph['correct_times_in_batch']
                                                    , feed_dict={graph['x_placeholder']: test_feature,
                                                                 graph['y_placeholder']: test_label,
                                                                 graph['keep_prob']: 1.0
                                                                 })
                        cost = sess.run(graph['cost']
                                        , feed_dict={graph['x_placeholder']: test_feature,
                                                     graph['y_placeholder']: test_label,
                                                     graph['keep_prob']: 1.0
                                                     })
                        total_correction_times_test += correction_times
                        total_cost_test += (cost * BATCH_SIZE)
                        Batch_times_test += 1
                    accuracy_train=total_correction_times_train/float(Batch_times_train*BATCH_SIZE)
                    accuracy_test=total_correction_times_test/float(152)
                    print('epoch- {:2d},accuray_train:{:6.2f}%({}/{}),loss_train={:6.2f},accuray_test   :{:6.2f}%({}/{}),loss_test={:6.2f}'
                          .format(index,accuracy_train*100.0,total_correction_times_train,Batch_times_train*BATCH_SIZE,total_cost_train,
                                  accuracy_test*100.0,total_correction_times_test,152,total_cost_test))
            if(index%1==0): #每50次写一次日志
                result = sess.run(merged,feed_dict={graph['x_placeholder']:train_feature,
                            graph['y_placeholder']:train_label,
                            graph['keep_prob']:0.75
                            }) #计算需要写入的日志数据
                writer.add_summary(result,index) #将日志数据写入文件
        # sess_highdimen = tf.InteractiveSession()
        # saver = tf.train.Saver()
        # sess_highdimen.run(graph['init']) 
        saver.save(sess,"./logs/model.ckpt")   
                # if index==(num_epochs-1):
                #     constant_graph=graph_util.convert_variables_to_constants(sess,sess.graph_def,["out_softmax"])
                #     print(index)
                #     with tf.gfile.FastGFile(pb_file_path,mode='wb') as f:
                #         f.write(constant_graph.SerializeToString())
# def visualisation(final_result):
#     #定义一个新向量保存输出层向量的取值
#     y = tf.Variable(final_result,name=tensor_name)
#     #定义日志文件writer
#     summary_writer = tf.summary.FileWriter(log_dir)

#     #ProjectorConfig帮助生成日志文件
#     config = projector.ProjectorConfig()
#     #添加需要可视化的embedding
#     embedding = config.embeddings.add()
#     #将需要可视化的变量与embedding绑定
#     embedding.tensor_name = y.name

#     #指定embedding每个点对应的标签信息，
#     #这个是可选的，没有指定就没有标签信息
#     embedding.metadata_path = meta_file
#     #指定embedding每个点对应的图像，
#     #这个文件也是可选的，没有指定就显示一个圆点
#     embedding.sprite.image_path = sprite_file
#     #指定sprite图中单张图片的大小
#     embedding.sprite.single_image_dim.extend([28,28])

#     #将projector的内容写入日志文件
#     projector.visualize_embeddings(summary_writer,config)

#     #初始化向量y，并将其保存到checkpoints文件中，以便于TensorBoard读取
#     sess = tf.InteractiveSession()
#     sess.run(tf.global_variables_initializer())
#     saver = tf.train.Saver()
#     saver.save(sess,os.path.join(log_dir,'model'),training_steps)
#     summary_writer.close()


def main():
    pb_file_path='model.pb'
    g=build_network()
    train_network(g,pb_file_path)
main()