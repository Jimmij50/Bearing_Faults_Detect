import tensorflow as tf
import numpy as np
import PIL.Image as Image
from skimage import io, transform
import os
import pandas as pd
TIME_PERIODS=33
pb_file_path='model_MLP.pb'
TEST_DIR=TEST_PATH='feature_test_e.csv'
#TEST_DIR=TEST_PATH='train.csv'
Data=pd.read_csv(TEST_DIR)
Data=np.array(Data)
data=Data.transpose()
print(data[1])
print(data.shape)
feature=data
feature2 = []
for j in feature:
    # k=np.fft.fft(j)
    # feature2.append(abs(k))
    feature2.append(j)
feature2=np.array(feature2)
feature2=np.reshape(feature2,(-1,TIME_PERIODS))
print('feature')
print(feature[1])
print('feature2')
print(feature2[1])
def judge(data_path,pb_file_path):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()

        with open(pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())  # rb
            _ = tf.import_graph_def(output_graph_def, name="")

        config = tf.ConfigProto(allow_soft_placement=True)
        tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()

            input_x = sess.graph.get_tensor_by_name("input/input:0")
            print(input_x)
            out_softmax = sess.graph.get_tensor_by_name("out_softmax/out_softmax:0")
            print(out_softmax)
            keep_prob=sess.graph.get_tensor_by_name("input/keep_prob:0")
            # keep_prob = sess.graph.get_tensor_by_name("keep_prob_placeholder:0")
            # print(keep_prob)
            # out_label = sess.graph.get_tensor_by_name("output:0")
            # print(out_label)

            # img_datas = np.array(Image.open(data_path).convert('L'))
            # img_datas = np.array(Image.open(data_path))
            # new_path = os.path.join(os.path.abspath(NG_path), '1 (' + format(str(i + 1)) + ').bmp')

            # imgs = np.array(imgs).reshape(-1, 200, 200, 3);
            # data = np.multiply(imgs, 1.0 / 255.0)
            img_out_softmax = sess.run(out_softmax, feed_dict={
                input_x: feature2,
                keep_prob: 1.0,
                })
            # for i in range(13):
            #     prediction_label = np.argmax(img_out_softmax[i].reshape(1, 2, 1), axis=1)
            #     print(img_out_softmax[i])
            #     if prediction_label[0] == 0:
            #         c = c + 1
            #     # print(prediction_label[0])
            # for i in range(13, 22):
            #     prediction_label = np.argmax(img_out_softmax[i].reshape(1, 2, 1), axis=1)
            #     print(img_out_softmax[i])
            #     # print(img_out_softmax[i].reshape(1,2,1))
            #     if prediction_label[0] == 1:
            #         c = c + 1
            #         print(prediction_label[0])
            # print("acc:", c / 22)
            a=[]
            for i in img_out_softmax:
                a.append(np.argmax(i))
        df = pd.DataFrame()
        df["id"] = np.arange(1,len(a)+1)
        df["label"] = a
        df.to_csv("submmit_MLP.csv",index=None)
judge(TEST_DIR,pb_file_path)


