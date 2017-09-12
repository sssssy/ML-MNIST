import input_data
import read_data
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf  
  
#MNIST数据输入  
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
  
x = tf.placeholder(tf.float32,[None, 784]) #图像输入向量  
W = tf.Variable(tf.zeros([784,10]))  #权重，初始化值为全零  
b = tf.Variable(tf.zeros([10]))  #偏置，初始化值为全零  
  
#进行模型计算，y是预测，y_ 是实际  
y = tf.nn.softmax(tf.matmul(x,W) + b)  
  
y_ = tf.placeholder("float", [None,10])  
  
#计算交叉熵  
cross_entropy = -tf.reduce_sum(y_*tf.log(y))  
#接下来使用BP算法来进行微调,以0.01的学习速率*
#   *alpha > 0.03时会产生nan错误，alpha < 0.001后准确率有所下降
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)  
  
#添加初始化创建变量的操作  
init = tf.global_variables_initializer()  
#启动创建的模型，并初始化变量  
sess = tf.Session()  
sess.run(init)

#开始训练模型，循环训练1000次  
for i in range(1000):  
    #随机抓取训练数据中的100个批处理数据点  
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x:batch_xs,y_:batch_ys})
###输出权重W的字符图像
##w = sess.run(W)
##for i in range(10):
##    for j in range(784):
##        if w[j][i] > 0:
##            print('*',sep = ' ', end = ',')
##        else:
##            print(' ',sep = ' ', end = ',')
##    print('\n\n\n\n')
''''' 进行模型评估 '''''
  
#判断预测标签和实际标签是否匹配  
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))   
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  
#计算所学习到的模型在测试数据集上面的正确率  
print('accuracy =',sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))

'''''模型演示'''''

#导入测试数据集，抽取随机数据,计算结果
images = read_data.test_images()
index = random.randint(0, 10000)
image = []
image.append(images[index])
print('y =',sess.run(y, feed_dict = {x: image}))
#显示抽取的随机数据的图片
im = np.array(image[0])
im = im.reshape(28,28)
fig = plt.figure()
plotwindow = fig.add_subplot(111)
plt.imshow(im, cmap = 'gray')
plt.show()



