import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs,in_size,out_size,n_layer,activation_function=None):

    layer_name = 'layer%s' % n_layer

    with tf.name_scope(layer_name):

        Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W')
        tf.summary.histogram('/weights',Weights)

        biases = tf.Variable(tf.zeros([1,out_size])+0.1,name='b')
        tf.summary.histogram('biases',biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs,Weights),biases)

        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs=activation_function(Wx_plus_b)

        tf.summary.histogram('/outputs', outputs)
        return outputs

x_data = np.linspace(-1,1,300)[:, np.newaxis] #300行
noise = np.random.normal(0,0.05,x_data.shape)
y_data= np.square(x_data)-0.5+noise

#给神经网络定义输入的placeholder
with tf.name_scope('inputs'):   #添加可视化的大框架
    xs = tf.placeholder(tf.float32,[None,1],name='x_input') #无论给多少个例子都可以 None
    ys = tf.placeholder(tf.float32,[None,1],name='y_input')

#增加一个隐藏层
l1 = add_layer(xs,1,10,n_layer=1,activation_function=tf.nn.relu) #输入1层，隐藏层10，激励函数relu，不用定义也可以可视化显示名字
#增加一个隐藏层
predition = add_layer(l1,10,1,n_layer=2,activation_function=None)

#the error between prediction and real data

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-predition),
                                    reduction_indices=[1])) #平方，对差值求和，求平均值
    #可视化的内容里增加scalars,以前叫events
    tf.summary.scalar('loss',loss)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss) #学习效率0.1，目的是减小误差



#最后一步

sess = tf.Session()


#所有的summary合并到一起
merged=tf.summary.merge_all()

writer = tf.summary.FileWriter('logs/',sess.graph)

# init = tf.initialize_all_variables()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i % 50 ==0:
        result = sess.run(merged,
                          feed_dict={xs:x_data,ys:y_data})
        #把记录的内容记录到summary中
        writer.add_summary(result,i)
#
#
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
#
# ax.scatter(x_data,y_data)
# plt.ion()  #不暂停
# plt.show()
#
#
# for i in range(1000):
#     sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
#     if i % 50 == 0:
#         #print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
#         try:
#             ax.lines.remove(lines[0])
#         except:
#             pass
#         predition_value = sess.run(predition,feed_dict={xs:x_data,ys:y_data})
#
#         lines = ax.plot(x_data,predition_value,'r-',lw = 5)
#
#         plt.pause(0.1)