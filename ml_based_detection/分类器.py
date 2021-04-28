def discriminator(self, name, inputs, reuse):
    l = tf.shape(inputs)[0]
    inputs = tf.reshape(inputs, (l,self.img_size,self.img_size,self.dim)) # 定义输入(50000,100,48)(50000,4800)

    with tf.variable_scope(name,reuse=reuse): #变量的命名方法
        out = []                     #这里的out是横横的拼接的，虽然每层的输出维度不同，但是每层输出的第一维都是50（batch size），所以可以横向拼接
        output = conv2d('d_con1',inputs,5, 64, stride=2, padding='SAME') #14*14 #卷积
        output1 = lrelu(self.bn('d_bn1',output))                                #激活
        out.append(output1)
        # output1 = tf.contrib.keras.layers.GaussianNoise

        output = conv2d('d_con2', output1, 3, 64*2, stride=2, padding='SAME')#7*7
        output2 = lrelu(self.bn('d_bn2', output))
        out.append(output2)

        output = conv2d('d_con3', output2, 3, 64*4, stride=1, padding='VALID')#5*5
        output3 = lrelu(self.bn('d_bn3', output))
        out.append(output3)

        output = conv2d('d_con4', output3, 3, 64*4, stride=2, padding='VALID')#2*2
        output4 = lrelu(self.bn('d_bn4', output))
        out.append(output4)

        output = tf.reshape(output4, [l, 2*2*64*4])# 2*2*64*4
        output = fc('d_fc', output, self.num_class)
        # output = tf.nn.softmax(output)
        return output, out #output (50.3) ,out ()