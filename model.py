from vgg import *
from data import *

"""
利用定义好的vgg fine-tuning
"""

class Model(object):

    def __init__(self):
        self.vgg = VGG16()
        self.input_x = self.vgg.input
        self.input_y = tf.placeholder(tf.int64,[None,])
        self.output_num = 4
        self.learning_rate = 1e-4
        self.epoch = 50
        self.logits, self.softmax = self.net()
        self.loss, self.optimizer, self.accuracy = self.train_op()

    def net(self):
        vgg_output = self.vgg.conv5_3 #只调用了vgg的卷积层
        x = tf.layers.flatten(vgg_output)
        x = tf.layers.dense(x,512,activation=tf.nn.relu)
        x = tf.layers.dropout(x,0.5)
        x = tf.layers.dense(x,256,activation=tf.nn.relu)
        x = tf.layers.dropout(x,0.5)
        logits = tf.layers.dense(x,self.output_num)
        softmax = tf.nn.softmax(logits)
        return logits,softmax

    def train_op(self):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,labels=self.input_y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(self.input_y,tf.argmax(self.logits,-1)),tf.float32))
        return loss,optimizer,accuracy

    def train(self,sess):
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        iter,batch_num= 0, 0
        train_avg_acc,train_avg_loss = 0, 0
        for epoch in range(self.epoch):
            data = train_data('E:\\train_data\\train',batch_size=8,resize_shape=(224,224))
            for train_x, train_y in data:
                _, train_loss, train_acc = sess.run([self.optimizer,self.loss,self.accuracy],feed_dict = {self.input_x:train_x,self.input_y:train_y})
                train_avg_loss+=train_loss
                train_avg_acc+=train_acc
                if iter%10==0:
                    print('epoch:',epoch,',iter:',iter,',train loss:',train_avg_loss/10,',train acc:',train_avg_acc/10)
                    train_avg_acc,train_avg_loss = 0,0
                if iter%50==0:
                     test_x, test_y, num_batch = test_data('E:\\train_data\\test',batch_size=16,batch_num=batch_num,reshap_size=(224,224))
                     test_loss, test_acc = sess.run([self.loss,self.accuracy],feed_dict={self.input_x:test_x,self.input_y:test_y})
                     print('epoch:',epoch,',iter:',iter,',-->test loss:',test_loss,'-->test acc:',test_acc)
                     batch_num = batch_num+1
                     if batch_num == num_batch:
                         batch_num = 0
                if iter%100==0 and iter!=0:
                    self.learning_rate = 0.5*self.learning_rate
                if iter!=0 and iter%200==0:
                    saver.save(sess=sess,save_path='./save/save.ckpt'+str(iter))
                iter = iter+1

def Train():
    sess = tf.Session()
    model = Model()
    model.train(sess)
    sess.close()

if __name__ == '__main__':
    Train()
