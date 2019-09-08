import argparse
import distutils.util
import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('.//MNIST_data//', one_hot=True)


# 定义参数
# 真实图像的size
img_size = mnist.train.images[0].shape[0]
lbl_size = mnist.train.labels[0].shape[0]

# 传入给generator的噪声size
noise_size = 100
# 生成器隐层参数
g_units = 128
# 判别器隐层参数
d_units = 128

# leaky ReLU的参数
alpha = 0.01
# label smoothing
smooth = 0.1

# 训练迭代轮数
epochs = 20

#learning_rate
learning_rate = 0.001 
# batch_size
batch_size = 64

label_size = 10

# 抽取样本数
n_sample = 25
    
def inputs(noise_size, img_height, img_width, img_depth):
    """
    真实图像tensor与噪声图像tensor
    """

    real_img_label = tf.placeholder(tf.float32, [batch_size, label_size], name='real_img_label')
    
    real_img = tf.placeholder(tf.float32, [batch_size, img_height, img_width, img_depth], name='real_img')
    
    noise_img = tf.placeholder(tf.float32, [batch_size, noise_size], name='noise_img')
      

    return real_img_label, real_img, noise_img

                                      
def generator(digit, noise_img, out_dim, reuse=False, is_train=True):
    """
    生成器    
    noise_img: 生成器的输入
    out_dim: 生成器输出tensor的size，这里应该为28*28=784
    alpha: leaky ReLU系数
    """
    with tf.variable_scope("generator", reuse=reuse):

        batch_label = tf.reshape(digit, [-1, 1, 1, label_size])
        #noise_img = tf.concat([noise_img, digit], 1)

        # 100 x 1 to 4 x 4 x 1024
        # 全连接层
        layer1 = tf.layers.dense(noise_img, 4*4*1024)
        layer1 = tf.reshape(layer1, [-1, 4, 4, 1024])
        # batch normalization
        layer1 = tf.layers.batch_normalization(layer1, training=is_train)
        # Leaky ReLU
        layer1 = tf.maximum(alpha * layer1, layer1)
        # dropout
        layer1 = tf.nn.dropout(layer1, keep_prob=0.8)
        #layer1 = conv_cond_concat(layer1, batch_label)

        # 4 x 4 x 1024 to 7 x 7 x 512
        layer2 = tf.layers.conv2d_transpose(layer1, 512, 4, strides=1, padding='valid')
        layer2 = tf.layers.batch_normalization(layer2, training=is_train)
        layer2 = tf.maximum(alpha * layer2, layer2)
        layer2 = tf.nn.dropout(layer2, keep_prob=0.8)
        #layer2 = conv_cond_concat(layer2, batch_label)
        
        # 7 x 7 512 to 14 x 14 x 128
        layer3 = tf.layers.conv2d_transpose(layer2, 256, 3, strides=2, padding='same')
        layer3 = tf.layers.batch_normalization(layer3, training=is_train)
        layer3 = tf.maximum(alpha * layer3, layer3)
        layer3 = tf.nn.dropout(layer3, keep_prob=0.8)
        #layer3 = conv_cond_concat(layer3, batch_label)
        
        # logits & outputs
        # 14 x 14 x 128 to 28 x 28 x 1
        logits = tf.layers.conv2d_transpose(layer3, out_dim, 3, strides=2, padding='same')
        outputs = tf.tanh(logits)
        
        #return logits, outputs
        return outputs
        
        
def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)
  
def discriminator(digit, img, reuse=False, alpha=0.2):
    """
    判别器
    alpha: Leaky ReLU系数
    """
    
    with tf.variable_scope("discriminator", reuse=reuse):

        batch_label = tf.reshape(digit, [-1, 1, 1, label_size])
        x_shapes = img.get_shape()
        y_shapes = batch_label.get_shape()
        print(x_shapes)
        print(y_shapes)

        #img = conv_cond_concat(img, batch_label)
        
        # 28 x 28 x 1 to 14 x 14 x 128
        # 第一层不加入BN
        layer1 = tf.layers.conv2d(img, 128, 3, strides=2, padding='same')
        layer1 = tf.maximum(alpha * layer1, layer1)
        layer1 = tf.nn.dropout(layer1, keep_prob=0.8)
        #layer1 = conv_cond_concat(layer1, batch_label)
				
        # 14 x 14 x 128 to 7 x 7 x 256
        layer2 = tf.layers.conv2d(layer1, 256, 3, strides=2, padding='same')
        layer2 = tf.layers.batch_normalization(layer2, training=True)
        layer2 = tf.maximum(alpha * layer2, layer2)
        layer2 = tf.nn.dropout(layer2, keep_prob=0.8)

        
        # 7 x 7 x 256 to 4 x 4 x 512
        layer3 = tf.layers.conv2d(layer2, 512, 3, strides=2, padding='same')
        layer3 = tf.layers.batch_normalization(layer3, training=True)
        layer3 = tf.maximum(alpha * layer3, layer3)
        layer3 = tf.nn.dropout(layer3, keep_prob=0.8)
        
        # logits & outputs
        # 4 x 4 x 512 to 4*4*512 x 1
        flatten = tf.reshape(layer3, (-1, 4*4*512))
        logits = tf.layers.dense(flatten, 1)
        outputs = tf.sigmoid(logits)                
        
        return logits, outputs
        
def conv_cond_concat(x, y):
  """Concatenate conditioning vector on feature map axis."""
  x_shapes = x.get_shape()
  y_shapes = y.get_shape()

  return tf.concat([
    x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)
    
def plot_images(samples):
    fig, axes = plt.subplots(nrows=1, ncols=25, sharex=True, sharey=True, figsize=(50,2))
    for img, ax in zip(samples, axes):
        ax.imshow(img.reshape((28, 28)), cmap='Greys_r')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig.tight_layout(pad=0)
    plt.show()

  
def train(sess, flag, data_shape):

    #tf.reset_default_graph()

    real_img_label, real_img, noise_img = inputs(noise_size, data_shape[1], data_shape[2], data_shape[3])

    # generator
    g_outputs = generator(real_img_label, noise_img, data_shape[3], reuse=False, is_train=True)

    # discriminator
    d_logits_real, d_outputs_real = discriminator(real_img_label, real_img)
    d_logits_fake, d_outputs_fake = discriminator(real_img_label, g_outputs, reuse=True)

        
    # discriminator的loss
    # 识别真实图片
    # 真实图片往1方向优化，sigmoid_cross_entropy_with_logits和sigmoid_softmax_entropy_with_logits一样，只是二分类
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, 
                                                                        labels=tf.ones_like(d_outputs_real)))  
    # 识别生成的图片
    # 生成图片往0方向优化
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, 
                                                                        labels=tf.zeros_like(d_outputs_fake)))
    # 总体loss
    d_loss = tf.add(d_loss_real, d_loss_fake)

    # generator的loss
    # 生成图片尽量往1方向优化
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                    labels=tf.ones_like(d_outputs_fake)))


    # ## Optimizer
    # 
    # 由于我们在GAN里面一共训练了两个网络，所以需要分别定义优化函数。

    train_vars = tf.trainable_variables()

    # generator中的tensor
    g_vars = [var for var in train_vars if var.name.startswith("generator")]
    # discriminator中的tensor
    d_vars = [var for var in train_vars if var.name.startswith("discriminator")]

    # optimizer
    g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(g_loss, var_list=g_vars)
    d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(d_loss, var_list=d_vars)
    
    # 存储测试样例
    samples = []
    # 存储loss
    losses = []
    # 保存生成器变量
    saver = tf.train.Saver(var_list = g_vars)

       
    steps = 0

    if flag == 0:
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            for batch_i in range(mnist.train.num_examples//batch_size):
            	
                batch = mnist.train.next_batch(batch_size)

                batch_labels = batch[1]
          	    
                steps += 1
                batch_images = batch[0].reshape((batch_size, data_shape[1], data_shape[2], data_shape[3]))
                # 对图像像素进行scale，这是因为tanh输出的结果介于(-1,1),real和fake图片共享discriminator的参数
                # 把图片灰度0~1变成 -1 到1的值， 以适应generator输出的结果（-1,1）
                batch_images = batch_images*2 - 1

                
                # generator的输入噪声
                batch_noise = np.random.uniform(-1, 1, size=(batch_size, noise_size)).astype(np.float32)
                
                # Run optimizers
                _ = sess.run(d_train_opt, feed_dict={real_img_label: batch_labels, real_img: batch_images, noise_img: batch_noise})

                _ = sess.run(g_train_opt, feed_dict={real_img_label: batch_labels, noise_img: batch_noise})
                	
                # Run g_train_opt twice
                #_ = sess.run(g_train_opt, feed_dict={real_img_label: batch_labels, noise_img: batch_noise})
                
                
                if steps % 1 == 0:
		            # real img loss
                    train_loss_d_real = d_loss_real.eval({real_img_label: batch_labels,
                                                real_img: batch_images})
                    # fake img loss                            					            
                    train_loss_d_fake = d_loss_fake.eval({real_img_label: batch_labels,
                                                noise_img: batch_noise})
				            
                                                    	                	
                    train_loss_d = d_loss.eval({real_img_label: batch_labels,
                                                real_img: batch_images,
                                                noise_img: batch_noise})
                    train_loss_g = g_loss.eval({real_img_label: batch_labels, noise_img: batch_noise}) 

                   
                    print("Epoch {}/{}....".format(e+1, epochs), 
                          "Discriminator Loss: {:.4f}....".format(train_loss_d_real+train_loss_d_fake),
                          "Generator Loss: {:.4f}....". format(train_loss_g))

						
            sample_z = np.random.uniform(-1, 1, size=(batch_size , noise_size)).astype(np.float32)
            batch = mnist.train.next_batch(batch_size)
            sample_inputs = batch[0].reshape((batch_size, data_shape[1], data_shape[2], data_shape[3]))
            sample_inputs = sample_inputs*2 - 1
            sample_labels = batch[1]
      	
            # 每一轮结束计算loss
            # real img loss
            train_loss_d_real = sess.run(d_loss_real, 
                                        feed_dict = {real_img_label: sample_labels,
                                                    real_img: sample_inputs})
            # fake img loss
            train_loss_d_fake = sess.run(d_loss_fake, 
                                        feed_dict = {real_img_label: sample_labels,
                                                    noise_img: sample_z})
            train_loss_d = sess.run(d_loss, 
                                    feed_dict = {real_img_label: sample_labels,
                                                 real_img: sample_inputs, 
                                                 noise_img: sample_z})
            # generator loss
            train_loss_g = sess.run(g_loss, 
                                    feed_dict = {real_img_label: sample_labels, noise_img: sample_z})
            
                
            print("Epoch {}/{}...".format(e+1, epochs),
                "Discriminator Loss: {:.4f}(Real: {:.4f} + Fake: {:.4f})...".format(train_loss_d, train_loss_d_real, train_loss_d_fake),
                "Generator Loss: {:.4f}".format(train_loss_g))    
            # 记录各类loss值
            losses.append((train_loss_d, train_loss_d_real, train_loss_d_fake, train_loss_g))
            
            # 抽取样本后期进行观察
            gen_samples = sess.run( generator(real_img_label, noise_img, data_shape[3], reuse=True, is_train=False),
                                feed_dict={real_img_label: sample_labels, noise_img: sample_z})
            samples.append(gen_samples)
            
            # 存储checkpoints
            saver.save(sess, './checkpoints/generator.ckpt')
            

        # 将sample的生成数据记录下来
        with open('train_samples.pkl', 'wb') as f:
            pickle.dump(samples, f)

        with open('train_loss.pkl', 'wb') as f:
            pickle.dump(losses, f)

        fig, ax = plt.subplots(figsize=(20,7))
        losses = np.array(losses)
        plt.plot(losses.T[0], label='Discriminator Total Loss')
        plt.plot(losses.T[1], label='Discriminator Real Loss')
        plt.plot(losses.T[2], label='Discriminator Fake Loss')
        plt.plot(losses.T[3], label='Generator')
        plt.title("Training Losses")
        plt.legend()
        plt.show()
    else:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, './checkpoints/generator.ckpt')

        sample_noise = np.random.uniform(-1, 1, size=(batch_size, noise_size))
        digits = np.zeros((batch_size, label_size))
        for i in range(0, batch_size):
            j = np.random.randint(0, 9, 1)
            digits[i][j] = 1

        #print(digits)
        gen_samples = sess.run(generator(real_img_label, noise_img, data_shape[3], reuse=True, is_train=False),
                            feed_dict={real_img_label: digits, noise_img: sample_noise})    

        
        gen_samples = gen_samples[batch_size - 25 - 1:-1]
        _ = view_samples(gen_samples)

def view_samples(samples):

    fig, axes = plt.subplots(figsize=(7,7), nrows=5, ncols=5, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples): 
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((28,28)), cmap='Greys_r')

    plt.show()

    return fig, axes

 
def show_result():

    #显示保存的samples 图片
    # Load samples from generator taken while training
    with open('train_samples.pkl', 'rb') as f:
        samples = pickle.load(f)
    
 
    # 查看的轮次
    epoch_idx = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 19]
    show_imgs = []
    for i in epoch_idx:
        show_imgs.append(samples[i][batch_size - 25 - 1:-1])
        #show_imgs.append(samples[i])

    # 图片形状
    rows, cols = 10, 25
    fig, axes = plt.subplots(figsize=(30,12), nrows=rows, ncols=cols, sharex=True, sharey=True)

    for sample, ax_row in zip(show_imgs, axes):
        for img, ax in zip(sample, ax_row):
            #print(int(len(sample)/cols))
            #print(sample[0])
            ax.imshow(img.reshape((28,28)), cmap='Greys_r')
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
    plt.show()
       


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("-g", "--gpu_number", type=str, default="0")
    # 1:train 0:test
    parser.add_argument("-f", "--flag", type=float, default=1e-4)

    # Train Iteration
    parser.add_argument("-e" , "--epochs", type=int, default=300)
    
    # Train Parameter
    parser.add_argument("-b" , "--batch_size", type=int, default=64)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)


    args = parser.parse_args()

    epochs = args.epochs
    batch_size = args.batch_size
    #learning_rate = args.learning_rate
    

    gpu_number = args.gpu_number
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_number

    with tf.device('/gpu:{0}'.format(gpu_number)):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.90)
        config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
        

        with tf.Session(config=config) as sess:
            
            # TRAIN / TEST
            if args.flag == 0:
                train(sess, 0, [-1, 28, 28, 1])
            elif args.flag == 1:
                train(sess, 1, [-1, 28, 28, 1])
            else:
                show_result()
  

if __name__ == "__main__":
    main()

