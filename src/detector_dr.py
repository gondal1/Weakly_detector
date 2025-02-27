import tensorflow as tf
import numpy as np
import cPickle
#import ipdb
class Detector():
    def __init__(self, n_labels, weight_file_path=None):
        self.image_mean = [108.64628601, 75.86886597, 54.34005737]
        self.n_labels = n_labels
        
        if weight_file_path is not None:
            with open(weight_file_path) as f:
                self.pretrained_weights = cPickle.load(f)
                
                
    # Accessing weights and biases according to layer names, check out the layers name 
    def get_weight( self, layer_name):
        layer = self.pretrained_weights[layer_name]
        return layer[0]

    def get_bias( self, layer_name ):
        layer = self.pretrained_weights[layer_name]
        return layer[1]

    def get_conv_weight( self, name ):
        f = self.get_weight( name )
        return f.transpose(( 2,3,1,0 ))   # We have to do this since weights are stored in caffe format

    # This could be for using pre stored weights
    def conv_layer( self, bottom, name ):
        with tf.variable_scope(name) as scope:

            w = self.get_conv_weight(name)
            b = self.get_bias(name)
            # tf.get_variable(name, shape, initializer)
            # Over here because of tf.variable_scope, names of varibales will be "name/W" and "name/b"
            # with tf.get_varible there are 2 options 
            # 1. you can create a new variable like the one with tf.Variable
            # 2. you can use the previous values, it will search for the matching variable name in the graph for e.g. "conv1/W" and
            # will assign it that value
            conv_weights = tf.get_variable(
                    "W",
                    shape=w.shape,
                    initializer=tf.constant_initializer(w)
                    )
            conv_biases = tf.get_variable(
                    "b",
                    shape=b.shape,
                    initializer=tf.constant_initializer(b)
                    )

            conv = tf.nn.conv2d( bottom, conv_weights, [1,1,1,1], padding='SAME')
            bias = tf.nn.bias_add( conv, conv_biases )
            relu = tf.nn.relu( bias, name=name )

        return relu

    
    def new_conv_layer( self, bottom, filter_shape, name ):
        with tf.variable_scope(name) as scope:
            w = tf.get_variable(
                    "W",
                    shape=filter_shape,
                    initializer=tf.random_normal_initializer(0., 0.01))
            b = tf.get_variable(
                    "b",
                    shape=filter_shape[-1],
                    initializer=tf.constant_initializer(0.))

            conv = tf.nn.conv2d( bottom, w, [1,1,1,1], padding='SAME')
            bias = tf.nn.bias_add(conv, b)

        return bias #relu

    def fc_layer(self, bottom, name, create=False):
        
        # Exactly like the reshaping we do before entering a fully connected layer
        shape = bottom.get_shape().as_list()
        dim = np.prod( shape[1:] )
        x = tf.reshape(bottom, [-1, dim])

        cw = self.get_weight(name)
        b = self.get_bias(name)

        if name == "fc6":
            cw = cw.reshape((4096, 512, 7,7))   # It supposes that until this depth in our network we have a resolution of 7x7 f.maps
            cw = cw.transpose((2,3,1,0))
            cw = cw.reshape((25088,4096))
        else:
            cw = cw.transpose((1,0))

        with tf.variable_scope(name, reuse=True) as scope:
            cw = tf.get_variable(
                    "W",
                    shape=cw.shape,
                    initializer=tf.constant_initializer(cw))
            b = tf.get_variable(
                    "b",
                    shape=b.shape,
                    initializer=tf.constant_initializer(b))

            fc = tf.nn.bias_add( tf.matmul( x, cw ), b, name=scope)

        return fc

    def new_fc_layer( self, bottom, input_size, output_size, name ):
        shape = bottom.get_shape().to_list()
        dim = np.prod( shape[1:] )
        x = tf.reshape( bottom, [-1, dim])

        with tf.variable_scope(name, reuse=True) as scope:
            w = tf.get_variable(
                    "W",
                    shape=[input_size, output_size],
                    initializer=tf.random_normal_initializer(0., 0.01))
            b = tf.get_variable(
                    "b",
                    shape=[output_size],
                    initializer=tf.constant_initializer(0.))
            fc = tf.nn.bias_add( tf.matmul(x, w), b, name=scope)

        return fc

    def inference( self, rgb, dropout, train=False ):
        rgb *= 255.
        r, g, b = tf.split(3, 3, rgb)
        bgr = tf.concat(3,
            [
                b-self.image_mean[0],
                g-self.image_mean[1],
                r-self.image_mean[2]
            ])

        relu1_1 = self.new_conv_layer( bgr,[3,3,3,64], "conv1_1" )
        #self.activation_summary(relu1_1)
        relu1_2 = self.new_conv_layer( relu1_1, [3,3,64,64], "conv1_2" )
        #self.activation_summary(relu1_2)
        pool1 = tf.nn.max_pool(relu1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                         padding='SAME', name='pool1')
        
        relu2_1 = self.new_conv_layer(pool1, [3,3,64,128],"conv2_1")
        #self.activation_summary(relu2_1)
        relu2_2 = self.new_conv_layer(relu2_1, [3,3,128,128],"conv2_2")
        #self.activation_summary(relu2_2)
        pool2 = tf.nn.max_pool(relu2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool2')

        relu3_1 = self.new_conv_layer( pool2, [3,3,128,256], "conv3_1")
        #self.activation_summary(relu3_1)
        relu3_2 = self.new_conv_layer( relu3_1, [3,3,256,256], "conv3_2")
        #self.activation_summary(relu3_2)
        relu3_3 = self.new_conv_layer( relu3_2, [3,3,256,256],"conv3_3")
        #self.activation_summary(relu3_3)
        pool3 = tf.nn.max_pool(relu3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool3')

        relu4_1 = self.new_conv_layer( pool3, [3,3,256,512], "conv4_1")
        #self.activation_summary(relu4_1)
        relu4_2 = self.new_conv_layer( relu4_1, [3,3,512,512], "conv4_2")
        #self.activation_summary(relu4_2)
        relu4_3 = self.new_conv_layer( relu4_2, [3,3,512,512],"conv4_3")
        #self.activation_summary(relu4_3)
        pool4 = tf.nn.max_pool(relu4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool4')

        relu5_1 = self.new_conv_layer( pool4, [3,3,512,512],"conv5_1")
        #self.activation_summary(relu5_1)
        relu5_2 = self.new_conv_layer( relu5_1, [3,3,512,512],"conv5_2")
        #self.activation_summary(relu5_2)
        relu5_3 = self.new_conv_layer( relu5_2, [3,3,512,512],"conv5_3")
        #self.activation_summary(relu5_3)
        
        # Here is introduction of new conv layer in conventional VGG Net, because we want to increase the resolution
        # of feature maps here, therefore from 512 to 1024
        
        conv6 = self.new_conv_layer( relu5_3, [3,3,512,1024], "conv6")
        #self.activation_summary(conv6)
        gap = tf.reduce_mean( conv6, [1,2] )    # Computing global mean in 2D
        
        dropout = tf.nn.dropout(gap, dropout) # For testing purpose, don't use dropout
        
        # this GAP is layer is for inference as weighted sum of the GAP values give us the output, here weights (gap_w)
        # play an important role
        
        with tf.variable_scope("GAP"):
            gap_w = tf.get_variable(
                    "W",
                    shape=[1024, self.n_labels],
                    initializer=tf.random_normal_initializer(0., 0.01))

        output = tf.matmul( dropout, gap_w)

        return pool1, pool2, pool3, pool4, relu5_3, conv6, gap, output

    def get_classmap(self, label, conv6):
        # upsample the weighted sum of filter maps from the last conv layer and upsample them to input image size using bilinear
        # interpolation
        conv6_resized = tf.image.resize_bilinear( conv6, [224, 224] )
        
        with tf.variable_scope("GAP", reuse=True):  # Reusing the same "gap_w" from variable_scope("GAP")
            label_w = tf.gather(tf.transpose(tf.get_variable("W")), label)  # label_w stands for "label weights"
            label_w = tf.reshape( label_w, [-1, 1024, 1] ) # [batch_size, 1024, 1]

        conv6_resized = tf.reshape(conv6_resized, [-1, 224*224, 1024]) # [batch_size, 224*224, 1024]

        classmap = tf.batch_matmul( conv6_resized, label_w )
        classmap = tf.reshape( classmap, [-1, 224,224] )
        return classmap






