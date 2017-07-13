from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import logging

import os
import time
#import base64
import flask
import werkzeug
import datetime
import numpy as np
#from PIL import Image
#import cStringIO as StringIO
#import io

from nets import inception_resnet_v2
from preprocessing import inception_preprocessing
import tensorflow as tf

slim = tf.contrib.slim


UPLOAD_FOLDER = '/tmp/LT_Imagenet_Flask/demos_uploads'
#UPLOAD_FOLDER = 'C:\\tmp\\LT_Imagenet_Flask\\demos_uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


app = flask.Flask(__name__, static_folder=UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

@app.route('/')
def classify_index():
    return flask.render_template('index.html', has_result=False)



@app.route('/classify_upload', methods=['POST'])
def classify_upload():
    try:
        # We will save the file to disk for possible data collection.
        imagefile = flask.request.files['imagefile']
        if imagefile and allowed_file(imagefile.filename):
            filename = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-') + '_' + werkzeug.secure_filename(imagefile.filename)
            imagefile.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        else:
            return flask.render_template('index.html', has_result=True,
                                          result=(False, 'Allowed image extension: .png, .jpg, .jpeg'))

        

    except Exception as err:
        return flask.render_template('index.html', has_result=True,
                                      result=(False, 'Cannot open uploaded image.'))

    image_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    #image = open_oriented_im(image_file)
    names, probs, time_cost = app.clf.classify_IR(image_file)

    str_probs = []
    for p in probs:
        str_probs.append('%0.2f' %p)
    return flask.render_template('index.html', has_result=True,
                                  result=[True, zip(names, str_probs), '%.3f' % time_cost],
                                  imagesrc=flask.url_for('static', filename=filename))



class ImagenetClassifier(object):
    def __init__(self):
        self.sessIR = tf.Session()
        self.labels_names = self.imagenet_labels()
        self.image_sizeIR = inception_resnet_v2.inception_resnet_v2.default_image_size

        # Define Inception Resnet v2 neural network
        self.processed_imagesIR = tf.placeholder(tf.float32, shape=(None, self.image_sizeIR, self.image_sizeIR, 3))
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logitsIR, _ = inception_resnet_v2.inception_resnet_v2(self.processed_imagesIR, num_classes=1001, is_training=False)
        self.probabilitiesIR = tf.nn.softmax(logitsIR)

        # Load saved checkpoint for Resnet v2 neural network
        checkpoints_dir = './checkpoints'
        init_fn = slim.assign_from_checkpoint_fn(
            os.path.join(checkpoints_dir, 'inception_resnet_v2_2016_08_30.ckpt'),
            slim.get_model_variables('InceptionResnetV2'))
        init_fn(self.sessIR)
        
    
    def classify_IR(self, sample_images):
        try:
            # Prerape the image for evalution
            start_time = time.time()
            image_rawIR = tf.image.decode_image(tf.read_file(sample_images), channels=3)
            imageIR = inception_preprocessing.preprocess_image(image_rawIR, self.image_sizeIR, self.image_sizeIR, is_training=False)
            imageIR  = tf.expand_dims(imageIR, 0)
            imageIR = self.sessIR.run(imageIR)
            # Run NN
            predsIR = self.sessIR.run(self.probabilitiesIR, feed_dict={self.processed_imagesIR: imageIR})
            end_time = time.time()

            # Prepare results for web-page
            probsIR = predsIR[0, 0:]
            sorted_indsIR = [i[0] for i in sorted(enumerate(-probsIR), key=lambda x:x[1])]
            class_name = []
            probabilities_for_class = []
            for i in range(3):
                indexIR = sorted_indsIR[i]
                class_name.append(self.labels_names[indexIR])
                probabilities_for_class.append(probsIR[indexIR])
            return [class_name, probabilities_for_class, end_time - start_time]
        except Exception as err:
            return None
    
    def imagenet_labels(self):
        synset_list = [s.strip() for s in open('./imagenet/imagenet_lsvrc_2015_synsets.txt').readlines()]
        num_synsets_in_ilsvrc = len(synset_list)
        assert num_synsets_in_ilsvrc == 1000

        synset_to_human_list = open('./imagenet/imagenet_metadata.txt').readlines()
        num_synsets_in_all_imagenet = len(synset_to_human_list)
        assert num_synsets_in_all_imagenet == 21842

        synset_to_human = {}
        for s in synset_to_human_list:
            parts = s.strip().split('\t')
            assert len(parts) == 2
            synset = parts[0]
            human = parts[1]
            synset_to_human[synset] = human

        label_index = 1
        labels_to_names = {0: 'background'}
        for synset in synset_list:
            name = synset_to_human[synset]
            labels_to_names[label_index] = name
            label_index += 1

        return labels_to_names


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if __name__ == '__main__':
    # Add parsing command line

    app.clf = ImagenetClassifier()
    
    # Test service on demo image????

    #app.run(debug=True, threaded=True)
    app.run()
    #app.run(debug=True)
    #app.run(host='0.0.0.0') # Public server
    #app.run(threaded=True)