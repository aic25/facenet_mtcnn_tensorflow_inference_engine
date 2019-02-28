import tensorflow as tf
import os
from mtcnn.mtcnn import MTCNN
import cv2
import facenet
import logging
import sys
logging.basicConfig(
        stream=sys.stdout,
         format='%(asctime)s %(levelname)-8s %(message)s',
         level=logging.DEBUG,
         datefmt='%Y-%m-%d %H:%M:%S')

model_dir = 'model/20180402-114759'
meta_file = 'model-20180402-114759.meta'
ckpt_file = 'model-20180402-114759.ckpt-275'

mtcnn_dir = 'mtcnn_model/all_in_one'
mtcnn_meta = 'mtcnn-3000000.meta'
mtcnn_ckpt = 'mtcnn-3000000'

image_size = 160

img_path = 'face6.jpg'
detector = MTCNN()

with tf.Graph().as_default():
    with tf.Session() as sess:
        print('start facenet session')

        # Restore the graph
        saver = tf.train.import_meta_graph(os.path.join(model_dir,meta_file))
        # Load weight
        saver.restore(sess,os.path.join(model_dir,ckpt_file))

        img = cv2.imread(img_path)

        logging.debug('start process')

        print(detector.detect_faces(img))

        for face in detector.detect_faces(img):
            [x,y,w,h] = face['box']
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0,),2)
            cv2.putText(img,"johny johny", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
            cv2.imshow('frame',img)
            cv2.waitKey(0)
            crop_img = img[y:y + h, x:x + w]
            resize_img = facenet.prewhiten(cv2.resize(crop_img,(image_size,image_size)))
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            # images_placeholder = tf.image.resize_images(images_placeholder, (image_size, image_size))
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embeddings = tf.get_default_graph().get_tensor_by_name('embeddings:0')
            logging.debug(f'resize_img: {resize_img.shape}')
            feed_dict = {images_placeholder: resize_img.reshape(-1,image_size,image_size,3),phase_train_placeholder:False}
            result = sess.run(embeddings,feed_dict=feed_dict)
            logging.debug(f'embeddings: {result.shape}')













