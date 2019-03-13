"""
This file is designed for prediction of bounding boxes for a given video and output individual frames after drawing rectangles on the predicted ROIs. It creates a folder with '_labelled' postscript, and adds all the images with bounding boxes into this folder.
"""

import tensorflow as tf
import os, json, subprocess
import cv2
import numpy
from itertools import count
from optparse import OptionParser

from scipy.misc import imread, imresize
from PIL import Image, ImageDraw

from train3 import build_forward
from utils.annolist import AnnotationLib as al
from utils.train_utils import add_rectangles, rescale_boxes


def initialize(weights_path, hypes_path, options=None):
    """Initialize prediction process.
     
    All long running operations like TensorFlow session start and weights loading are made here.
     
    Args:
        weights_path (string): The path to the model weights file. 
        hypes_path (string): The path to the hyperparameters file. 
        options (dict): The options dictionary with parameters for the initialization process.
    Returns (dict):
        The dict object which contains `sess` - TensorFlow session, `pred_boxes` - predicted boxes Tensor, 
          `pred_confidences` - predicted confidences Tensor, `x_in` - input image Tensor, 
          `hypes` - hyperparametets dictionary.
    """

    H = prepare_options(hypes_path, options)

    tf.reset_default_graph()
    x_in = tf.placeholder(tf.float32, name='x_in', shape=[H['image_height'], H['image_width'], 3])
    if H['use_rezoom']:
        pred_boxes, pred_logits, pred_confidences, pred_confs_deltas, pred_boxes_deltas \
            = build_forward(H, tf.expand_dims(x_in, 0), 'test', reuse=None)
        grid_area = H['grid_height'] * H['grid_width']
        pred_confidences = tf.reshape(
            tf.nn.softmax(tf.reshape(pred_confs_deltas, [grid_area * H['rnn_len'], H['num_classes']])),
            [grid_area, H['rnn_len'], H['num_classes']])
        if H['reregress']:
            pred_boxes = pred_boxes + pred_boxes_deltas
    else:
        pred_boxes, pred_logits, pred_confidences = build_forward(H, tf.expand_dims(x_in, 0), 'test', reuse=None)

    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, weights_path)
    return {'sess': sess, 'pred_boxes': pred_boxes, 'pred_confidences': pred_confidences, 'x_in': x_in, 'hypes': H}


def hot_predict(image_path, parameters, to_json=True):
    """Makes predictions when all long running preparation operations are made. 
    
    Args:
        image_path (string): The path to the source image. 
        parameters (dict): The parameters produced by :func:`initialize`.
    Returns (Annotation):
        The annotation for the source image.
    """

    H = parameters['hypes']
    # The default options for prediction of bounding boxes.
    options = H['evaluate']
    if 'pred_options' in parameters:
        # The new options for prediction of bounding boxes
        for key, val in parameters['pred_options'].items():
            options[key] = val

    # predict
    img = imresize(image_path, (H['image_height'], H['image_width']), interp='cubic')
    (np_pred_boxes, np_pred_confidences) = parameters['sess'].\
        run([parameters['pred_boxes'], parameters['pred_confidences']], feed_dict={parameters['x_in']: img})
    pred_anno = al.Annotation()
    pred_anno.imageName = image_path
    _, rects = add_rectangles(H, [img], np_pred_confidences, np_pred_boxes, use_stitching=True,
                              rnn_len=H['rnn_len'], min_conf=options['min_conf'], tau=options['tau'],
                              show_suppressed=False)

    pred_anno.rects = [r for r in rects if r.x1 < r.x2 and r.y1 < r.y2]
    pred_anno = rescale_boxes((H['image_height'], H['image_width']), pred_anno, image_path.shape[0], image_path.shape[1])
    result = [r.writeJSON() for r in pred_anno if r.score > options['min_conf']] if to_json else pred_anno
    return result


def prepare_options(hypes_path='hypes.json', options=None):
    """Sets parameters of the prediction process. If evaluate options provided partially, it'll merge them. 
    The priority is given to options argument to overwrite the same obtained from the hyperparameters file.
        
    Args:
        hypes_path (string): The path to model hyperparameters file.
        options (dict): The command line options to set before start predictions.
    Returns (dict):
        The model hyperparameters dictionary.
    """

    with open(hypes_path, 'r') as f:
        H = json.load(f)

    # set default options values if they were not provided
    if options is None:
        if 'evaluate' in H:
            options = H['evaluate']
        else:
            print ('Evaluate parameters were not found! You can provide them through hyperparameters json file '
                   'or hot_predict options parameter.')
            return None
    else:
        if 'evaluate' not in H:
            H['evaluate'] = {}
        # merge options argument into evaluate options from hyperparameters file
        for key, val in options.items():
            H['evaluate'][key] = val

    os.environ['CUDA_VISIBLE_DEVICES'] = str(H['evaluate']['gpu'])
    return H


def draw_results(image_path, anno):
    """This function draws rectangle around the ROI."""
    # draw
    new_img = image_path
    d = ImageDraw.Draw(new_img)
    rects = anno['rects'] if type(anno) is dict else anno.rects
    for r in rects:
        d.rectangle([r.left(), r.top(), r.right(), r.bottom()], outline=(255, 0, 0))
    return new_img


def processVideo(videoPath, init_params):
    f_name = os.path.dirname(videoPath)
    b_name = os.path.basename(videoPath)
    [b_name1,bname2] = os.path.splitext(b_name)
    directory = f_name+'/'+b_name1+'_labelled'
    if not os.path.exists(directory):
        os.makedirs(directory)
    capture = cv2.VideoCapture(videoPath)
    i = 0
    while (capture.isOpened()):
        ret, frame = capture.read()
        if ret==True:          
            data_list = hot_predict(frame, init_params, False)
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            drawn_frame = draw_results(pil_img, data_list)
            new_frame = cv2.cvtColor(numpy.array(drawn_frame), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(directory, 'pict{:>05}.png'.format(i)) , new_frame)
            i += 1
        else:
            break
    capture.release()
    cv2.destroyAllWindows()


def main():
    parser = OptionParser(usage='usage: %prog [options] <image> <weights> <hypes>')
    parser.add_option('--gpu', action='store', type='int', default=0)
    parser.add_option('--tau', action='store', type='float',  default=0.25)
    parser.add_option('--min_conf', action='store', type='float', default=0.85)

    (options, args) = parser.parse_args()
    if len(args) < 3:
        print ('Provide image, weights and hypes paths')
        return

    init_params = initialize(args[1], args[2], options.__dict__)
    processVideo(args[0], init_params)

if __name__ == '__main__':
    main()
