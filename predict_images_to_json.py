"""
This file is designed for prediction of bounding boxes for a given video and output individual frames after drawing rectangles on the predicted ROIs. It creates a json file with all the detected bounding box coordinates and confidence scores in each frame of the video input.
"""

import tensorflow as tf
import os, json, subprocess
import argparse
import cv2
import numpy
from itertools import count
from optparse import OptionParser

from scipy.misc import imread, imresize
from PIL import Image, ImageDraw

from train import build_forward
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

    #manual fix
    H['grid_height'] = 15
    H['grid_width'] = 20

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
    coord = []
    r = []
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
    if not r: 
        coord = [0,0,0,0,0]
    else:
        coord = [r.x1, r.x2, r.y1, r.y2, r.score]
  
    return result, coord


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
    global outfile
    one_decimal = "{0:0.1f}"
    json_images = []
    directory =  os.path.dirname(videoPath)


    img_list = os.listdir(videoPath)
    img_list = [os.path.join(videoPath, img) for img in img_list]
    
    i=0
    for img in img_list:
        rects = []
        frame = cv2.imread(img) 
        #print(i) 
        ret=True
        if ret==True:          
            data_list, coor = hot_predict(frame, init_params, False)

            #We just need to coordinated. No need to draw anything new
            #pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            #drawn_frame = draw_results(pil_img, data_list)
            #new_frame = cv2.cvtColor(numpy.array(drawn_frame), cv2.COLOR_RGB2BGR)
            #cv2.imwrite(os.path.join(directory, 'pict{:>05}.png'.format(i)) , new_frame)
            
            x1 = float(one_decimal.format(coor[0]))
            x2 = float(one_decimal.format(coor[1]))
            y1 = float(one_decimal.format(coor[2]))
            y2 = float(one_decimal.format(coor[3]))

        #enforce x1,y1 = top left, x2,y2 = bottom right

            tlx = min(x1,x2)
            tly = min(y1,y2)
            brx = max(x1,x2)
            bry = max(y1,y2)
            
            bbox = []
            bbox = dict([("x1",tlx),("y1",tly),("x2",brx),("y2",bry),("score",coor[4])])
            rects.append(bbox)  
            json_image = dict([("frame_number",'pict{:>05}.png'.format(i)),("coordinates",rects)])
            json_images.append(json_image)
            #outfile.write(json.dumps(json_image, indent = 1))                      
            i += 1
            if i%10000==0:
                print(i)

        else:
            break
        
    outfile.write(json.dumps(json_images, indent = 1))


def run_CNN(folder, weights, hypes, outfile_name, gpu=0, tau=0.25, min_conf=0.95):

    options = {}
    options['gpu'] = gpu
    options['tau'] = tau
    options['min_conf'] = min_conf

    init_params = initialize(weights, hypes, options)
    
    global outfile
    outfile = open(outfile_name, 'w')
    
    processVideo(folder, init_params)

    outfile.close()

if __name__ == '__main__':

    #Parse parameters inputed by user
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder')
    parser.add_argument('--hypes')
    parser.add_argument('--weights')
    parser.add_argument('--gpu')
    args = parser.parse_args()

    #Capture arguments
    folder = args.folder
    hypes = args.hypes
    weights = args.weights
    gpu = args.gpu

    #Path to save coordinates to
    outpath = os.path.join(os.getcwd(),folder,'_coordinates.json')

    print('Saving coordinates to: ', outpath)
    

    run_CNN(folder, weights, hypes, outpath, gpu=gpu)
