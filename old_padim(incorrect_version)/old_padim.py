# -*- coding: utf-8 -*-
# """
# padim.py
#   Original Author: 2021.05.02. @chanwoo.park
#   Edited by MarShao0124 2025.03.06
#   Edit: corrected the code(function padim) to the algorithm applied in the paper, and improved
#         the accuracy compared with the original code. 
#   PaDiM algorithm
#   Reference:
#       Defard, Thomas, et al. "PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization."
#       arXiv preprint arXiv:2011.08785 (2020).
# """

############
#   IMPORT #
############
# 1. Built-in modules
import os, sys, time

# 2. Third-party modules
import numpy as np
import tensorflow as tf
import sklearn.metrics as metrics

from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import mahalanobis


# 3. Own module
from old_PaDiM_utils import embedding_concat, plot_fig, draw_auc, draw_precision_recall, save_result
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code'))
from old_mvtec_loader import MVTecADLoader
from old_VisA_loader import VisALoader


################
#   Definition #
################
def embedding_net(net_type='res'):
    input_tensor = tf.keras.layers.Input([224, 224, 3], dtype=tf.float32)

    if net_type == 'res':
        # resnet 50v2
        x = tf.keras.applications.resnet_v2.preprocess_input(input_tensor)
        model = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet', input_tensor=x, pooling=None)

        layer1 = model.get_layer(name='conv3_block1_preact_relu').output
        layer2 = model.get_layer(name='conv4_block1_preact_relu').output
        layer3 = model.get_layer(name='conv5_block1_preact_relu').output

    elif net_type == 'eff':
        # efficient net B5
        x = tf.keras.applications.efficientnet.preprocess_input(input_tensor)
        model = tf.keras.applications.EfficientNetB5(include_top=False, weights='imagenet', input_tensor=x,
                                                     pooling=None)

        layer1 = model.get_layer(name='block2a_activation').output
        layer2 = model.get_layer(name='block3a_activation').output
        layer3 = model.get_layer(name='block4a_activation').output

    else:
        raise Exception("[NotAllowedNetType] network type is not allowed ")

    model.trainable = False
    # model.summary(line_length=100)
    shape = (layer3.shape[1], layer3.shape[2], layer1.shape[3] + layer2.shape[3] + layer3.shape[3])

    return tf.keras.Model(model.input, outputs=[layer1, layer2, layer3]), shape


def padim(category, batch_size, rd, net_type='eff', is_plot=False, data='mvtec'):
    if data == 'mvtec':
        loader = MVTecADLoader()
    elif data == 'visa':
        loader = VisALoader()
    else:
        raise Exception("[NotAllowedDataset] dataset is not allowed ")
    
    
    loader.load(category=category, repeat=1, max_rot=10)

    train_set = loader.train.batch(batch_size=batch_size, drop_remainder=True).shuffle(buffer_size=loader.num_train,
                                                                                       reshuffle_each_iteration=True)
    test_set = loader.test.batch(batch_size=1, drop_remainder=False)

    net, _shape = embedding_net(net_type=net_type)
    h, w, c = _shape  # height and width of layer3, channel sum of layer 1, 2, and 3, and randomly sampled dimension

    # concatenate patch of layer1, layer2, and layer3 of the pre-trained network
    out = []
    for x, _, _ in train_set:
        l1, l2, l3 = net(x)
        _out = embedding_concat(l1,l2,l3) # (bs, h * w, c1+c2+c3)
        out.append(_out.numpy())

    out = np.concatenate(out, axis=0)  # or np.stack(out, axis=0)
    out = out.transpose(0, 2, 1)  # (batch_size, c1+c2+c3, h*w)

    # calculate multivariate Gaussian distribution.
    # RD: random dimension selecting
    c = out.shape[-2]
    rd_indices = np.random.choice(c, size=rd, replace=False)
    out = out[:, rd_indices, :]  # shape: (batch_size, rd, h*w)


    # Compute mean and covariance
    mu = np.mean(out, axis=0)  # shape: (rd, h*w)
    cov = np.zeros((rd, rd, h*w))
    print(cov.shape)
    identity = np.identity(rd)
    for idx in range(h*w):
        cov[:, :, idx] = np.cov(out[:, :, idx], rowvar=False) + 0.01 * identity

    train_outputs = [mu, cov]

    ################
    #   TEST DATA  #
    ################
    out, gt_list, gt_mask, batch_size, test_imgs = [], [], [], 1, []

    start_time = time.time()

    #  x - data |   y - mask    |   z - binary label
    for x, y, z in test_set:
        test_imgs.append(x.numpy())
        gt_list.append(z.numpy())
        gt_mask.append(y.numpy())

        l1, l2, l3 = net(x)
        _out = embedding_concat(l1,l2,l3) # (bs, h * w, c1+c2+c3)
        out.append(_out.numpy()) 

    # calculate multivariate Gaussian distribution. skip random dimension selecting
    out = np.concatenate(out, axis=0)
    gt_list = np.concatenate(gt_list, axis=0)
    out = np.transpose(out, axes=[0, 2, 1]) # (bs, c1+c2+c3, h*w)

    # RD
    tmp = tf.unstack(out, axis=0)
    _tmp = []
    for tensor in tmp:
        _tmp.append(tf.gather(tensor, rd_indices))
    out = tf.stack(_tmp, axis=0)

    b, _, _ = out.shape

    dist_list = []
    for idx in range(h*w):
        mu = train_outputs[0][:, idx]
        cov_inv = np.linalg.inv(train_outputs[1][:, :, idx])
        dist = [mahalanobis(sample[:, idx], mu, cov_inv) for sample in out]
        dist_list.append(dist)

    dist_list = np.reshape(np.transpose(np.asarray(dist_list), axes=[1, 0]), (b, h, w))

    end_time = time.time()
    inference_time = (end_time - start_time)/len(test_set)
    ################
    #   DATA Level #
    ################
    # upsample
    score_map = tf.squeeze(tf.image.resize(np.expand_dims(dist_list, -1), size=[h, w])).numpy()

    for i in range(score_map.shape[0]):
        score_map[i] = gaussian_filter(score_map[i], sigma=4)

    # Normalization
    max_score = score_map.max()
    min_score = score_map.min()
    scores = (score_map - min_score) / (max_score - min_score)
    scores = -scores

    # calculate image-level ROC AUC score
    img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)

    gt_list = np.asarray(gt_list)
    img_roc_auc = metrics.roc_auc_score(gt_list, img_scores)

    if is_plot is True:
        fpr, tpr, _ = metrics.roc_curve(gt_list, img_scores)
        precision, recall, _ = metrics.precision_recall_curve(gt_list, img_scores)

        save_dir = os.path.join(os.path.dirname(__file__), 'img_'+net_type,data,'img_'+category)
        if os.path.isdir(save_dir) is False:
            os.makedirs(save_dir)
        draw_auc(fpr, tpr, img_roc_auc, os.path.join(save_dir, 'AUROC-{}.png'.format(category)))
        base_line = np.sum(gt_list) / len(gt_list)
        f1 = draw_precision_recall(precision, recall, base_line, os.path.join(os.path.join(save_dir,
                                                                                      'PR-{}.png'.format(category))))

    #################
    #   PATCH Level #
    #################
    # upsample
    score_map = tf.squeeze(tf.image.resize(np.expand_dims(dist_list, -1), size=[224, 224])).numpy()

    for i in range(score_map.shape[0]):
        score_map[i] = gaussian_filter(score_map[i], sigma=4)

    # Normalization
    max_score = score_map.max()
    min_score = score_map.min()
    scores = (score_map - min_score) / (max_score - min_score)
    # Note that Binary mask indicates 0 for good and 1 for anomaly. It is opposite from our setting.
    # scores = -scores

    # calculate per-pixel level ROCAUC
    gt_mask = np.asarray(gt_mask)
    fp_list, tp_list, _ = metrics.roc_curve(gt_mask.flatten(), scores.flatten())
    patch_auc = metrics.auc(fp_list, tp_list)

    precision, recall, threshold = metrics.precision_recall_curve(gt_mask.flatten(), scores.flatten(), pos_label=1)
    numerator = 2 * precision * recall
    denominator = precision + recall

    numerator[np.where(denominator == 0)] = 0
    denominator[np.where(denominator == 0)] = 1

    # get optimal threshold
    f1_list = numerator / denominator
    best_ths = threshold[np.argmax(f1_list).astype(int)]

    print('[{}] image ROCAUC: {:.04f}\t pixel ROCAUC: {:.04f}'.format(category, img_roc_auc, patch_auc))

    if is_plot is True:
        save_dir = os.path.join(os.path.dirname(__file__), 'img_'+net_type,data,'img_'+category)
        print('save_dir:', save_dir)
        if os.path.isdir(save_dir) is False:
            os.makedirs(save_dir)
        plot_fig(test_imgs, scores, gt_mask, best_ths, save_dir, category)

    save_dir = os.path.join(os.path.dirname(__file__), 'img_'+net_type,data,data+'.csv')
    save_result(save_dir, category,net_type,batch_size,rd,img_roc_auc, patch_auc, f1, base_line, inference_time)

    return img_roc_auc, patch_auc