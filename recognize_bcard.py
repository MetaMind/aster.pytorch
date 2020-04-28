from __future__ import absolute_import
import sys

sys.path.append('./')

import argparse
import os
import os.path as osp
import numpy as np
import math
import time
from PIL import Image, ImageFile
import pickle
import cv2

import torch
from torch import nn, optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

from config import get_args
from lib import datasets, evaluation_metrics, models
from lib.models.model_builder import ModelBuilder
from lib.datasets.dataset import LmdbDataset, AlignCollate
from lib.loss import SequenceCrossEntropyLoss
from lib.trainers import Trainer
from lib.evaluators import Evaluator
from lib.utils.logging import Logger, TFLogger
from lib.utils.serialization import load_checkpoint, save_checkpoint
from lib.utils.osutils import make_symlink_if_not_exists
from lib.evaluation_metrics.metrics import get_str_list
from lib.utils.labelmaps import get_vocabulary, labels2strs

global_args = get_args(sys.argv[1:])


def _crop_rotated_rectangle(image, polygon):
    polygon = np.array(polygon, dtype=np.float32)
    rect = cv2.minAreaRect(polygon)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = box.astype("float32")

    dst_pts = np.array([[0, height - 1],
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1]], dtype="float32")

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    warped_image = cv2.warpPerspective(image, M, (width, height))

    return warped_image, box


def image_process(img, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
    if keep_ratio:
        w, h = img.size
        ratio = w / float(h)
        imgW = int(np.floor(ratio * imgH))
        imgW = max(imgH * min_ratio, imgW)

    img = img.resize((imgW, imgH), Image.BILINEAR)
    img = transforms.ToTensor()(img)
    img.sub_(0.5).div_(0.5)

    return img


class DataInfo(object):
    """
    Save the info about the dataset.
    This a code snippet from dataset.py
    """

    def __init__(self, voc_type):
        super(DataInfo, self).__init__()
        self.voc_type = voc_type

        assert voc_type in ['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS']
        self.EOS = 'EOS'
        self.PADDING = 'PADDING'
        self.UNKNOWN = 'UNKNOWN'
        self.voc = get_vocabulary(voc_type, EOS=self.EOS, PADDING=self.PADDING, UNKNOWN=self.UNKNOWN)
        self.char2id = dict(zip(self.voc, range(len(self.voc))))
        self.id2char = dict(zip(range(len(self.voc)), self.voc))

        self.rec_num_classes = len(self.voc)


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    args.cuda = args.cuda and torch.cuda.is_available()
    if args.cuda:
        print('using cuda.')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    # Create data loaders
    if args.height is None or args.width is None:
        args.height, args.width = (32, 100)

    dataset_info = DataInfo(args.voc_type)

    # Create model
    model = ModelBuilder(arch=args.arch, rec_num_classes=dataset_info.rec_num_classes,
                         sDim=args.decoder_sdim, attDim=args.attDim, max_len_labels=args.max_len,
                         eos=dataset_info.char2id[dataset_info.EOS], STN_ON=args.STN_ON)

    # Load from checkpoint
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])

    if args.cuda:
        device = torch.device("cuda")
        model = model.to(device)
        model = nn.DataParallel(model)

    # Evaluation
    model.eval()

    # loop through business cards
    pkl_file = open(args.detection_path, 'rb')
    cards = pickle.load(pkl_file)

    for img_name in cards.keys():
        img_path = os.path.join(args.image_path, img_name)
        img_card = cv2.imread(img_path)
        img_card = cv2.cvtColor(img_card, cv2.COLOR_BGR2RGB)
        words = cards[img_name]
        recognized = []
        words = words + (recognized,)
        for bboxes in words[0]:
            for idx, bbox in enumerate(bboxes):
                recognized.append('')
                if words[1][0][idx] >= args.detection_threshold:
                    img, box_contour = _crop_rotated_rectangle(img_card, bbox)
                    img = Image.fromarray(img)
                    img = image_process(img)
                    
                    with torch.no_grad():
                        img = img.to(device)
                    input_dict = {}
                    input_dict['images'] = img.unsqueeze(0)
                    # TODO: testing should be more clean.
                    # to be compatible with the lmdb-based testing, need to construct some meaningless variables.
                    rec_targets = torch.IntTensor(1, args.max_len).fill_(1)
                    rec_targets[:, args.max_len - 1] = dataset_info.char2id[dataset_info.EOS]
                    input_dict['rec_targets'] = rec_targets
                    input_dict['rec_lengths'] = [args.max_len]
                    output_dict = model(input_dict)
                    pred_rec = output_dict['output']['pred_rec']
                    pred_str, _ = get_str_list(pred_rec, input_dict['rec_targets'], dataset=dataset_info)
                    print('Recognition result: {0}'.format(pred_str[0]))
                    cv2.drawContours(img_card, [box_contour], 0, (0,255,0),2)                    
                    img_card = cv2.putText(img_card, pred_str[0], (int(bbox[0][0]), int(bbox[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 2, [255, 0, 0], 3)
                    recognized[-1] = pred_str[0]
        img_card = cv2.cvtColor(img_card, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(args.image_out_path, img_name), img_card)

    pickle.dump(cards, open(args.detection_out_path, "wb"))


if __name__ == '__main__':
    # parse the config
    args = get_args(sys.argv[1:])
    main(args)
