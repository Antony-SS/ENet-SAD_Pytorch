import argparse
import json
import os.path as osp
import os
import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torchvision
from torch.utils.data import Dataset
from tqdm import tqdm

import utils.transforms as tf
from fuzzy_algocompare import *
from runner.evaluator.tusimple.lane import LaneEval
from runner.logger import get_logger
from runner.runner import Runner
from utils.config import Config

cannythreshold = 1
cannycontrol = 0


class ExampleData:
    def __init__(self, img_path, list_path='list', cfg=None):

        self.cfg = cfg
        self.img_path = img_path
        self.list_path = osp.join(img_path, list_path)
        # self.is_training = ('train' in data_list)
        self.is_training = False

        self.img_name_list = []
        self.full_img_path_list = []
        self.label_list = []
        self.exist_list = []
        self.load_annotations()

        self.transform = self.transform_val()

        # self.init()

    def init(self):
        # raise NotImplementedError()
        pass

    def load_annotations(self):
        print('Loading TuSimple annotations...')
        #self.data_infos = []

        files = os.listdir(self.img_path)
        files = sorted(files, key=lambda x: int(x.split('-')[1].split('.')[0]))

        for file in files:
            self.full_img_path_list.append(osp.join(self.img_path, file))
            self.img_name_list.append(file)

    def view(self, img, coords, file_path=None):
        for coord in coords:
            for x, y in coord:
                if x <= 0 or y <= 0:
                    continue
                x, y = int(x), int(y)
                cv2.circle(img, (x, y), 4, (255, 0, 0), 2)

        if file_path is not None:
            if not os.path.exists(osp.dirname(file_path)):
                os.makedirs(osp.dirname(file_path))
            cv2.imwrite(file_path, img)

    def transform_val(self):
        val_transform = torchvision.transforms.Compose([
            tf.SampleResize((self.cfg.img_width, self.cfg.img_height)),
            tf.GroupNormalize(mean=(self.cfg.img_norm['mean'], (0,)), std=(
                self.cfg.img_norm['std'], (1,))),
        ])
        return val_transform

    def __len__(self):
        return len(self.full_img_path_list)

    def preprocess(self, inputimg):
        global cannycontrol
        global cannythreshold
        readimg = cv2.imread(inputimg)
        out = cv2.cvtColor(readimg, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(out, 7, 25, 50)
        height, width = out.shape
        y_intercept = int(1 / 4 * height)
        x_intercept = int(width / 2)
        cannythreshold = cannythreshold + cannycontrol
        # print('cannythreshold = ', cannythreshold)
        if cannythreshold < 0:
            threshold = 1
        if cannythreshold > 500:
            cannythreshold = 500
        high = cannythreshold
        low = high / 3
        edge = cv2.Canny(filtered, low, high, None, 3)
        edge = np.uint8(edge)
        myROI = np.array([[(x_intercept, y_intercept), (0, height - 1), (width - 1, height - 1)]],
                         dtype=np.int32)  # 30->10
        mask = np.zeros_like(edge)
        region = cv2.fillPoly(mask, myROI, 255)
        roi = cv2.bitwise_and(edge, region)

        lines = cv2.HoughLines(roi, 1, np.pi / 180, 3, None, 0, 0)
        if lines is not None:
            rhoall = lines[:, :, 0]
            thetaall = lines[:, :, 1]
            totalinesall = len(rhoall)
        else:
            totalinesall = 0
        if totalinesall > 58000:
            totalinesall = 58000
        cannycontrol = fuzzy_canny(totalinesall)
        #readimg[:, :, 0] = edge
        #readimg[:, :, 1] = edge
        #readimg[:, :, 2] = edge

        min_val = np.min(edge)
        max_val = np.max(edge)
        normalized_edge = (edge - min_val) / (max_val - min_val)
        scaled_edge = (normalized_edge * 255).astype(np.uint8)
        # expand_edge = np.expand_dims(scaled_edge, axis=2)
        # output = np.concatenate((img, expand_edge), axis=2)
        readimg[:, :, 0] = scaled_edge
        readimg[:, :, 2] = scaled_edge
        output = readimg

        return output

    def __getitem__(self, idx):
        #readimg = cv2.imread(self.full_img_path_list[idx]).astype(np.float32)
        #print(self.full_img_path_list[idx])
        img = self.preprocess(self.full_img_path_list[idx])
        #img = readimg
        img = img[self.cfg.cut_height:, :, :]

        if self.is_training:
            label = cv2.imread(self.label_list[idx], cv2.IMREAD_UNCHANGED)
            if len(label.shape) > 2:
                label = label[:, :, 0]
            label = label.squeeze()
            label = label[self.cfg.cut_height:, :]
            exist = self.exist_list[idx]
            if self.transform:
                img, label = self.transform((img, label))
            label = torch.from_numpy(label).contiguous().long()
        else:
            img, = self.transform((img,))

        img = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()
        meta = {'full_img_path': self.full_img_path_list[idx],
                'img_name': self.img_name_list[idx]}

        data = {'img': img, 'meta': meta}
        if self.is_training:
            data.update({'label': label, 'exist': exist})
        return data




class EvaluationFns(nn.Module):
    def __init__(self, cfg):
        super(EvaluationFns, self).__init__()
        self.cfg = cfg
        # exp_dir = os.path.join(self.cfg.work_dir, "output")
        # if not os.path.exists(exp_dir):
        #    os.mkdir(exp_dir)
        # self.out_path = os.path.join(exp_dir, "coord_output")
        # if not os.path.exists(self.out_path):
        #    os.mkdir(self.out_path)
        self.dump_to_json = []
        self.thresh = 0.60
        self.logger = get_logger('resa')
        if cfg.view:
            self.view_dir = os.path.join(self.cfg.work_dir, 'vis')

    def fix_gap(self, coordinate):
        if any(x > 0 for x in coordinate):
            start = [i for i, x in enumerate(coordinate) if x > 0][0]
            end = [i for i, x in reversed(list(enumerate(coordinate))) if x > 0][0]
            lane = coordinate[start:end + 1]
            if any(x < 0 for x in lane):
                gap_start = [i for i, x in enumerate(
                    lane[:-1]) if x > 0 and lane[i + 1] < 0]
                gap_end = [i + 1 for i,
                x in enumerate(lane[:-1]) if x < 0 and lane[i + 1] > 0]
                gap_id = [i for i, x in enumerate(lane) if x < 0]
                if len(gap_start) == 0 or len(gap_end) == 0:
                    return coordinate
                for id in gap_id:
                    for i in range(len(gap_start)):
                        if i >= len(gap_end):
                            return coordinate
                        if id > gap_start[i] and id < gap_end[i]:
                            gap_width = float(gap_end[i] - gap_start[i])
                            lane[id] = int((id - gap_start[i]) / gap_width * lane[gap_end[i]] + (
                                    gap_end[i] - id) / gap_width * lane[gap_start[i]])
                if not all(x > 0 for x in lane):
                    print("Gaps still exist!")
                coordinate[start:end + 1] = lane
        return coordinate

    def is_short(self, lane):
        start = [i for i, x in enumerate(lane) if x > 0]
        if not start:
            return 1
        else:
            return 0

    def get_lane(self, prob_map, y_px_gap, pts, thresh, resize_shape=None):
        """
        Arguments:
        ----------
        prob_map: prob map for single lane, np array size (h, w)
        resize_shape:  reshape size target, (H, W)

        Return:
        ----------
        coords: x coords bottom up every y_px_gap px, 0 for non-exist, in resized shape
        """
        if resize_shape is None:
            resize_shape = prob_map.shape
        h, w = prob_map.shape
        H, W = resize_shape
        H -= self.cfg.cut_height

        coords = np.zeros(pts)
        coords[:] = -1.0
        for i in range(pts):
            y = int((H - 10 - i * y_px_gap) * h / H)
            if y < 0:
                break
            line = prob_map[y, :]
            id = np.argmax(line)
            if line[id] > thresh:
                coords[i] = int(id / w * W)
        if (coords > 0).sum() < 2:
            coords = np.zeros(pts)
        self.fix_gap(coords)
        # print(coords.shape)

        return coords

    def probmap2lane(self, seg_pred, exist, resize_shape=(720, 1280), smooth=True, y_px_gap=10, pts=56, thresh=0.6):
        """
        Arguments:
        ----------
        seg_pred:      np.array size (5, h, w)
        resize_shape:  reshape size target, (H, W)
        exist:       list of existence, e.g. [0, 1, 1, 0]
        smooth:      whether to smooth the probability or not
        y_px_gap:    y pixel gap for sampling
        pts:     how many points for one lane
        thresh:  probability threshold

        Return:
        ----------
        coordinates: [x, y] list of lanes, e.g.: [ [[9, 569], [50, 549]] ,[[630, 569], [647, 549]] ]
        """
        if resize_shape is None:
            resize_shape = seg_pred.shape[1:]  # seg_pred (5, h, w)
        _, h, w = seg_pred.shape
        H, W = resize_shape
        coordinates = []

        for i in range(self.cfg.num_classes - 1):
            prob_map = seg_pred[i + 1]
            if smooth:
                prob_map = cv2.blur(prob_map, (9, 9), borderType=cv2.BORDER_REPLICATE)
            coords = self.get_lane(prob_map, y_px_gap, pts, thresh, resize_shape)
            if self.is_short(coords):
                continue
            coordinates.append(
                [[coords[j], H - 10 - j * y_px_gap] if coords[j] > 0 else [-1, H - 10 - j * y_px_gap] for j in
                 range(pts)])

        if len(coordinates) == 0:
            coords = np.zeros(pts)
            coordinates.append(
                [[coords[j], H - 10 - j * y_px_gap] if coords[j] > 0 else [-1, H - 10 - j * y_px_gap] for j in
                 range(pts)])
        # print(coordinates)

        return coordinates

    def evaluate_pred(self, dataset, seg_pred, exist_pred, batch):
        #print("call evaluate_pred")
        img_name = batch['meta']['img_name']
        img_path = batch['meta']['full_img_path']
        for b in range(len(seg_pred)):
            seg = seg_pred[b]
            exist = [1 if exist_pred[b, i] >
                          0.5 else 0 for i in range(6)]
            lane_coords = self.probmap2lane(seg, exist, thresh=0.6)
            for i in range(len(lane_coords)):
                lane_coords[i] = sorted(
                    lane_coords[i], key=lambda pair: pair[1])

            # path_tree = split_path(img_name[b])
            path_tree = img_name[b]
            # save_dir, save_name = path_tree[-3:-1], path_tree[-1]
            # save_dir = os.path.join(self.out_path, *save_dir)
            # save_name = save_name[:-3] + "lines.txt"
            # save_name = os.path.join(save_dir, save_name)
            # if not os.path.exists(save_dir):
            #    os.makedirs(save_dir, exist_ok=True)

            # with open(save_name, "w") as f:
            #    for l in lane_coords:
            #        for (x, y) in l:
            #            print("{} {}".format(x, y), end=" ", file=f)
            #        print(file=f)

            json_dict = {}
            json_dict['lanes'] = []
            json_dict['h_sample'] = []
            # json_dict['raw_file'] = os.path.join(*path_tree[-4:])
            json_dict['raw_file'] = path_tree
            json_dict['run_time'] = 0
            for l in lane_coords:
                if len(l) == 0:
                    continue
                json_dict['lanes'].append([])
                for (x, y) in l:
                    json_dict['lanes'][-1].append(int(x))
            for (x, y) in lane_coords[0]:
                json_dict['h_sample'].append(y)
            self.dump_to_json.append(json.dumps(json_dict))
            #print("Dump to Json")
            if self.cfg.view:
                img = cv2.imread(img_path[b])
                new_img_name = img_name[b].replace('/', '_')
                save_dir = os.path.join(self.view_dir, new_img_name)
                dataset.view(img, lane_coords, save_dir)

    def evaluate(self, dataset, output, batch):
        seg_pred, exist_pred = output['seg'], output['exist']
        seg_pred = F.softmax(seg_pred, dim=1)
        seg_pred = seg_pred.detach().cpu().numpy()
        exist_pred = exist_pred.detach().cpu().numpy()
        self.evaluate_pred(dataset, seg_pred, exist_pred, batch)

    def summarize(self):
        best_acc = 0
        output_file = os.path.join(self.cfg.work_dir, 'predict_test.json')

        self.logger.info("SAVING TO DIRECTORY: " + str(output_file))

        with open(output_file, "w+") as f:
            for line in self.dump_to_json:
                print(line, end="\n", file=f)

        #eval_result, acc = LaneEval.bench_one_submit(output_file,
        #                                             self.cfg.test_json_file)

        #self.logger.info(eval_result)
        self.dump_to_json = []
        #best_acc = max(acc, best_acc)
        return 0


def validate(self, val_loader, cfg):
    self.net.eval()
    evaluator = EvaluationFns(cfg)

    for i, data in enumerate(tqdm(val_loader, desc=f'Validate')):
        data = self.to_cuda(data)
        with torch.no_grad():
            output = self.net(data['img'])
            evaluator.evaluate(val_loader.dataset, output, data)
    metric = evaluator.summarize()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(gpu) for gpu in args.gpus)

    cfg = Config.fromfile(args.config)
    cfg.gpus = len(args.gpus)

    cfg.load_from = args.load_from
    cfg.finetune_from = args.finetune_from
    cfg.view = args.view

    cfg.work_dirs = args.work_dirs + '/' + cfg.dataset.train.type

    cudnn.benchmark = True
    cudnn.fastest = True

    runner = Runner(cfg)

    if args.validate:
        # val_loader = build_dataloader(cfg.dataset.val, cfg, is_train=False)
        dataset = ExampleData('./data/weather/new_D_R_9_figure/', 'list', cfg=cfg)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.workers, pin_memory=False, drop_last=False)
        validate(runner, data_loader, cfg)
    else:
        runner.train()


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--work_dirs', type=str, default='work_dirs',
        help='work dirs')
    parser.add_argument(
        '--load_from', default=None,
        help='the checkpoint file to resume from')
    parser.add_argument(
        '--finetune_from', default=None,
        help='whether to finetune from the checkpoint')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--view',
        action='store_true',
        help='whether to show visualization result')
    parser.add_argument('--gpus', nargs='+', type=int, default='0')
    parser.add_argument('--seed', type=int,
                        default=None, help='random seed')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
