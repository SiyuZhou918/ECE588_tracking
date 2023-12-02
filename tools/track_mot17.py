import json
import sys
import os
from glob import glob

import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import motmetrics as mm
import numpy as np
import scipy
import torch
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from yolox.data.data_augment import ValTransform
from yolox.exp import get_exp
from yolox.utils import postprocess


# ============= Detection Functions ==============
def init_detector(exp_file=None, exp_name=None, checkpoint_path=None, testing=False):
    # exp = get_exp('./exps/yolox_tiny.py', 'yolox-tiny')
    exp = get_exp(exp_file, exp_name)
    if testing:
        exp.test_conf = 0.1
    model = exp.get_model()
    model.eval()
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    return model, exp


def detect(model, exp, img_path):
    img = cv2.imread(img_path)
    ratio = min(exp.test_size[0] / img.shape[0], exp.test_size[1] / img.shape[1])
    pre_proc = ValTransform()
    img,_ = pre_proc(img, None, exp.test_size)
    img = torch.from_numpy(img).unsqueeze(0).to(device)
    img = img.float()
    with torch.no_grad():
        outputs = model(img)
    outputs = postprocess(outputs, exp.num_classes, exp.test_conf, exp.nmsthre)
    outputs = (outputs[0]).cpu()
    bboxes = outputs[:, 0:4]
    bboxes /= ratio # resize
    scores = outputs[:, 4] * outputs[:, 5]
    return bboxes, scores


def display(img_path, bboxes, scores=None):
    if scores != None:
        assert(len(bboxes) == len(scores))
    n = len(bboxes)
    colors = plt.get_cmap('tab20', n)
    img = plt.imread(img_path)
    for i in range(n):
        bb = bboxes[i]
        score = str(scores[i].item())[:4]
        color = colors(i)
        rect = plt.Rectangle((int(bb[0]),int(bb[1])),int(np.abs(bb[2]-bb[0])),int(np.abs(bb[3]-bb[1])), fill=False, edgecolor=color, linewidth=2)
        plt.gca().add_patch(rect)
        if scores != None:
            plt.text(int(bb[0])+50,int(bb[1])+50, score, color=color, fontsize=10)
    plt.imshow(img, origin='upper', interpolation='nearest')
    plt.axis('off')
    plt.show()
    plt.close()

# ============= Kalman Filter for ByteTrack ==============
"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""

chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(self):
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        #mean = np.dot(self._motion_mat, mean)
        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov


    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False, metric='maha'):
        """Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.
        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.
        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False,
                overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError('invalid distance metric')


# ============= Track and Kalman Filter Functions =============
# Track Class keeps track of an object over time.
# Includes lists for frames and bounding boxes and scores and a kalman filter
class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class Track:
    def __init__(self, frame, bbox, score, testing=False):
        try:
            assert bbox[0] < bbox[2]
            assert bbox[1] < bbox[3]
        except:
            print('Track Initiation Error:')
            print('\tframe number:',frame)
            print('\tbounding box:', bbox)
            print('\tscore:', score)
            sys.exit()
        # Get the center of bounding box
        cx = (bbox[0] + bbox[2])/2
        cy = (bbox[1] + bbox[3])/2
        # Get the width and height
        w = np.abs(bbox[0] - bbox[2])
        h = np.abs(bbox[1] - bbox[3])
        a = w/h # aspect ratio
        # Assign everything
        self.testing = testing
        self.frames = [frame]
        self.bboxes = [bbox]
        self.scores = [score]
        self.state = TrackState.Tracked
        self.kalman_filter = KalmanFilter() # Define the Kalman Filter for this track
        self.mean, self.covariance = self.kalman_filter.initiate([cx, cy, a, h]) # (cx, cy, a, h)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0 # x, y, a, h, vx, vy, va, vh
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    def update(self, frame, bbox, score):
        try:
            assert bbox[0] < bbox[2]
            assert bbox[1] < bbox[3]
        except:
            print('Track Update Error:')
            print('\tframe number:',frame)
            print('\tbounding box:', bbox)
            print('\tscore:', score)
            sys.exit()
        # Get the center of bounding box
        cx = (bbox[0] + bbox[2])/2
        cy = (bbox[1] + bbox[3])/2
        # Get the width and height
        w = np.abs(bbox[0] - bbox[2])
        h = np.abs(bbox[1] - bbox[3])
        a = w/h # aspect ratio
        self.frames.append(frame)
        self.bboxes.append(bbox)
        self.scores.append(score)
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, [cx, cy, a, h])

    def getTrack(self):
        return [self.frames.copy(), self.bboxes.copy(), self.scores.copy()]


# ============= Association Functions =============
def IOU(bbox1, bbox2): # bbox1: A list of four numbers [x1, y1, x2, y2] representing the bounding box
    try:
        assert bbox1[0] < bbox1[2]
        assert bbox1[1] < bbox1[3]
        assert bbox2[0] < bbox2[2]
        assert bbox2[1] < bbox2[3]
    except:
        print('track:', bbox1)
        print('detection:', bbox2)
        sys.exit()
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    if x1 > x2 or y1 > y2:
        return 0
    intersection_area = (x2 - x1) * (y2 - y1)
    union_area =  (bbox1[2] - bbox1[0])*(bbox1[3] - bbox1[1]) + (bbox2[2] - bbox2[0])*(bbox2[3] - bbox2[1]) - intersection_area
    return intersection_area / union_area


def gen_cost_matrix(tracks, detections):
    # generate cost matrix with tracks as rows and detections as columns
    num = max(len(tracks), len(detections))
    cost_matrix = np.zeros((num,num))
    for i in range(num):
        if i < len(tracks):
            t = tracks[i]
            for j in range(num):
                if j < len(detections):
                    det = detections[j]
                    cost_matrix[i][j] = IOU(t, det)
    return cost_matrix


def associate(tracks, detections, rejection_threshold):
    cost_matrix = gen_cost_matrix(tracks, detections)
    hungarian_associations = linear_sum_assignment(cost_matrix,maximize=True)
    track_accociations = []
    detection_associations = []
    for i in range(len(hungarian_associations[0])):
        track_ind = hungarian_associations[0][i]
        detection_ind = hungarian_associations[1][i]
        if cost_matrix[track_ind][detection_ind] > 0.2:
            if track_ind < len(tracks) and detection_ind < len(detections):
                track_accociations.append(track_ind)
                detection_associations.append(detection_ind)
    return [track_accociations, detection_associations]


def getRemainingIndices(input_list, associated_indices):
    indices = list(range(len(input_list)))
    for i in associated_indices:
        indices.remove(i)
    return indices


def getRemaining(input_list, remaining_indices):
    remaining = []
    for i in remaining_indices:
        remaining.append(input_list[i])
    return remaining


class Track_List:
    def __init__(self, testing=False):
        self.curr_ind = 0
        self.track_list = []
    def addTrack(self, frame, bbox, score):
        self.track_list.append(Track(frame, bbox, score))
        self.curr_ind += 1
    def predictAll(self):
        for track in self.track_list:
            track.predict()
        bboxes = []
        for track in self.track_list:
            mean_state = track.mean
            cx = mean_state[0]
            cy = mean_state[1]
            a = np.abs(mean_state[2])
            h = np.abs(mean_state[3])
            w = a * h # since a = w / h
            bboxes.append([cx-w/2, cy-h/2, cx+w/2, cy+h/2])
        return bboxes
    def updateTrack(self, track_ind, frame, bbox, score):
        self.track_list[track_ind].update(frame, bbox, score)
    def getTrack(self, track_ind):
        track = self.track_list[track_ind]
        return track.getTrack()
    def getAllTracks(self):
        tracks = {}
        for track_ind in range(len(self.track_list)):
            tracks[track_ind] = self.track_list[track_ind].getTrack()
        return tracks


# ============= Test Tracking Algorithm =============
def visualizeTrack(img_paths, frame_nums, bboxes, scores, out_path):
    frames = []
    fig, ax = plt.subplots()
    for i in tqdm(range(len(img_paths))):
        img = cv2.imread(img_paths[i])
        if i in frame_nums:
            ind = frame_nums.index(i)
            bbox = bboxes[ind]
            w = bbox[2]-bbox[0]
            h = bbox[3]-bbox[1]
            score = scores[ind]
            img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,0,255), 3)
            img = cv2.putText(img, str(round(score,2)), (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=2)
        img = ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), origin='upper', interpolation='nearest', animated=True)
        frames.append([img])
    plt.axis('off')
    print('Saving tracking video...', end="")
    ani = animation.ArtistAnimation(fig, frames, interval=33, blit=True, repeat_delay=1000)
    ani.save(out_path)
    print('Done!')
    html = ani.to_html5_video()
    return html


def visualizeAllTracks(img_paths, tracks_dict, out_path):
    num_tracks = len(tracks_dict.keys())
    track_id_list = sorted(tracks_dict.keys())
    cmap = plt.get_cmap('tab20', num_tracks)
    colors = {}
    for i in range(num_tracks):
        track_id = track_id_list[i]
        colors[track_id] = [int(x*256) for x in cmap(i)[:3]]
    frames = []
    fig, ax = plt.subplots()
    for frame_num in tqdm(range(len(img_paths))):
        img = cv2.imread(img_paths[frame_num])
        for track_id in track_id_list:
            color = colors[track_id]
            frame_nums = tracks_dict[track_id][0]
            bboxes = tracks_dict[track_id][1]
            scores = tracks_dict[track_id][2]
            if frame_num in frame_nums:
                ind = frame_nums.index(frame_num)
                bbox = bboxes[ind]
                w = bbox[2]-bbox[0]
                h = bbox[3]-bbox[1]
                score = scores[ind]
                img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 3)
                img = cv2.putText(img, str(round(score,2)), (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=color, thickness=2)
        img = ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), origin='upper', interpolation='nearest', animated=True)
        frames.append([img])
    plt.axis('off')
    print('Saving tracking video...', end="")
    ani = animation.ArtistAnimation(fig, frames, interval=33, blit=True, repeat_delay=1000)
    ani.save(out_path)
    print('Done!')
    return


# ============= Run Tracking Algorithm =============
def track_video(model, frames_dir_list, detection_threshold=0.5, rejection_threshold=0.2):
    tracks = Track_List()
    frame_num = 0
    for img_path in tqdm(sorted(frames_dir_list)):
        # predict detection boxes and scores
        bboxes, scores = detect(model, exp, img_path)
        det_high = []
        scores_high = []
        det_low = []
        scores_low = []
        for i in range(len(bboxes)):
            bbox = bboxes[i].tolist()
            score = scores[i].item()
            if score > detection_threshold:
                det_high.append(bbox)
                scores_high.append(score)
            else:
                det_low.append(bbox)
                scores_low.append(score)

        # predict new locations of tracks
        track_bboxes = tracks.predictAll()
        track_indices = list(range(len(track_bboxes)))

        # first association
        first_associations = associate(track_bboxes, det_high, rejection_threshold)
        for i in range(len(first_associations[0])):  # update kalman filter for associated track,detection pairs
            track_ind = first_associations[0][i]
            detection_ind = first_associations[1][i]
            bbox = det_high[detection_ind]
            score = scores_high[detection_ind]
            tracks.updateTrack(track_ind, frame_num, bbox, score)
        # get remaining tracks and detections
        track_ind_remaining = getRemainingIndices(track_bboxes, first_associations[0])
        tracks_remaining = getRemaining(track_bboxes, track_ind_remaining)
        detection_ind_remaining = getRemainingIndices(det_high, first_associations[1])
        # detections_remaining = getRemaining(det_high, detection_ind_remaining)

        # second association
        second_associations = associate(tracks_remaining, det_low, rejection_threshold)
        for i in range(len(second_associations[0])):  # update kalman filter for associated track,detection pairs
            track_ind = track_ind_remaining[second_associations[0][i]]
            detection_ind = second_associations[1][i]
            bbox = det_low[detection_ind]
            score = scores_low[detection_ind]
            tracks.updateTrack(track_ind, frame_num, bbox, score)

        # initialize new tracks
        for detection_ind in detection_ind_remaining:
            bbox = det_high[detection_ind]
            score = scores_high[detection_ind]
            tracks.addTrack(frame_num, bbox, score)
        frame_num += 1

    return tracks, frame_num


if __name__ == '__main__':
    # path and parameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_path = '../datasets/MOT17/annotations/val_half.json'
    with open(data_path, 'r') as f:
        annotations = json.load(f)
        f.close()

    detection_threshold = 0.5  # Detection score threshold for high vs low
    rejection_threshold = 0.2  # Reject associations if the IOU is less than this

    # load the detection model
    model, exp = init_detector(
        exp_file='../exps/mot/yolox_tiny_mot.py',
        checkpoint_path='../pretrained/yolox_tiny_best_ckpt.pth.tar',
    )

    for video_name in annotations['videos']:
        # run tracking in a video
        frames_dir_list = [x['file_name'] for x in annotations['images'] if x['file_name'].split('/')[0] == video_name['file_name']]
        frames_dir_list = [os.path.join('../datasets/MOT17/train/', x) for x in frames_dir_list]
        frames_id_list = sorted([x['id'] for x in annotations['images'] if x['file_name'].split('/')[0] == video_name['file_name']])

        tracks, num_frames = track_video(
            model=model,
            frames_dir_list=frames_dir_list,
            detection_threshold=detection_threshold,
            rejection_threshold=rejection_threshold,
        )

        # quantitative evaluation
        # adapted from https://github.com/cheind/py-motmetrics
        tracks_dict = tracks.getAllTracks()
        acc = mm.MOTAccumulator(auto_id=True)
        for idx_frame in range(num_frames):
            track_ids = [[x['track_id']] for x in annotations['annotations'] if x['image_id'] == frames_id_list[idx_frame]]
            bboxes = [x['bbox'] for x in annotations['annotations'] if x['image_id'] == frames_id_list[idx_frame]]
            track_ids, bboxes = np.array(track_ids), np.array(bboxes)
            gt_dets = np.concatenate([track_ids, bboxes], axis=1)
            # print('gt')
            # print(gt_dets[:5])

            t_dets = []
            for track_id in sorted(tracks_dict.keys()):
                frames_list = tracks_dict[track_id][0]
                if idx_frame in frames_list:
                    bboxes = tracks_dict[track_id][1]
                    ind = frames_list.index(idx_frame)
                    bbox = bboxes[ind]
                    t_dets.append(np.array([track_id+1, bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]))
            t_dets = np.stack(t_dets, axis=0)

            C = mm.distances.iou_matrix(gt_dets[:, 1:], t_dets[:, 1:], max_iou=0.5)  # format: gt, t
            acc.update(
                gt_dets[:, 0].astype('int').tolist(),
                t_dets[:, 0].astype('int').tolist(),
                C
            )

        mh = mm.metrics.create()
        summary = mh.compute(
            acc,
            metrics=[
                'num_frames', 'idf1', 'idp', 'idr', 'recall', 'precision', 'num_objects',
                'mostly_tracked', 'partially_tracked', 'mostly_lost', 'num_false_positives',
                'num_misses', 'num_switches', 'num_fragmentations', 'mota', 'motp'],
            name='acc'
        )
        strsummary = mm.io.render_summary(
            summary,
            # formatters={'mota' : '{:.2%}'.format},
            namemap={
                'idf1': 'IDF1', 'idp': 'IDP', 'idr': 'IDR', 'recall': 'Rcll', 'precision': 'Prcn',
                'num_objects': 'GT', 'mostly_tracked': 'MT', 'partially_tracked': 'PT',
                'mostly_lost': 'ML', 'num_false_positives': 'FP', 'num_misses': 'FN',
                'num_switches': 'IDsw', 'num_fragmentations': 'FM', 'mota': 'MOTA', 'motp': 'MOTP'
            }
        )
        print('')
        print(strsummary)








