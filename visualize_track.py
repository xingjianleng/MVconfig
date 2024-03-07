import json
import os.path as osp

import cv2
import matplotlib.pyplot as plt
import numpy as np


def visualize_track(t, map_range):
    # The x,y ranges of the map, in meter
    x_min, x_max, y_min, y_max = map_range

    # We cut them into 2.5 cm grid
    out = np.zeros((int((x_max - x_min) * 40), int((y_max - y_min) * 40), 3), dtype=np.uint8)

    unique_ids = np.unique(t[:, 1])
    for unique_id in unique_ids:
        sequence = t[t[:, 1] == unique_id]
        trace = []
        for frame, _, x, y in sequence:
            print(x, y)
            # Find the mapped x and y coordinates
            x = int(y)
            y = int(x)

            # If the point is out of the map, we raise a warning and skip it
            if x < 0 or x >= out.shape[0] or y < 0 or y >= out.shape[1]:
                print(f'Warning: point ({x}, {y}) is out of the map')
                continue

            trace.append((x, y))
        
        # Draw the trace using OpenCV
        for i in range(1, len(trace)):
            cv2.line(out, trace[i-1], trace[i], (0, 255, 0),10)
        
    return out



if __name__ == '__main__':
    # NOTE: Remember to change these two attributes
    cfgdir = 'cfg/RL/'
    logdir = 'logs/carlax_new/town03cafe_1_TASK_reID_max_e10_2024-03-07_14-52-51'
    cfg_name = 'town03cafe_1'

    gt = np.loadtxt(osp.join(logdir, 'track_gt_1.txt'))
    t = np.loadtxt(osp.join(logdir, 'track_pred_1.txt'))

    with open(osp.join(cfgdir, f'{cfg_name}.cfg'), 'r') as f:
        cfg = json.load(f)
        map_range = cfg['spawn_area']
    
    gt_track = visualize_track(gt, map_range)
    plt.imsave('1.png', gt_track)
