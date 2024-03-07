import json
import os.path as osp

import cv2
import matplotlib.pyplot as plt
import numpy as np


import numpy as np
import cv2

import numpy as np
import cv2

def visualize_track(t, map_range, scale_multiplier=10):
    """
    Visualize tracks with high precision and unique colors for each trace.
    
    Args:
        t (numpy.ndarray): Array containing tracking data.
        map_range (tuple): The x_min, x_max, y_min, y_max ranges of the map, in meters.
        scale_multiplier (int, optional): Multiplier to scale float coordinates for precision. Defaults to 100.
    """
    # The x, y ranges of the map, in meter
    x_min, x_max, y_min, y_max = map_range

    # Calculate output image size with scaling for precision
    out = np.zeros((int((x_max - x_min) * 40 * scale_multiplier),
                    int((y_max - y_min) * 40 * scale_multiplier), 3), dtype=np.uint8)

    unique_ids = np.unique(t[:, 1])
    
    # Define a list or a colormap for different traces
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    color_index = 0

    for unique_id in unique_ids:
        sequence = t[t[:, 1] == unique_id]
        trace = []
        for _, _, x, y in sequence:
            # Apply scaling to convert float coordinates to integer with high precision
            x_mapped = int(y * scale_multiplier)
            y_mapped = int(x * scale_multiplier)

            if y_mapped < 0 or y_mapped >= out.shape[0] or x_mapped < 0 or x_mapped >= out.shape[1]:
                print(f'Warning: point ({x_mapped}, {y_mapped}) is out of the map')
                continue

            trace.append((x_mapped, y_mapped))
        
        # Draw the trace with a unique color and adjusted thickness for better visibility
        for i in range(1, len(trace)):
            cv2.line(out, trace[i-1], trace[i], colors[color_index % len(colors)], thickness=20)
        
        color_index += 1  # Move to the next color for the next trace
        
    return out


if __name__ == '__main__':
    # NOTE: Remember to change these two attributes
    cfgdir = 'cfg/RL/'
    logdir = 'logs/carlax/town04building_1_TASK_reID_max_e10_2024-03-07_16-47-57'
    cfg_name = 'town04building_1'

    gt = np.loadtxt(osp.join(logdir, 'track_gt_0.txt'))
    t = np.loadtxt(osp.join(logdir, 'track_pred_0.txt'))

    with open(osp.join(cfgdir, f'{cfg_name}.cfg'), 'r') as f:
        cfg = json.load(f)
        map_range = cfg['spawn_area']
    
    t_track = visualize_track(t, map_range)
    gt_track = visualize_track(gt, map_range)

    plt.imsave('t.png', t_track)
    plt.imsave('gt.png', gt_track)
