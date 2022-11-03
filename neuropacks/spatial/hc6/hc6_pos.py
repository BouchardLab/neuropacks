import os
import numpy as np
import pandas as pd

from scipy.io import loadmat

from .hc6 import HC6
from .hc6_geom import load_annotated_well_xys


# === position linearization ===

def distance_to_line(xys, ref1, ref2):
    '''
    xys: (n_points, 2) array
    ref1: (2, ) or (1, 2) array - xy coordinates
    ref2: (2, ) or (1, 2) array - xy coordinates
    '''
    x1, y1 = np.array(ref1).ravel()
    x2, y2 = np.array(ref2).ravel()
    # slope = (y2 - y1) / (x2 - x1)
    perp = ((y2 - y1), -(x2 - x1))
    perp = perp / np.linalg.norm(perp)
    return np.abs((xys - np.array(ref1).reshape(1, -1)) @ perp.reshape(-1, 1))


def pos_along_seg(xys, ref1, ref2):
    '''
    xys: (n_points, 2) array
    ref1: (2, ) or (1, 2) array - xy coordinates
    ref2: (2, ) or (1, 2) array - xy coordinates
    '''
    x1, y1 = np.array(ref1).ravel()
    x2, y2 = np.array(ref2).ravel()
    para = ((x2 - x1), (y2 - y1))
    para = para / np.linalg.norm(para)
    # l0 = np.array(ref1).reshape(1, -1) @ para.reshape(-1, 1)
    return (xys - np.array(ref1).reshape(1, -1)) @ para.reshape(-1, 1)


def linearize_position_single_traj(x_tr, y_tr,
                                   # x_start, x_stop, y0, y1,
                                   xy_nodes):
    # d_seg0 = np.abs(x_tr - x_start)  # down
    # d_seg1 = np.abs(y_tr - y0)  # horizontal
    # d_seg2 = np.abs(x_tr - x_stop)
    xy_tr = np.stack((x_tr, y_tr), axis=1)
    # d_seg0 = distance_to_line(xy_tr, (x_start, y1), (x_start, y0))
    # d_seg1 = distance_to_line(xy_tr, (x_start, y0), (x_stop, y0))
    # d_seg2 = distance_to_line(xy_tr, (x_stop, y0), (x_stop, y1))
    d_seg0 = distance_to_line(xy_tr, xy_nodes[0], xy_nodes[1])
    d_seg1 = distance_to_line(xy_tr, xy_nodes[1], xy_nodes[2])
    d_seg2 = distance_to_line(xy_tr, xy_nodes[2], xy_nodes[3])

    d_allsegs = np.stack((d_seg0, d_seg1, d_seg2), axis=1).squeeze()
    segments = np.argmin(d_allsegs, axis=-1)

    # l_seg0 = y1 - y0
    # l_seg1 = np.abs(x_stop - x_start)
    # l_seg2 = y1 - y0
    l_seg0 = np.linalg.norm(np.array(xy_nodes[0]) - np.array(xy_nodes[1]))
    l_seg1 = np.linalg.norm(np.array(xy_nodes[1]) - np.array(xy_nodes[2]))
    l_seg2 = np.linalg.norm(np.array(xy_nodes[2]) - np.array(xy_nodes[3]))

    # linpos_seg0 = y1 - y_tr
    # linpos_seg1 = np.abs(x_tr - x_start) + l_seg0
    # linpos_seg2 = (y_tr - y0) + l_seg0 + l_seg1
    # linpos_seg0 = pos_along_seg(xy_tr, (x_start, y1), (x_start, y0))
    # linpos_seg1 = pos_along_seg(xy_tr, (x_start, y0), (x_stop, y0)) + l_seg0
    # linpos_seg2 = pos_along_seg(xy_tr, (x_stop, y0), (x_stop, y1)) + l_seg0 + l_seg1
    linpos_seg0 = pos_along_seg(xy_tr, xy_nodes[0], xy_nodes[1])
    linpos_seg1 = pos_along_seg(xy_tr, xy_nodes[1], xy_nodes[2]) + l_seg0
    linpos_seg2 = pos_along_seg(xy_tr, xy_nodes[2], xy_nodes[3]) + l_seg0 + l_seg1
    linpos_allsegs = np.stack((linpos_seg0, linpos_seg1, linpos_seg2), axis=1).squeeze()

    linpos = np.empty(segments.shape) * np.nan
    for i, seg in enumerate(segments):
        linpos[i] = linpos_allsegs[i, seg]

    l_nodes = (0, l_seg0, l_seg0 + l_seg1, l_seg0 + l_seg1 + l_seg2)
    return linpos, segments, d_allsegs, l_nodes


def get_linpos_nodes(well_xys_dict, traj_type):
    # traj_type = f'{well_start}{well_stop}'
    well_start = traj_type[0]
    well_stop = traj_type[1]
    node_names = (well_start, f'T{well_start}', f'T{well_stop}', well_stop)
    x_nodes = [well_xys_dict[well][0] for well in node_names]
    y_nodes = [well_xys_dict[well][1] for well in node_names]
    xy_nodes = [(x, y) for x, y in zip(x_nodes, y_nodes)]

    l_seg0 = np.linalg.norm(np.array(xy_nodes[0]) - np.array(xy_nodes[1]))
    l_seg1 = np.linalg.norm(np.array(xy_nodes[1]) - np.array(xy_nodes[2]))
    l_seg2 = np.linalg.norm(np.array(xy_nodes[2]) - np.array(xy_nodes[3]))
    l_nodes = (0, l_seg0, l_seg0 + l_seg1, l_seg0 + l_seg1 + l_seg2)

    if 'TC' in node_names:
        idx_choice_point = np.where(np.array(node_names) == 'TC')[0].item()
    else:
        idx_choice_point = None

    return l_nodes, idx_choice_point


def get_pos_corr_nodes(epoch_kwargs, traj_type):
    well_xys_dict = load_annotated_well_xys(
        animal=epoch_kwargs['animal'], day=epoch_kwargs['day'],
        epoch=epoch_kwargs['epoch'])
    return get_linpos_nodes(well_xys_dict, traj_type)


# === trialization ===

def in_bb(xs, ys, xmin, xmax, ymin, ymax):
    ''' determine if the (x, y) points are in the bounding box.
    bounding box is axis-aligned.
    xs, ys: (n_points,) arrays
    '''
    hit_x = np.logical_and(xs >= xmin, xs < xmax)
    hit_y = np.logical_and(ys >= ymin, ys < ymax)
    return np.logical_and(hit_x, hit_y)


def in_any_bb(xs, ys, bbs):
    ''' check if points are in any of the bounding boxes '''
    n_points = len(xs)
    check_each_bb = np.zeros((n_points, len(bbs)))
    for i, bb in enumerate(bbs):
        check_each_bb[:, i] = in_bb(xs, ys, **bb)
    return np.any(check_each_bb, axis=1)


def _well_bounding_boxes(well_xys_dict, end_well_names=('L', 'C', 'R'),
                         w=15, h0=5, h1=20):
    # normally center arm is vertical (normal orientation, W-shaped);
    # if horizontal (x span > y span), rotated
    xy_well_C = np.array(well_xys_dict['C'])
    xy_well_TC = np.array(well_xys_dict['TC'])
    c_arm = xy_well_C - xy_well_TC
    rotated_w_maze = (np.abs(c_arm[0]) > np.abs(c_arm[1]))

    # bounding boxes
    bbs = []
    for well in end_well_names:
        x, y = well_xys_dict[well]
        if rotated_w_maze:
            if xy_well_C[0] < xy_well_TC[0]:
                # end wells on the left
                bbs.append({'xmin': x - h1, 'xmax': x + h0,
                            'ymin': y - w, 'ymax': y + w})
            else:
                # end wells on the right
                bbs.append({'xmin': x - h0, 'xmax': x + h1,
                            'ymin': y - w, 'ymax': y + w})
        else:
            bbs.append({'xmin': x - w, 'xmax': x + w,
                        'ymin': y - h0, 'ymax': y + h1})
    return bbs, rotated_w_maze


def trialize_epoch_pos(pos, well_xys_dict, end_well_names=('L', 'C', 'R'),
                       skip_shorter_than=1., **bounding_box_kwargs):
    '''
    skip_shorter_than: (float) skip trials shorter than this cutoff, in seconds.
        To filter out some cases where rat position tracking fluctuate
        and simply touch the next bounding box, which is not a real trajectory.
    '''
    # get the bounding boxes for the three end wells
    bbs, rotated_w_maze = _well_bounding_boxes(well_xys_dict,
                                               end_well_names=end_well_names,
                                               **bounding_box_kwargs)

    # detect when the animal goes into / out of the bounding boxes
    check = in_any_bb(pos['x'], pos['y'], bbs)
    c_prev = check[:-1]
    c_next = check[1:]
    i_well_out = np.where(np.logical_and(c_prev > 0, c_next == 0))[0]
    i_well_in = np.where(np.logical_and(c_prev == 0, c_next > 0))[0]
    if len(i_well_out) == 0 or len(i_well_in) == 0:
        trials = pd.DataFrame([])
        return trials
    i_well_in = i_well_in[i_well_in >= i_well_out[0]]

    well_xs = np.array([well_xys_dict[well][0] for well in end_well_names])
    well_ys = np.array([well_xys_dict[well][1] for well in end_well_names])

    # parse into trials and collect trajectory information
    trials_list = []
    for i_out, i_in in zip(i_well_out, i_well_in):
        if rotated_w_maze:
            traj_end_ys = np.array([pos['y'][i_out], pos['y'][i_in]])
            distances_to_wells = np.abs(
                well_ys.reshape(1, -1) - traj_end_ys.reshape(-1, 1))
        else:
            traj_end_xs = np.array([pos['x'][i_out], pos['x'][i_in]])
            distances_to_wells = np.abs(
                well_xs.reshape(1, -1) - traj_end_xs.reshape(-1, 1))
        well_inds = np.argmin(distances_to_wells, axis=-1)
        well_start = end_well_names[well_inds[0]]
        well_stop = end_well_names[well_inds[1]]
        if well_start == well_stop:
            # skip
            continue
        t_start = pos['time'][i_out]
        t_stop = pos['time'][i_in]
        if t_stop - t_start < skip_shorter_than:
            # false detection between neighboring end wells; skip
            continue
        trial = {'t_start': t_start, 't_stop': t_stop,
                 'well_start': well_start, 'well_stop': well_stop,
                 }
        trials_list.append(trial)
    trials = pd.DataFrame(trials_list)
    return trials
