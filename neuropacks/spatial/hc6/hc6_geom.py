import numpy as np


def load_annotated_well_xys(animal, day, epoch):
    ''' manually annotated (by eyes) '''
    animal = animal.lower()
    if animal == 'bon':
        pts, rotated_w_maze = _bon_maze_geometry(day, epoch)
    elif animal == 'con':
        pts, rotated_w_maze = _con_maze_geometry(day, epoch)
    elif animal == 'cor':
        pts, rotated_w_maze = _cor_maze_geometry(day, epoch)
    else:
        raise ValueError(f'animal {animal}: not yet annotated')

    if pts is None:
        raise ValueError(f'{animal} day{day} epoch{epoch}: not yet annotated')

    dyC = pts['dy']
    dyR = pts['dy'] * (pts['xR'] - pts['xL']) / (pts['xC'] - pts['xL'])
    well_xys_dict = {
        'L': (pts['xL'], pts['y1']),
        'C': (pts['xC'], pts['y1'] + dyC),
        'R': (pts['xR'], pts['y1'] + dyR),
        'TL': (pts['xL'] + pts['dx'], pts['y0']),
        'TC': (pts['xC'] + pts['dx'], pts['y0'] + dyC),
        'TR': (pts['xR'] + pts['dx'], pts['y0'] + dyR)}
    if rotated_w_maze:
        well_xys_dict = {k: (v[1], v[0]) for k, v in well_xys_dict.items()}
    return well_xys_dict


def _bon_maze_geometry(day, epoch):
    rotated_w_maze = False
    if day == 2 and epoch == 1:
        pts = {'xL': 43, 'xC': 75, 'xR': 110, 'dx': 0,
               'y0': 77, 'y1': 153, 'dy': 0}
    elif day == 2 and epoch == 3:
        pts = {'xL': 43, 'xC': 75, 'xR': 110, 'dx': 0,
               'y0': 75, 'y1': 150, 'dy': 0}
    elif day == 2 and epoch == 5:
        pts = {'xL': 80, 'xC': 116, 'xR': 152, 'dx': 3,
               'y0': 222, 'y1': 148, 'dy': 0}
        rotated_w_maze = True
    elif day == 3 and epoch == 1:
        pts = {'xL': 38, 'xC': 72, 'xR': 108, 'dx': 4,
               'y0': 70, 'y1': 145, 'dy': 0}
    elif day == 3 and epoch == 3:
        pts = {'xL': 38, 'xC': 72, 'xR': 105, 'dx': 4,
               'y0': 63, 'y1': 140, 'dy': 5}
    elif day == 3 and epoch == 5:
        pts = {'xL': 84, 'xC': 120, 'xR': 160, 'dx': 5,
               'y0': 225, 'y1': 150, 'dy': -3}
        rotated_w_maze = True
    elif day == 4 and epoch == 1:
        pts = {'xL': 38, 'xC': 72, 'xR': 106, 'dx': 4,
               'y0': 68, 'y1': 148, 'dy': 4}
    elif day == 4 and epoch == 3:
        pts = {'xL': 38, 'xC': 72, 'xR': 106, 'dx': 4,
               'y0': 67, 'y1': 146, 'dy': 5}
    elif day == 4 and epoch == 5:
        pts = {'xL': 85, 'xC': 122, 'xR': 160, 'dx': 5,
               'y0': 225, 'y1': 150, 'dy': -3}
        rotated_w_maze = True
    elif day == 5 and epoch == 1:
        pts = {'xL': 38, 'xC': 71, 'xR': 106, 'dx': 6,
               'y0': 68, 'y1': 148, 'dy': 5}
    elif day == 5 and epoch == 3:
        pts = {'xL': 38, 'xC': 71, 'xR': 106, 'dx': 6,
               'y0': 68, 'y1': 148, 'dy': 5}
    elif day == 5 and epoch == 5:
        pts = {'xL': 85, 'xC': 122, 'xR': 158, 'dx': 6,
               'y0': 225, 'y1': 150, 'dy': -5}
        rotated_w_maze = True
    elif day == 6 and epoch == 1:
        pts = {'xL': 38, 'xC': 71, 'xR': 106, 'dx': 6,
               'y0': 68, 'y1': 148, 'dy': 5}
    elif day == 6 and epoch == 3:
        pts = {'xL': 38, 'xC': 71, 'xR': 106, 'dx': 6,
               'y0': 70, 'y1': 148, 'dy': 3}
    elif day == 6 and epoch == 5:
        pts = {'xL': 85, 'xC': 122, 'xR': 158, 'dx': 6,
               'y0': 225, 'y1': 150, 'dy': -5}
        rotated_w_maze = True
    elif day == 7 and epoch == 1:
        pts = {'xL': 38, 'xC': 71, 'xR': 106, 'dx': 6,
               'y0': 68, 'y1': 148, 'dy': 5}
    elif day == 7 and epoch == 3:
        pts = {'xL': 38, 'xC': 71, 'xR': 106, 'dx': 6,
               'y0': 68, 'y1': 148, 'dy': 5}
    elif day == 7 and epoch == 5:
        pts = {'xL': 85, 'xC': 122, 'xR': 158, 'dx': 6,
               'y0': 225, 'y1': 150, 'dy': -5}
        rotated_w_maze = True
    elif day == 8 and epoch == 1:
        pts = {'xL': 38, 'xC': 71, 'xR': 106, 'dx': 6,
               'y0': 68, 'y1': 152, 'dy': 4}
    elif day == 8 and epoch == 3:
        pts = {'xL': 38, 'xC': 71, 'xR': 106, 'dx': 6,
               'y0': 68, 'y1': 152, 'dy': 4}
    elif day == 8 and epoch == 5:
        pts = {'xL': 85, 'xC': 122, 'xR': 158, 'dx': 6,
               'y0': 223, 'y1': 148, 'dy': -4}
        rotated_w_maze = True
    elif day == 9 and epoch == 1:
        pts = {'xL': 38, 'xC': 71, 'xR': 106, 'dx': 5,
               'y0': 68, 'y1': 152, 'dy': 4}
    elif day == 9 and epoch == 3:
        pts = {'xL': 38, 'xC': 71, 'xR': 106, 'dx': 5,
               'y0': 68, 'y1': 152, 'dy': 4}
    elif day == 9 and epoch == 5:
        pts = {'xL': 85, 'xC': 122, 'xR': 158, 'dx': 6,
               'y0': 225, 'y1': 148, 'dy': -4}
        rotated_w_maze = True
    else:
        return None, None

    return pts, rotated_w_maze


def _con_maze_geometry(day, epoch):
    rotated_w_maze = True  # all tracks are rotated sideways (90 or 270)
    if day in (0, 1, 2) and epoch in (1, 3):
        pts = {'xL': 106, 'xC': 70, 'xR': 33, 'dx': -2,
               'y0': 23, 'y1': 98, 'dy': 1}
    elif day >= 3 and epoch in (1, 3):
        pts = {'xL': 34, 'xC': 71, 'xR': 110, 'dx': 0,
               'y0': 210, 'y1': 134, 'dy': 0}
    elif day >= 3 and epoch == 5:
        pts = {'xL': 106, 'xC': 70, 'xR': 33, 'dx': -2,
               'y0': 23, 'y1': 98, 'dy': 1}
    else:
        return None, None

    return pts, rotated_w_maze


def _cor_maze_geometry(day, epoch):
    rotated_w_maze = False
    if day in (0, 1, 2) and epoch in (1, 3):
        pts = {'xL': 248, 'xC': 282, 'xR': 314, 'dx': 0,
               'y0': 112, 'y1': 190, 'dy': 0}
    elif day >= 3 and epoch in (1, ):
        pts = {'xL': 248, 'xC': 282, 'xR': 310, 'dx': 0,
               'y0': 112, 'y1': 195, 'dy': 0}
    elif day >= 3 and epoch in (3, 5):
        pts = {'xL': 108, 'xC': 148, 'xR': 185, 'dx': 2,
               'y0': 205, 'y1': 132, 'dy': -2}
        rotated_w_maze = True
    else:
        return None, None

    return pts, rotated_w_maze
