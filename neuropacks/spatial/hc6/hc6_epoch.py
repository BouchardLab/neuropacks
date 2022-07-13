import os
import numpy as np
import pandas as pd

from scipy.io import loadmat

from .hc6 import HC6
from .hc6_geom import load_annotated_well_xys
from .hc6_pos import linearize_position_single_traj, trialize_epoch_pos, get_linpos_nodes

from neuropacks.spatial.utils import get_binned_times_rates_pos


class HC6Epoch:
    def __init__(self, *, path, animal, day, epoch, prefix=None, verbose=True):
        # self.animal = animal
        self.day = day
        self.epoch = epoch
        self.animal_dir = os.path.join(path, f'{animal}/')
        self.pack = HC6(self.animal_dir, base=prefix)
        self.animal = self.pack.base
        self.verbose = verbose

        self.epoch_tag_long = (f'hc6_{self.animal}_'
                               f'day{(self.day + 1):02d}_epoch{(self.epoch + 1)}')

        if day not in self.pack.valid_days:
            raise ValueError(f'day {day}: not a valid day')

        # --- load task and positions ---

        task_file = os.path.join(self.animal_dir,
                                 f'{self.animal}task{(day + 1):02d}.mat')
        day_task = loadmat(task_file, struct_as_record=False)['task'][0, day]
        try:
            epoch_task = day_task[0, epoch][0, 0]
        except IndexError:
            # bad epoch
            print(f'bad epoch: {self.epoch_tag_long}')
            self.task = None
            self.bad_epoch = True
            return

        self.task = {'type': getattr(epoch_task, 'type', [None])[0],
                     'description': getattr(epoch_task, 'description', [None])[0],
                     'environment': getattr(epoch_task, 'environment', [None])[0]}

        self.bad_epoch = False

        # self.spike_times_by_units = self.pack.spike_times[day][epoch]
        self.pos = self.pack.positions[day][epoch]

        try:
            self.well_xys_dict = load_annotated_well_xys(animal=self.animal,
                                                         day=self.day, epoch=self.epoch)
        except ValueError as e:
            print(e)
            self.well_xys_dict = None

        if self.well_xys_dict is not None:
            self.traj_times_table = self.make_traj_times_table()
        else:
            self.traj_times_table = None

        # --- load spiking units ---

        cell_info_file = os.path.join(self.animal_dir, f'{self.animal}cellinfo.mat')
        cell_info = loadmat(cell_info_file, struct_as_record=False)['cellinfo']
        spikes_file = os.path.join(self.animal_dir,
                                   f'{self.animal}spikes{(day + 1):02d}.mat')
        spikes = loadmat(spikes_file, struct_as_record=False)['spikes']

        epoch_cell_info = cell_info[0, day][0, epoch]
        epoch_spikes = spikes[0, day][0, epoch]
        self.units_table = self.make_epoch_units_table(epoch_cell_info, epoch_spikes)

        # exclude None or empty string, and store unique list of regions
        all_regions = self.units_table['region'].tolist()
        all_regions = [reg for reg in all_regions if reg is not None]
        self.all_regions = np.unique(all_regions).tolist()

    def make_traj_times_table(self):
        # return traj_times_table
        trials = trialize_epoch_pos(self.pos, self.well_xys_dict)
        return trials

    def make_epoch_units_table(self, epoch_cell_info, epoch_spikes):
        units_table_src = []
        for tet, tet_info in enumerate(epoch_cell_info[0]):
            for cell, unit_info in enumerate(tet_info[0]):
                try:
                    area = unit_info[0, 0].area[0]
                # except AttributeError:
                except Exception:
                    area = None
                try:
                    numspikes = unit_info[0, 0].numspikes[0, 0]
                # except AttributeError:
                except Exception:
                    numspikes = None

                if area is None and numspikes is None:
                    # skip this unit
                    continue

                unit_spikes = epoch_spikes[0, tet][0, cell]
                num_spikes_in_data = unit_spikes[0, 0].data.shape[0]
                if num_spikes_in_data > 0:
                    spike_times = unit_spikes[0, 0].data[:, 0]  # first column
                else:
                    spike_times = np.array([])
                if numspikes is not None and numspikes != num_spikes_in_data:
                    print('WARNING: numspikes mismatch')
                # print((tet, cell, area, numspikes, num_spikes_in_data))
                row = {'unit_key': f'{tet}_{cell}', 'tet': tet, 'cell': cell,
                       'region': area,
                       'num_spikes': num_spikes_in_data,
                       'spike_times': spike_times}
                units_table_src.append(row)
        units_table = pd.DataFrame(units_table_src)
        return units_table

    def parse_and_bin_traj(self, all_regions=None, **binning_kwargs):
        # if target_region is not specified
        all_regions = all_regions or self.all_regions

        parsed_binned_dict = {}
        for region in all_regions:
            if self.verbose:
                print(region)
            parsed_binned = self.parse_and_bin_traj_region(region, **binning_kwargs)
            parsed_binned_dict[region] = parsed_binned
        return parsed_binned_dict

    def parse_and_bin_traj_region(self, target_region,
                                  spike_threshold=0, **binning_kwargs):

        units_table_filtered = self.units_table[
            (self.units_table['region'] == target_region) &
            (self.units_table['num_spikes'] >= spike_threshold)]
        unit_spiking_times = units_table_filtered['spike_times'].copy().to_numpy()

        # binning_kwargs = dict(
        #     bin_width=bin_width, bin_rep=bin_rep,
        #     boxcox=boxcox, log=log,
        #     filter_fn=filter_fn, **filter_kwargs)

        trials = self.traj_times_table
        num_traj = trials.shape[0]

        # first parse by trajectories, then bin each traj
        parsed_binned = []
        for i_trial in range(num_traj):
            t_start = trials.iloc[i_trial]['t_start']
            t_stop = trials.iloc[i_trial]['t_stop']
            unit_spiking_times_tr = [times[np.logical_and(times >= t_start, times < t_stop)]
                                     for times in unit_spiking_times]

            t_tr, x_tr, y_tr, traj_type, xy_nodes = self._get_trial(i_trial)
            linpos_tr, _, _, l_nodes = linearize_position_single_traj(x_tr, y_tr, xy_nodes)

            t_binned, spike_rates, pos_binned = get_binned_times_rates_pos(
                t_tr, unit_spiking_times_tr, linpos_tr,
                **binning_kwargs)
            t_binned_rel = t_binned - t_binned[0]

            traj_info = trials.iloc[i_trial].to_dict()

            # collect and return binned data
            binned_tr = {'times': t_binned, 'spike_rates': spike_rates, 'pos': pos_binned,
                         'times_rel': t_binned_rel,
                         'pos_corr': pos_binned,  # duplicate
                         'traj': {'index': i_trial, **traj_info},
                         'binning_kwargs': {spike_threshold: spike_threshold,
                                            **binning_kwargs}}
            parsed_binned.append(binned_tr)
        return parsed_binned

    def _get_trial(self, i_trial):
        trials = self.traj_times_table
        pos = self.pos

        t_start = trials.iloc[i_trial]['t_start']
        t_stop = trials.iloc[i_trial]['t_stop']
        sl = np.logical_and(pos['time'] >= t_start, pos['time'] < t_stop)

        well_start = trials.iloc[i_trial]['well_start']
        well_stop = trials.iloc[i_trial]['well_stop']
        traj_type = f'{well_start}{well_stop}'

        node_names = (well_start, f'T{well_start}', f'T{well_stop}', well_stop)
        x_nodes = [self.well_xys_dict[well][0] for well in node_names]
        y_nodes = [self.well_xys_dict[well][1] for well in node_names]
        xy_nodes = [(x, y) for x, y in zip(x_nodes, y_nodes)]

        return pos['time'][sl], pos['x'][sl], pos['y'][sl], traj_type, xy_nodes

    def get_traj_type(self, i_trial):
        trials = self.traj_times_table
        well_start = trials.iloc[i_trial]['well_start']
        well_stop = trials.iloc[i_trial]['well_stop']
        traj_type = f'{well_start}{well_stop}'
        return traj_type

    # --- for analysis pipeline ---

    def get_linpos_nodes(self, traj_type):
        boundaries, idx_choice_point = get_linpos_nodes(self.well_xys_dict, traj_type)
        return boundaries, idx_choice_point

    def get_select_traj_type_binned(self, *, target_region, traj_type, **binning_kwargs):
        boundaries, _ = self.get_linpos_nodes(traj_type)
        parsed_binned = self.parse_and_bin_traj_region(target_region, **binning_kwargs)

        # unpacking collect_trajs_binned
        all_trajs_binned = []
        traj_durations = []
        for i_traj, traj_binned in enumerate(parsed_binned):
            if (traj_type is not None) and (self.get_traj_type(i_traj) != traj_type):
                continue
            all_trajs_binned.append(traj_binned)
            tt_rel = traj_binned['times_rel']
            traj_durations.append(tt_rel[-1])

        return all_trajs_binned, traj_durations, boundaries
