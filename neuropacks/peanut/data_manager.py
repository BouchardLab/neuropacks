import os
import numpy as np
import pandas as pd
from copy import deepcopy

from .task import get_traj_outcomes

from neuropacks.utils.binning import create_bins, bin_spike_times
from neuropacks.utils.io import load_pickle
from neuropacks.utils.signal import downsample_by_interp


class DayDataManager():
    ''' Loads and handles a day's data from forkmaze experiment
    (temporary preprocessed .obj format)
    '''
    def __init__(self, path, *, animal, day, verbose=1):
        '''
        path: (str) path to .obj file or the enclosing folder
        animal: (str) animal name
        day: (int) day number in experiment (like 3 or 14)
        verbose: (bool or int) verbose level
            0: no message displayed
            1: only minimal messages
            2: more detailed messages for debugging etc.
        '''
        self.animal = animal
        self.day = day
        self.day_key = f'{animal}_day{day}'

        # check data path
        dirname, data_basename = os.path.split(path)
        if '.obj' not in data_basename:
            dirname = path
            data_basename = f'data_dict_{self.day_key}.obj'
            # data_basename = f'loaded_data_{animal}_day{day}.obj'  # old version
        data_path = os.path.join(dirname, data_basename)
        self.data_path = data_path
        self.data_dir = dirname

        # self.load()
        self.verbose = verbose

    def load_day(self, scan_all_epochs=True, return_data=True):
        # load the preprocessed data from the day
        day_data_dict = load_pickle(self.data_path)

        # also load information for getting linearized position,
        # if available in the same folder
        self.lin_path = os.path.join(
            self.data_dir, f'linearization_dict_{self.day_key}.obj')
        if os.path.isfile(self.lin_path):
            day_lin_dict = load_pickle(self.lin_path)
        else:
            day_lin_dict = {}

        # and also the parsed trajectories
        self.traj_path = os.path.join(
            self.data_dir, f'trajectory_dict_{self.day_key}.obj')
        if os.path.isfile(self.traj_path):
            day_traj_dict = load_pickle(self.traj_path)
        else:
            day_traj_dict = {}

        if scan_all_epochs:
            # this sets self.day_summary_dict
            self.scan(day_data_dict, day_lin_dict, day_traj_dict)

        # by default do not store the full data as attributes
        # (instead load specific epochs)
        if return_data:
            return day_data_dict, day_lin_dict, day_traj_dict

    def scan(self, day_data_dict, day_lin_dict, day_traj_dict):
        day_summary_dict = {}
        day_summary_dict['day'] = {'num_epochs': len(day_data_dict)}
        day_summary_dict['epochs'] = {}

        # epoch_num_key_dict = {}
        day_traj_times_table_dict = {}
        all_nt_unit_pairs_day = []
        for epoch_key, epoch_data in day_data_dict.items():
            epoch_summary = {}

            # basic metadata
            epoch_meta = epoch_data['identification']
            # epoch_num = epoch_meta['epoch']
            # epoch_num_key_dict[epoch_num] = epoch_key
            for key in ('rat_name', 'day', 'epoch', 'day_string',
                        'environment', 'rule',
                        'environment_abstract', 'rule_abstract'):
                epoch_summary[key] = epoch_meta.get(key, None)
            epoch_type = 'run' if (epoch_summary['environment'] is not None) else 'sleep'
            epoch_summary['epoch_type'] = epoch_type

            # scan unit spiking data
            unique_nt_unit_pairs, unique_unit_labels = self._scan_unit_labels(epoch_data['spike_times'])
            epoch_summary['num_units'] = len(unique_unit_labels)
            epoch_summary['unique_unit_labels'] = unique_unit_labels
            all_nt_unit_pairs_day.extend(unique_nt_unit_pairs)

            # scan parsed trajectories
            traj_dict = day_traj_dict.get(epoch_key, None)
            if traj_dict is None:
                epoch_summary['num_trajs'] = None
            else:
                traj_times_dict = traj_dict['trajectory_start_end_times_in_pos_df']
                traj_times_table = self._make_traj_times_table(traj_times_dict)
                day_traj_times_table_dict[epoch_key] = traj_times_table
                epoch_summary['num_trajs'] = len(traj_times_table)

            day_summary_dict['epochs'][epoch_key] = epoch_summary

        unique_nt_unit_pairs_day = list(np.unique(all_nt_unit_pairs_day, axis=0))
        unique_unit_labels_day = []
        for nt_ind, unit_ind in unique_nt_unit_pairs_day:
            unit_label = f'nt{nt_ind}_u{unit_ind}'
            # unit_label = (nt_ind, unit_ind)
            unique_unit_labels_day.append(unit_label)

        day_summary = day_summary_dict['day']
        # day_summary['all_epochs'] = epoch_num_key_dict
        day_summary['num_units'] = len(unique_unit_labels_day)
        day_summary['unique_unit_labels'] = unique_unit_labels_day

        self.day_summary_dict = day_summary_dict
        self.day_traj_times_table_dict = day_traj_times_table_dict

    def _scan_unit_labels(self, spike_times):
        ''' spike sorting is done for each ntrode and across entire day;
        each (nt, unit) pair makes a unique unit label.
        '''
        all_nt_unit_pairs = []
        for probe_key, spike_times_by_unit in spike_times.items():
            # split probe key
            # probe_tag looks like 'tetrode_nt1' or 'probe_nt25'
            split_probe_key = probe_key.split('_nt')
            if len(split_probe_key) != 2:
                raise ValueError('unknown format for probe key')
            nt_ind = int(split_probe_key[1])

            nt_unit_pairs = []
            for unit_key, times in spike_times_by_unit.items():
                if len(times) > 0:
                    unit_ind = int(unit_key.split('unit_')[1])
                    nt_unit_pairs.append((nt_ind, unit_ind))
            all_nt_unit_pairs.extend(nt_unit_pairs)
        unique_nt_unit_pairs = np.unique(all_nt_unit_pairs, axis=0)
        unique_unit_labels = []
        for nt_ind, unit_ind in unique_nt_unit_pairs:
            unique_unit_labels.append(f'nt{nt_ind}_u{unit_ind}')
        return unique_nt_unit_pairs, unique_unit_labels

    def _make_traj_times_table(self, traj_times_dict):
        data_list = []
        for well_pair, start_end_times_list in traj_times_dict.items():
            for start_end_times in start_end_times_list:
                start_well, end_well = tuple(well_pair)
                traj_type = f'{start_well[0].upper()}{end_well[0].upper()}'
                row = {'start_time': start_end_times[0],
                       'end_time': start_end_times[1],
                       'start_well': start_well,
                       'end_well': end_well,
                       'traj_type': traj_type}
                data_list.append(row)

        column_names = list(row.keys())
        traj_times_table = pd.DataFrame(data_list).sort_values(by=column_names,
                                                               ignore_index=True)
        return traj_times_table

    def print_all_epochs_behavior_summary(self, run_epochs_only=True, long=False):
        ''' ad hoc helper for exploratory analysis '''
        if not hasattr(self, 'day_summary_dict'):
            self.load_day(scan_all_epochs=True, return_data=False)

        indent = '  - '
        num_epochs = self.day_summary_dict['day']['num_epochs']
        print(f'{self.day_key} ({num_epochs} epochs)')
        for epoch_key, epoch_summary in self.day_summary_dict['epochs'].items():
            epoch_type = epoch_summary['epoch_type']
            if run_epochs_only and (epoch_type != 'run'):
                continue
            epoch_num = epoch_summary['epoch']
            print(f'epoch {epoch_num}')
            print('{}{}, {} ({}-alt)'.format(indent,
                epoch_summary['environment_abstract'],
                epoch_summary['rule_abstract'],
                epoch_summary['rule'][0].upper()))

            if long:
                _, traj_types_tally = self._count_epoch_trajs(epoch_key)
                tally_str = f'{indent}traj types: '
                for traj_type, cnt in traj_types_tally:
                    tally_str += f'{traj_type}({cnt}), '
                tally_str = tally_str.rstrip(', ')
                print(tally_str)

            num_trajs, num_success, choice_reward = self._summarize_choice_reward(epoch_key)
            success_rate = int(100 * num_success / num_trajs)
            print(f'{indent}success rate = {num_success}/{num_trajs} = {success_rate}%')

    def _count_epoch_trajs(self, epoch_key=None):
        if epoch_key is None and hasattr(self, 'epoch_key'):
            # in case this is called from EpochDataManager
            epoch_key = self.epoch_key

        traj_times_table = self.day_traj_times_table_dict[epoch_key]
        if traj_times_table is None:
            # not a run epoch
            return None, None
        traj_types = traj_times_table['traj_type'].tolist()
        unique_traj_types, traj_counts = np.unique(traj_types, return_counts=True)
        traj_types_tally = [(v, c) for v, c in zip(unique_traj_types, traj_counts)]
        return traj_types, traj_types_tally

    def _summarize_choice_reward(self, epoch_key=None):
        if epoch_key is None and hasattr(self, 'epoch_key'):
            # in case this is called from EpochDataManager
            epoch_key = self.epoch_key

        epoch_summary = self.day_summary_dict['epochs'][epoch_key]
        rule = epoch_summary['rule']

        traj_times_table = self.day_traj_times_table_dict[epoch_key]
        if traj_times_table is None:
            # not a run epoch
            return None, None, None
        traj_types = traj_times_table['traj_type'].tolist()
        num_trajs = len(traj_types)

        traj_outcomes = get_traj_outcomes(traj_types, rule)
        choice_reward = [cr for cr in zip(traj_types, traj_outcomes)]
        num_success = np.sum(np.array(traj_outcomes) > 0)  # outcome 1 or 2
        return num_trajs, num_success, choice_reward


class EpochDataManager(DayDataManager):
    ''' Loads and handles an epoch's data from forkmaze experiment
    (temporary preprocessed .obj format)
    '''
    def __init__(self, path, *, animal, day, epoch, verbose=1):
        '''
        path: (str) path to .obj file or the enclosing folder
        animal: (str) animal name
        day: (int) day number in experiment (like 3 or 14)
        epoch: (int) epoch number in this day (even-numbered epochs are run epochs)
        verbose: (bool or int) verbose level
            0: no message displayed
            1: only minimal messages
            2: more detailed messages for debugging etc.
        '''
        DayDataManager.__init__(self, path=path, animal=animal, day=day, verbose=verbose)
        self.epoch = epoch
        self.epoch_key = f'{self.animal}_day{self.day}_epoch{self.epoch}'

        self.load()
        self.prep()

    def load(self):
        # load (but not store) the full day's data
        # when scan_all_epochs is True, this sets self.day_summary_dict
        day_data_dict, day_lin_dict, day_traj_dict = self.load_day(scan_all_epochs=True)

        # store epoch data
        self.data_dict = day_data_dict[self.epoch_key]
        self.spike_times = self.data_dict['spike_times']
        self.meta = self.data_dict['identification']
        self.pos = self.data_dict['position_df']

        # linearization and trajectory information: for run epochs only
        self.lin_dict = day_lin_dict.get(self.epoch_key, None)
        self.traj_dict = day_traj_dict.get(self.epoch_key, None)
        self.traj_times_table = self.day_traj_times_table_dict.get(self.epoch_key, None)

        # list of target brain regions: normally ['HPc', 'OFC', 'mPFC'] for peanut
        self.all_regions = np.unique(
            [v for _, v in self.meta['nt_brain_region_dict'].items()]).tolist()

        # detect if run/sleep epoch
        self.is_run_epoch = (self.meta['environment'] is not None)
        self.epoch_type = 'run' if self.is_run_epoch else 'sleep'

        # more descriptive tags
        self.epoch_tag = self.epoch_key.replace('_', '-')
        if self.is_run_epoch:
            env_code = self.meta['environment_abstract']
            task_code = '{} ({}-alt)'.format(self.meta['rule_abstract'],
                                             self.meta['rule'][0].upper())
            self.epoch_tag_long = f'{self.epoch_tag}, {env_code}, {task_code}'
        else:
            self.epoch_tag_long = f'{self.epoch_tag} (sleep)'
        self.meta['epoch_tag'] = self.epoch_tag
        self.meta['epoch_tag_long'] = self.epoch_tag_long

    def prep(self):
        # collect all unit spiking times
        self.units_table, self.spike_times_range = self._collect_all_units()

        if self.is_run_epoch:
            # first timestamp for the epoch (for now from position data)
            self.t0 = self.pos['time'].values[0]
        else:
            self.t0 = self.spike_times_range[0]

    def get_traj_type(self, i_traj):
        return self.traj_times_table.iloc[i_traj]['traj_type']

    def collect_binned_parsed(epoch_kwargs, traj_type, all_regions=None,
                              return_epoch_tag=False,
                              verbose=True, **kwargs):
        # adapted from: analysis.collect_trajs_by_type_all_regions
        all_regions = all_regions or self.all_regions
        collected_trajs_by_region = []
        for target_region in all_regions:
            epoch_data = EpochDataManager(**epoch_kwargs, verbose=verbose)
            if verbose:
                print(epoch_data.epoch_tag_long)
                print(target_region)

            binning_kwargs = deepcopy(DEFAULT_BINNING_KWARGS)
            binning_kwargs.update(**kwargs)
            binned_parsed = epoch_data.bin_and_parse_traj(
                target_region=target_region, **binning_kwargs)

            # coll is a tuple (all_trajs_binned, traj_durations, boundaries)
            coll = collect_trajs_binned(epoch_data, binned_parsed, traj_type)
            collected_trajs_by_region.append(coll)
        # if return_epoch_tag:
        #     return collected_trajs_by_region, all_regions, epoch_data.epoch_tag_long
        # return collected_trajs_by_region, all_regions
        self.all_binned_traj_dict = collected_trajs_by_region

    def bin_and_parse_traj(self, target_region=None, all_regions=None, **binning_kwargs):
        if target_region is not None:
            # old behavior
            return self.bin_and_parse_traj_region(target_region, **binning_kwargs)

        # if target_region is not specified
        all_regions = all_regions or self.all_regions
        binned_parsed_dict = {}
        for region in all_regions:
            if self.verbose:
                print(region)
            binned_parsed = self.bin_and_parse_traj_region(region, **binning_kwargs)
            binned_parsed_dict[region] = binned_parsed
        return binned_parsed_dict

    def bin_and_parse_traj_region(self, target_region, **binning_kwargs):
        binned = self.bin_region(target_region, **binning_kwargs)

        num_traj = self.traj_times_table.shape[0]
        binned_parsed = []
        for traj_ind in range(num_traj):
            binned_traj = self._parse_traj(binned, traj_ind)
            if binning_kwargs is not None:
                binned_traj['binning_kwargs'] = deepcopy(binning_kwargs)
            binned_parsed.append(binned_traj)
        return binned_parsed

    def bin(self, target_region=None, all_regions=None, **binning_kwargs):
        if target_region is not None:
            # old behavior
            return self.bin_region(target_region, **binning_kwargs)

        # if target_region is not specified
        all_regions = all_regions or self.all_regions
        binned_dict = {}
        for region in all_regions:
            if self.verbose:
                print(region)
            binned = self.bin_region(region, **binning_kwargs)
            binned_dict[region] = binned
        return binned_dict

    def bin_region(self, target_region, spike_threshold=0,
            bin_width=200, bin_rep='left',
            boxcox=0.5, log=None,
            filter_fn='none', **filter_kwargs):
        '''
        spike_threshold: int
            throw away neurons that spike less than the threshold during the epoch
            default value is 0 (keep all units, including those with 0 spikes)
        bin_width:     float
            Bin width for binning spikes. Note the behavior is sampled at 25ms
        bin_type : str
            Whether to bin spikes along time or position. Currently only time supported
        boxcox: float or None
            Apply boxcox transformation
        filter_fn: str
            Check filter_dict
        filter_kwargs
            keyword arguments for filter_fn
        '''

        # filter by target region and spike threshold
        units_table_filtered = self.units_table[
            (self.units_table['nt_region'] == target_region) &
            (self.units_table['num_spikes'] >= spike_threshold)]
        unit_spiking_times = units_table_filtered['spike_times'].copy().to_numpy()

        if self.is_run_epoch:
            t = self.pos['time'].values
            pos_linear = self.pos['position_linear'].values
        else:
            t = self.spike_times_range  # for the rates, just the range is enough
            pos_linear = None           # this will return pos_binned = None

        t_binned, spike_rates, pos_binned = get_binned_times_rates_pos(
            t, unit_spiking_times, pos_linear,
            bin_width=bin_width, bin_rep=bin_rep,
            boxcox=boxcox, filter_fn=filter_fn, **filter_kwargs)

        # collect and return binned data
        dat = {}
        dat['times'] = t_binned
        dat['spike_rates'] = spike_rates
        if pos_binned is not None:
            dat['pos'] = pos_binned
        return dat

    def _collect_all_units(self):
        ''' Collect all single unit spiking times from epoch, by probe '''
        unique_unit_labels = self.day_summary_dict['day']['unique_unit_labels']
        unique_unit_ids = {unit_label: i for i, unit_label
                           in enumerate(unique_unit_labels)}

        t_min = np.inf
        t_max = -np.inf

        units_summary_list = []
        for probe_key, times_by_unit in self.spike_times.items():
            nt_ind = int(probe_key.split('_nt')[1])
            nt_region = self.meta['nt_brain_region_dict'][f'nt{nt_ind}']
            for unit_key, times in times_by_unit.items():
                unit_ind = int(unit_key.split('unit_')[1])
                unit_label = f'nt{nt_ind}_u{unit_ind}'
                # unit_label = (nt_ind, unit_ind)
                if unit_label not in unique_unit_labels:
                    if len(times) == 0:
                        continue
                    raise RuntimeError('something wrong with unique_unit_labels')
                row = {'nt_ind': nt_ind, 'unit_ind': unit_ind,
                       'unit_label': unit_label,
                       'unique_id': unique_unit_ids[unit_label],
                       'nt_region': nt_region,
                       'num_spikes': len(times),
                       'spike_times': times,
                       }
                units_summary_list.append(row)
                if len(times) > 0:
                    t_min = min(t_min, np.min(times))
                    t_max = max(t_max, np.max(times))
        units_table = pd.DataFrame(units_summary_list)
        spike_times_range = (t_min, t_max)
        return units_table, spike_times_range

    def _parse_traj(self, binned, traj_ind):
        t = binned['times']

        traj_info = self.traj_times_table.iloc[traj_ind].to_dict()
        traj_interval = (traj_info['start_time'], traj_info['end_time'])
        t_mask = (t >= traj_interval[0]) & (t <= traj_interval[1])

        rt = binned['times'][t_mask]
        rx = binned['spike_rates'][t_mask, :]
        ru = binned['pos'][t_mask]
        binned_traj = {'times': rt, 'spike_rates': rx, 'pos': ru}
        binned_traj['traj'] = dict(index=traj_ind, **traj_info)
        return binned_traj


class Peanut_SingleEpoch(EpochDataManager):
    def __init__(self, path, *, day, epoch):
        # Note: for peanut_day14, the rat is running during even numbered epochs.
        super().__init__(path, animal='peanut', day=day, epoch=epoch)


def get_binned_times_rates_pos(t, unit_spiking_times, pos_linear,
                               bin_width=200, bin_rep='left',
                               boxcox=0.5, filter_fn='none', **filter_kwargs):
    '''
    bin_width : float
        Bin width for binning spikes. Note the behavior is sampled at 25ms
    bin_type : str
        Whether to bin spikes along time or position. Currently only time supported
    boxcox: float or None
        Apply boxcox transformation
    filter_fn: str
        Check filter_dict
    filter_kwargs
        keyword arguments for filter_fn
    '''
    # (ad hoc) check if time variables are in ms; if so, convert to seconds
    if bin_width >= 10:
        # guessing that this is in ms; convert to s
        bin_width /= 1000
    if 'sigma' in filter_kwargs:
        if filter_kwargs['sigma'] >= 10:
            # guessing that this is in ms; convert to seconds
            filter_kwargs['sigma'] /= 1000

    bins = create_bins(t_start=t[0], t_end=t[-1], bin_width=bin_width,
                       bin_type='time')

    t_binned = get_binned_times(bins, bin_rep=bin_rep)

    # get spike rates time series from unit spike times, by binning in time
    spike_rates = get_spike_rates(unit_spiking_times, bins,
                                  filter_fn=filter_fn,
                                  boxcox=boxcox, **filter_kwargs)

    if pos_linear is not None:
        # downsample behavior to align with the binned spike rates
        pos_binned = downsample_by_interp(pos_linear, t, t_binned)
    else:
        pos_binned = None

    return t_binned, spike_rates, pos_binned


def get_binned_times(bins, bin_rep='center'):
    # assign a representative time value to each bin
    if bin_rep == 'center':
        return (bins[1:] + bins[:-1]) / 2  # midpoints
    elif bin_rep == 'left':
        return bins[:-1]    # left endpoints
    elif bin_rep == 'right':
        return bins[1:]     # right endpoints

    raise ValueError('bin_rep should be one of (center, left, right); '
                     f'got {bin_rep}.')


def get_spike_rates(unit_spiking_times, bins,
                    boxcox=0.5, log=None,
                    filter_fn='none', **filter_kwargs):
    if filter_fn == 'none':
        filter = {}
    elif filter_fn == 'gaussian':
        bin_width = bins[1] - bins[0]
        sigma = filter_kwargs['sigma'] / bin_width  # convert to unit of bins
        # sigma = min(1, sigma) # -- ??
        filter = {'gaussian': sigma}
    else:
        raise ValueError(f'unknown filter_fn: got {filter_fn}')

    transform = {'boxcox': boxcox, 'log': log}

    spike_rates = bin_spike_times(unit_spiking_times, bins,
                                  transform=transform, filter=filter)
    return spike_rates
