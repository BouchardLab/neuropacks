import numpy as np
import warnings

from allensdk.core.brain_observatory_cache import BrainObservatoryCache

warnings.simplefilter('ignore')


class Allen():
    """Class to generate design and response matrices from Allen Institute
    Brain Observatory data.

    Attributes
    ----------
    manifest_file : string
        The location of the manifest file for the BrainObservatoryCache.

    boc : BrainObservatoryCache object
        The object containing the cached BrainObservatory data.
    """
    def __init__(self, manifest_file=None):
        self.manifest_file = manifest_file
        self.boc = BrainObservatoryCache(manifest_file=manifest_file)

    def get_ophys_experiment_id(self, experiment_id, stimulus_name):
        """Gets the experiment id for the session corresponding to a desired
        stimulus within an experiment.

        Parameters
        ----------
        experiment_id : int
            The experiment id. Note that this is distinct from the experiment
            id for the specific session (A, B, C) which is often denoted by
            ophys_experiment_id.

        stimulus_name : string
            The stimulus. Currently supports 'static_gratings' or
            'drifting_gratings'.

        Returns
        -------
        ophys_experiment_id : int
            The id for the specific session.
        """
        # see visual coding overview on Allen Institute website
        if stimulus_name == 'drifting_gratings':
            key = 'three_session_A'
        elif stimulus_name == 'static_gratings':
            key = 'three_session_B'
        else:
            raise ValueError('Stimulus name not supported.')

        experiments = self.boc.get_ophys_experiments(
            experiment_container_ids=[experiment_id]
        )

        # get ophys id for session containing the stimulus
        ophys_experiment_id = [
            exp['id'] for exp in experiments if exp['session_type'] == key
        ][0]

        return ophys_experiment_id

    def get_cell_specimen_ids(self, ophys_experiment_id):
        """Gets the cell specimen ids for a specific experiment.

        Parameters
        ----------
        ophys_experiment_id : int
            The id for the specific session.

        Returns
        -------
        cell_specimen_ids : ndarray
            The ids for the cells in the experiment.
        """
        data = self.boc.get_ophys_experiment_data(ophys_experiment_id)
        cell_specimen_ids = data.get_cell_specimen_ids()
        return cell_specimen_ids

    def get_design_matrix(self, experiment_id, stimulus_name, stimulus_key, design):
        """Gets the experiment id for the session corresponding to a desired
        stimulus within an experiment.

        Parameters
        ----------
        experiment_id : int
            The experiment id. Note that this is distinct from the experiment
            id for the specific session (A, B, C) which is often denoted by
            ophys_experiment_id.

        stimulus_name : string
            The stimulus. Currently supports 'static_gratings' or
            'drifting_gratings'.

        stimulus_key : string
            The stimulus dimension over which to construct the design matrix.

        design : string
            The type of design matrix to construct.

        Returns
        -------
        X : nd-array, shape (n_samples, n_features)
            The design matrix.
        """
        # get ophys data object
        ophys_experiment_id = self.get_ophys_experiment_id(
            experiment_id=experiment_id,
            stimulus_name=stimulus_name
        )
        data = self.boc.get_ophys_experiment_data(ophys_experiment_id)

        # get timestamps and responses for all cell specimens
        timestamps, dffs = data.get_dff_traces()
        # get stimulus table and remove NaNs
        table = data.get_stimulus_table(stimulus_name)
        table = table.dropna('index', 'any')
        stimulus = table[stimulus_key].values
        n_samples = stimulus.size

        # the values of the stimulus
        if design == 'values':
            X = stimulus

        # one-hot encoding for unique stimulus values
        elif design == 'one-hot':
            unique = np.unique(stimulus)
            n_features = unique.size
            X = np.zeros((n_samples, n_features))

            for sample in range(n_samples):
                idx = np.asscalar(np.argwhere(stimulus[sample] == unique))
                X[sample, idx] = 1

        else:
            raise ValueError('Incorrect design matrix.')

        return X

    def get_response_matrix(self, experiment_id, stimulus_name):
        """Calculates a responses matrix over cells for a specific stimulus
        from a given experiment.

        Responses are calculated as the mean dF/F over the stimulus
        presentation.

        Parameters
        ----------
        experiment_id : int
            The experiment id. Note that this is distinct from the experiment
            id for the specific session (A, B, C) which is often denoted by
            ophys_experiment_id.

        stimulus_name : string
            The stimulus. Currently supports 'static_gratings' or
            'drifting_gratings'.

        Returns
        -------
        responses : nd-array, shape (n_samples, n_cells)
            The mean dF/F over all stimulus presentations and cell for the
            specified experiment.
        """

        # get ophys data object
        ophys_experiment_id = self.get_ophys_experiment_id(
            experiment_id=experiment_id,
            stimulus_name=stimulus_name
        )
        data = self.boc.get_ophys_experiment_data(ophys_experiment_id)

        # get timestamps and responses for all cell specimens
        timestamps, dffs = data.get_dff_traces()
        # get stimulus table and remove NaNs
        table = data.get_stimulus_table(stimulus_name)
        table = table.dropna('index', 'any')
        starts = table['start'].values
        ends = table['end'].values
        n_samples = table.shape[0]
        n_cells = dffs.shape[0]

        # get mean responses to each stimulus
        if stimulus_name == 'drifting_gratings':
            responses = np.array([
                np.mean(dffs[:, starts[idx]:ends[idx]], axis=1)
                for idx in range(n_samples)])
        elif stimulus_name == 'static_gratings':
            responses = np.zeros((n_samples, n_cells))
            for sample in range(n_samples):
                start_idx = int(starts[sample])
                end_idx = np.argwhere(
                    ((timestamps[start_idx] + 0.5) - timestamps) < 0
                ).ravel()[0]
                responses[sample] = np.mean(dffs[:, start_idx:end_idx], axis=1)

        return responses
