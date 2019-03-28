import h5py
import neuropacks as packs


def generate_nhp_dataset(input_path, output_path, bin_width=0.5):
    nhp = packs.NHP(data_path=input_path)

    # extract datasets
    M1 = nhp.get_response_matrix(bin_width=bin_width)
    S1 = nhp.get_response_matrix(bin_width=bin_width)

    output = h5py.File(output_path, 'w')
    output['M1'] = M1
    output['S1'] = S1
    output.close()
