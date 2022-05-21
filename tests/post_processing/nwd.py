from post_processing import process_nwd_results


def test_process_nwd_results():
    file_name_str = '18_01_2021_21_59_06_SMOOTH_DISCONTINUOUS_n_prior_layers1.hdf'
    process_nwd_results(file_name_str)
