import numpy as np
from glob import glob
from argparse import ArgumentParser



def main(in_path, needed_shape):
    needed_shape = tuple(needed_shape)
    dataset = sorted(glob(in_path))
    counter = 0
    file_found = False
    corrupted_data = []

    for compression in dataset:
        counter += 1
        data = None
        try:
            data = np.load(compression)['arr_0']
        except KeyError:
            print compression + " has no key 'arr_0'"
            file_found = True
            corrupted_data.append(compression)
        except Exception:
            print compression + " could not be loaded"
            file_found = True
            corrupted_data.append(compression)
        if data is not None:
            if (data.shape[0] != needed_shape[0]) or (data.shape[1] != needed_shape[1]) or (data.shape[2] != needed_shape[2]):
                print compression + " has wrong shape: " + str(data.shape)
                file_found = True
                corrupted_data.append(compression)
        if counter % 100 == 0:
            if counter > 100 and not file_found:
                CURSOR_UP_ONE = '\x1b[1A'
                ERASE_LINE = '\x1b[2K'
                print ""+(CURSOR_UP_ONE + ERASE_LINE) + CURSOR_UP_ONE
            print "Checked " + str(counter) + " files, " + str(len(corrupted_data)) + " corrupted"
            file_found = False

    print "="*100
    print "="*100
    print "="*100
    print "Corrupted data"
    path_list = ""
    for path in corrupted_data:
        path_list = path_list + " " + path
    print path_list

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--inpath", type=str, dest="in_path",
                        required=True, help="Path to datasets")
    parser.add_argument("--shape", nargs='+',type=int, dest="needed_shape",
                        required=True, help="Shape of the datasets on which should be tested")

    args = parser.parse_args()
    main(**vars(args))