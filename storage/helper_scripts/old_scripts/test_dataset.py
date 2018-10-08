import numpy as np
from glob import glob

DATA_SET = sorted(
    glob("/lhome/phlippe/dataset/TwoHourSequence_crop/deepTracking96x96/*.npz"))
NEEDED_SHAPE = (96, 96, 200)

counter = 0
corrupted_data = []

for compression in DATA_SET:
    counter += 1
    data = None
    try:
        data = np.load(compression)['arr_0']
    except KeyError:
        print compression + " has no key 'arr_0'"
        corrupted_data.append(compression)
    except Exception:
        print compression + " could not be loaded"
        corrupted_data.append(compression)
    if data is not None:
        if (data.shape[0] != NEEDED_SHAPE[0]) or (data.shape[1] != NEEDED_SHAPE[1]) or (data.shape[2] != NEEDED_SHAPE[2]):
            print compression + " has wrong shape: " + str(data.shape)
            corrupted_data.append(compression)
    if counter % 100 == 0:
        print "Checked " + str(counter) + " files, " + str(len(corrupted_data)) + " corrupted"

print "="*100
print "="*100
print "="*100
print "Corrupted data"
print corrupted_data