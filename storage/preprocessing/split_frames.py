from glob import glob
from scipy.ndimage import imread
import scipy.misc
from argparse import ArgumentParser
import os
import time
from tqdm import tqdm

def main(in_path, out_path, folder_name_base, split_number):
    frames = sorted(glob(os.path.join(in_path, "*")))
    folder = []
    for i in range(split_number):
        folder_name = os.path.join(out_path, folder_name_base) + "_" + str(i)
        folder.append(folder_name)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

    start_time = time.time()
    for index, frame in tqdm(enumerate(frames)):
        img = imread(frame)
        frame_name = frame.split('/',-1)[-1]
        scipy.misc.toimage(img).save(os.path.join(folder[index % split_number], frame_name))
        #if (time.time() - start_time) > 1:
            #print(str(index)+"|"+str(len(frames))+": "+frame+" to "+folder[index % split_number])
            #start_time = time.time()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--inpath", type=str, dest="in_path",
                        required=True, help="Path to directory of the images")
    parser.add_argument("--outpath", type=str, dest="out_path",
                        required=True, help="Path to directory where images should be saved (base directory)")
    parser.add_argument("--basename", type=str, dest="folder_name_base",
                        required=True, help="Base name of the folders in which the splitted images should be saved")
    parser.add_argument("--splits", type=int, dest="split_number",
                        required=True, help="In how many groups the images should be split up ")

    args = parser.parse_args()
    main(**vars(args))
