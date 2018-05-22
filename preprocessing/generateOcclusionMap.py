"""
C++ Code from Sensor Fusion:
==============================================================================================================================
tResult cDOGMaClusteringFilter::computeFreespaceContour(dgm::DynamicGrid& dynamic_grid,
                                                        Clusters_t&   f_clusters)
{
    startClockAutoStop(__FUNCTION__);

    // get grid map size
    uint32_t num_cells_per_edge_ui = dynamic_grid.GetNumCellsPerEdge();
    const float num_cells_per_edge_half_f = (float)num_cells_per_edge_ui / 2.0f;

    double grid_cell_size = dynamic_grid.GetGridCellSize();
    const float grid_cell_size_inv_f = 1.0f / grid_cell_size;

    const float radial_res_meter_f = GetPropertyFloat(kFSRadialResolutionStr);
    const float radial_limit_meter_f = GetPropertyFloat(kFSRadialLimitStr);
    const float angular_res_degree_f = GetPropertyFloat(kFSAngularResolutionStr);
    const float angular_res_rad_f = angular_res_degree_f * M_PI / 180.0f;
    const float two_pi_f = 2.0f * M_PI;

    // check parameters
    if (angular_res_degree_f <= 0.0f) {
        LOG_WARN("angular resolution must be larger than zero! cannot compute freespace contour...");
        RETURN_ERROR(ERR_FAILED);
    }
    if (radial_res_meter_f <= 0.0f) {
        LOG_WARN("radial resolution must be larger than zero! cannot compute freespace contour...");
        RETURN_ERROR(ERR_FAILED);
    }
    if (radial_limit_meter_f <= 0.0f) {
        LOG_WARN("radial max distance must be larger than zero! cannot compute freespace contour...");
        RETURN_ERROR(ERR_FAILED);
    }

    Cluster_t freespace_cluster;
    freespace_cluster.valid_b = true;

    const float occ_thresh_f = GetPropertyFloat(kGridMapThresholdStr);

    // iterate over all angles (0-360 degree)
    for (float ang_f = 0.0f; ang_f < two_pi_f; ang_f += angular_res_rad_f) {

        // iterate over all radial distances within this angle
        for (float radial_dist_f = 0.0f; radial_dist_f < radial_limit_meter_f; radial_dist_f += radial_res_meter_f) {

            // compute cell coordinates in grid map
            const float x_f = radial_dist_f * cos(ang_f) * grid_cell_size_inv_f + num_cells_per_edge_half_f;
            const int x_i = std::max(0, std::min((int)num_cells_per_edge_ui-1, (int)x_f));
            const float y_f = radial_dist_f * sin(ang_f) * grid_cell_size_inv_f + num_cells_per_edge_half_f;
            const int y_i = std::max(0, std::min((int)num_cells_per_edge_ui-1, (int)y_f));
            const uint32_t grid_idx_ui = x_i + y_i * num_cells_per_edge_ui;

            // get occupancy
            const float occ_f = (dynamic_grid.isValid(grid_idx_ui))?
                (dynamic_grid.GetOccupancyMassDyn(grid_idx_ui)):(0);

            // last iteration in inner loop?
            const bool limit_reached_b = radial_dist_f > radial_limit_meter_f - 1.5f*radial_res_meter_f;

            if (occ_f > occ_thresh_f || limit_reached_b) {
                freespace_cluster.contour_indices_v.push_back(grid_idx_ui);

                // leave inner loop over radial distances
                radial_dist_f = radial_limit_meter_f;
            }
        }
    }

    f_clusters.push_back(freespace_cluster);

    RETURN_NOERROR;
}
"""
import os
import numpy as np
from glob import glob
from scipy.ndimage import imread
import scipy.misc
import math
import _thread
import time
from argparse import ArgumentParser
from tqdm import tqdm



# Constants for calculation (if you need more precise data you can raise the resolution but it will take longer to calculate)
two_pi_f = 2 * math.pi
angular_res_rad_f = two_pi_f/900.0
radial_res_meter_f = 0.2
radial_limit_meter_f = 73
grid_cell_size = 0.4
grid_cell_size_inv_f = 1 / grid_cell_size
occ_thresh_f = 96

# Multithreading parameter
thread_number = 0
max_thread_number = 0

def createOcclusionMap(gridmap, save_path, max_occluded_steps=1):
    """
    Calculates occlusion map based on the given gridmap. The occlusion map will be saved as image to the given save path ( 1 - visible, 0 - occluded).
    ===========
    Parameters
    - gridmap: Gridmap as numpy array / image of shape [IMAGE_SIZE, IMAGE_SIZE]
    - save_path: Absolute file path where the gridmap should be saved. E.g. "/tmp/gridmap123.png"
    - max_occluded_steps: How many steps after the first detected occupancy begins the occlusion. For lower resolution take lower steps (has to be fine-tuned)
    """
    global thread_number
    global calculated_frames
    num_cells_per_edge_ui = gridmap.shape[0]
    num_cells_per_edge_half_f = gridmap.shape[0] // 2 - 1

    occlusion_map = np.ones(gridmap.shape, dtype=np.float32)    # 0 - occluded, 1 - non occluded/visible
    start_time = time.time()
    
    # Angle array captures 0 to 360 degree in radians to simulate the lidar beams
    angle_array = np.arange(0,two_pi_f,angular_res_rad_f)
    # Radial array captures 0 to max distance of detection to iterate over the distance to the ego vehicle
    radial_array = np.arange(0, radial_limit_meter_f, radial_res_meter_f)
    # For performance: repeat both arrays up to the shape of the other one to do faster matrix operations
    angle_array = np.stack([angle_array]*radial_array.shape[0], axis=1)
    radial_array = np.stack([radial_array]*angle_array.shape[0], axis=0)

    # x,y grid contains all x,y-Coordinates which correlate to the given angle and radius
    xy_grid = np.empty((angle_array.shape[0], radial_array.shape[1], 2), dtype=int)  
    xy_grid[:,:,0] = grid_cell_size_inv_f * np.multiply(np.cos(angle_array), radial_array) + num_cells_per_edge_half_f # 0 - x
    xy_grid[:,:,1] = grid_cell_size_inv_f * np.multiply(np.sin(angle_array), radial_array) + num_cells_per_edge_half_f # 1 - y
    xy_grid = np.clip(xy_grid, 0, int(num_cells_per_edge_ui-1)) 
    
    occluded_steps = np.zeros((xy_grid.shape[0]), dtype=np.int32)
    is_occluded_array = np.zeros((xy_grid.shape[0]), dtype=np.bool)
    occlusion_wo_occup = np.ones((xy_grid.shape[0]), dtype=np.bool)
    position_array = np.zeros((xy_grid.shape[0], 2), dtype=int)

    for radial_index in range(xy_grid.shape[1]):
        x_i = xy_grid[:, radial_index, 0]
        y_i = xy_grid[:, radial_index, 1]

        # occluded_steps += np.multiply(np.ones(occluded_steps.shape, dtype=np.int32), is_occluded_array)
        # occluded_steps = np.multiply(is_occluded_array, )
        occ_f = gridmap[y_i, x_i]
        is_occupied = (occ_f < occ_thresh_f)
        is_changed = is_occupied * (1 - is_occluded_array)
        position_array[:,0] = position_array[:,0] * (1 - is_changed) + x_i * (is_changed)
        position_array[:,1] = position_array[:,1] * (1 - is_changed) + y_i * (is_changed)
        is_occluded_array = is_occluded_array + is_occupied 
        is_first_pixel = (np.absolute(position_array[:,0] - x_i) <= max_occluded_steps) * (np.absolute(position_array[:,1] - y_i) <= max_occluded_steps) * is_occupied
        # occlusion_wo_occup = (1 - is_occluded_array) + (is_occluded_array * occlusion_wo_occup * is_occupied)
        # occlusion_map[y_i, x_i] = occlusion_map[y_i, x_i] * (1 - (is_occluded_array * (1 - occlusion_wo_occup)))
        occlusion_map[y_i, x_i] = occlusion_map[y_i, x_i] * (1 - (is_occluded_array * (1 - is_first_pixel)))

        

    """
    # Version with for-loops for better understanding
    # TODO: Customizing this loops to new version
    ====================================================================================================
    for angle_index in xrange(xy_grid.shape[0]):
        occluded_steps = 0
        occluded = False
        for radial_index in xrange(xy_grid.shape[1]):
            x_i = xy_grid[angle_index, radial_index, 0]
            y_i = xy_grid[angle_index, radial_index, 1]
            visited_map[y_i, x_i] += 1
            if occluded:
                occluded_steps += 1
                if occluded_steps >= 7:
                    occlusion_map[y_i, x_i] = 0
            else:
                occ_f = gridmap[y_i, x_i]
                if(occ_f < occ_thresh_f):
                    occluded = True
    """
    scipy.misc.toimage(occlusion_map).save(save_path)
    thread_number -= 1
    calculated_frames += 1

def main(in_path, out_path, overwrite, image_size, max_occluded_steps):
    global thread_number
    global max_thread_number
    global calculated_frames
    # Directories where to load the occupancy images from and where to save the occlusions
    FILE_DIRECTORY = in_path
    SAVE_DIRECTORY = out_path
    # Image size of occupancy map. Occlusion images will be the same size
    IMAGE_SIZE = image_size
    OVERWRITE_FILES = overwrite

    file_list = sorted(glob(os.path.join(FILE_DIRECTORY,'*')), reverse=False)   # If process should start with last image set reverse=True
    if not os.path.exists(SAVE_DIRECTORY):
        os.makedirs(SAVE_DIRECTORY)

    file_number = 0
    calculated_frames = 0
    start_time = time.time()

    for frame_path in tqdm(file_list):
        file_number += 1
        # Prepare image and paths for calculation
        gridmap = imread(frame_path)
        frame_name = frame_path.split("/",-1)[-1]
        save_frame_name = frame_name.split(".",-1)[-1]+"_occlusion.pgm"
        save_path = os.path.join(SAVE_DIRECTORY, save_frame_name)

        # If occlusion map already exists => Skip it
        if OVERWRITE_FILES or not os.path.isfile(save_path):
            # If not maximum number of threads are started => start new one with function "createOcclusionMap"
            if thread_number < max_thread_number:
                #print("Start thread "+str(thread_number))
                thread_number += 1
                thread.start_new_thread(createOcclusionMap,(gridmap, save_path, max_occluded_steps, ))
            else:
                # Main thread also executes the function
                thread_number += 1
                createOcclusionMap(gridmap, save_path, max_occluded_steps)
                # Predict time for further calculation until all files are processed
                pred_calc_time = (len(file_list)-file_number)*(time.time()-start_time)*1.0/calculated_frames
                # Output to terminal
                """
				if file_number % 100 == 0:
                    if calculated_frames > 100:
                        CURSOR_UP_ONE = '\x1b[1A'
                        ERASE_LINE = '\x1b[2K'
                        print ""+(CURSOR_UP_ONE + ERASE_LINE) * 4 + CURSOR_UP_ONE
                    print "File "+str(file_number).zfill(5)+": "+frame_path
                    print "#"*100
                    print "Predicted calculation time: "+str(int(pred_calc_time / 3600.0))+"h "+str(int(pred_calc_time / 60.0) % 60)+"min "+str(int(pred_calc_time) % 60)+" sec"
                    print "#"*100
				"""
        else:
            if file_number % 100 == 1:
                # print "Skip "+frame_path+" (File number "+str(file_number)+")"
                start_time = time.time()

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--inpath", type=str, dest="in_path",
                            required=True, help="Path to directory of velocity images")
    parser.add_argument("--outpath", type=str, dest="out_path",
                            required=True, help="Path to directory where the masks should be saved")
    parser.add_argument("--overwrite", type=str2bool, dest="overwrite",
                            default=False, help="Path to directory where the masks should be saved")
    parser.add_argument("--imsize", type=int, dest="image_size",
                            default=128, help="Size of the images used in here")
    parser.add_argument("--occsteps", type=int, dest="max_occluded_steps",
                            default=1, help="How many steps/pixels after the first detected occupancy the occlusion should start")

    args = parser.parse_args()
    main(**vars(args))


    

