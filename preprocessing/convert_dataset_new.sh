#!/bin/bash 
#
# @author Phillip Lippe
#
# Script for converting a recorded dataset into compressed numpy files for prediction learning. This includes (sequentially):
#
# -> generate_odometry_file.py  	| Generating gathered odometry file 
# -> cropAndResizeImage.py 			| Cropping and resizing original images into goal format
# -> generateOcclusionMap.py 		| Generating occlusion maps based on occupancy map
# -> createOccupancyMask.py 		| Mask occupancy maps with occlusion maps
# -> create_occlusion_images.py 	| Create RGB images combining
# -> extract_line_street.py 		| Splitting horizon maps into road (bool) and lanes (bool)
# -> calc_image_translation.py 		| Generate image translation matrizes for Spatial Transformer Module
# -> combineMasks.py 				| Combine occlusion and road mask into one to get only objects on the road
# -> compressMoveMapDataset.py 		| Compress dataset of (occupancy | occlusion | road | lanes | combined mask) into numpy arrays
# -> test_dataset.py 				| Check on corrupted compressed arrays (also checks the shape)
# -> find ... > list.txt			| Collect all absolute file paths into one txt-file 	
# 
#
# Parameters to change:
# -> PRESCALE - Usually the image is cropped first and afterwards scaled (default for PRESCALE < 0). 
#				If the image should be scaled first then set this variable to the needed size (see cropAndResizeImage.py).
# -> CROP_SIZE - Pixel size to which the original images should be cropped (see cropAndResizeImage.py).
# -> IMAGE_SIZE - Pixel size to which the cropped images should be scaled. This size is also used for further processing
# 				  (see cropAndResizeImage.py)
# -> OCCUP_STEPS - Number of pixels that should be visible in the occlusion behind the first occupied one (see generateOcclusionMap.py)
# -> SEQ_LENGTH - Number of frames that should be compressed in one array (see compressMoveMapDataset.py)
# -> STEP_SIZE - Save frequency of compressed arrays. 1 means save every sequence, 2 means save only every second and so on. 
# 				 Used to reduce the overlapping between sequences (see compressMoveMapDataset.py)
# -> OVERWRITE_OCCLUSION - Whether occlusion maps, if already saved, should be overwritten or skipped (see generateOcclusionMap.py)
# -> OVERWRITE_OCCUPMASK - Whether occupancy masks, if already saved, should be overwritten or skipped (see createOccupancyMask.py)
# -> OVERWRITE_OCCLIMGS - Whether combined occlusion images, if already saved, should be overwritten or skipped (see create_occlusion_images.py)
# -> BASEDIR - Base directory path where the datasets should be saved/stored
# -> SOURCEDIR - Direcory where all original images are stored (usually dumped data on shared memory)
# -> FRAMENUMBER - One timestamp number of a gridmap_..._grid_map_params.txt
# -> FRAME_COMPOSING - Factor with which the frame rate should be decreased. 2 means take only every second frame, so decreasing frame rate by 2.
# 					   Translations will be combined (see calc_image_translation.py) 
# -> DESTFOLDER - Local folder in BASEDIR where the data should be saved in (is created if not existing)
# -> OCCUPDIR, HORIZONDIR, OCCLDIR, OCCUPMASKDIR, ... - Name conventions for subfolders in DESTFOLDER. Can be changed if needed

# Parameters for original size 1232px, 0.13m cell to 96px, 0.475m
#----------------------------------------------------------------
# PRESCALE=337
# CROP_SIZE=96
# IMAGE_SIZE=96 
#
# ===============================================================
#
# Parameters for original size 456px, 0.1m cell to 96px, 0.475m
#----------------------------------------------------------------
# PRESCALE=-1
# CROP_SIZE=456
# IMAGE_SIZE=96 

PRESCALE=337
CROP_SIZE=96
IMAGE_SIZE=96
OCCUP_STEPS=1
SEQ_LENGTH=100
STEP_SIZE=5
FRAME_COMPOSING=2
OVERWRITE_OCCLUSION="False"
OVERWRITE_OCCUPMASK="False"
OVERWRITE_OCCLIMGS="False"
BASEDIR=./datasetNew/

# Short-cuts for common datasets. If other dataset should be converted, enter random name and 
# specify source directory and destination folder.
# FRAMENUMBER - deprecated!  
echo "Which dataset should be converted?"
read DATASET

if [ "$DATASET" == "ADTFFusion" ]
then
	echo "Setting up for ADTFFusion..."
	SOURCEDIR=/lhome/phlippe/data/ml_data/trajectory_learning/ADTFFusion20170622_104630/
	FRAMENUMBER="1498154823111590"
	DESTFOLDER=ADTFFusion
else
	if [ "$DATASET" == "Maude" ]
	then
		echo "Setting up for Maude..."
		SOURCEDIR=/lhome/phlippe/data/ml_data/trajectory_learning/20170517_around_maude_CNNDataWriter_dump_20170518/
		FRAMENUMBER="1495072737978553"
		DESTFOLDER=Maude
	else
		if [ "$DATASET" == "BigLoop" ]
		then
			echo "Setting up for BigLoop..."
			SOURCEDIR=/lhome/phlippe/data/ml_data/trajectory_learning/20170517_big_loop_CNNDataWriter_dump_20170518/
			FRAMENUMBER="1495073821008568"
			DESTFOLDER=BigLoop
		else
			if [ "$DATASET" == "MovingSeq_20170620" ]
			then
				echo "Setting up for MovingSeq_20170620..."
				SOURCEDIR=/lhome/phlippe/data/ml_data/trajectory_learning/20170620_151328/
				FRAMENUMBER="1497997988601285"
				DESTFOLDER=MovingSeq_20170620
			else
				if [ "$DATASET" == "MovingSeq_20170607034344" ]
				then
					SOURCEDIR=/lhome/phlippe/data/ml_data/trajectory_learning/20170607034344/
					FRAMENUMBER="1496876419267059"
					DESTFOLDER=MovingSeq_20170607034344
				else
					if [ "$DATASET" == "BigLoop2" ]
					then
						SOURCEDIR=/lhome/phlippe/data/ml_data/CNNDataWriter_dumps/cnn_data_writer_dump_20170808_164230_fusion_big_loop_cw_1_dense_traffic/
						FRAMENUMBER="1502236722111489"
						DESTFOLDER=BigLoop2
					else
						if [ "$DATASET" == "BigLoop3" ]
						then
							SOURCEDIR=/lhome/phlippe/data/ml_data/CNNDataWriter_dumps/cnn_data_writer_dump_20170809_102634_fusion_big_loop_ccw_medium_traffic_with_tl_dumped/
							FRAMENUMBER="1502300662063091"
							DESTFOLDER=BigLoop3
						else
							if [ "$DATASET" == "BigLoop4" ]
							then
								SOURCEDIR=/lhome/phlippe/data/ml_data/CNNDataWriter_dumps/cnn_data_writer_dump_20170809_104440_fusion_big_loop_ccw_medium_traffic_with_tl_and_stoplines_dumped/
								FRAMENUMBER="1502300689064129"
								DESTFOLDER=BigLoop4
							else	
								if [ "$DATASET" == "BigLoopNew" ]
								then
									SOURCEDIR=/mnt/ds3lab-scratch/lucala/20180117_1555_cw_big_loop_sunnyvale_ave_traffic_lights_annotation/
									FRAMENUMBER="12345"
									DESTFOLDER=BigLoopNew
								else
									echo "Dataset unknown..."
									echo "Please enter the source directory of your frames:"
									read SOURCEDIR
									#echo "Please enter one random frame number of your dataset:"
									#read FRAMENUMBER
									# Usage of FRAMENUMBER is deprecated
									FRAMENUMBER="12345"
									echo "Please enter the folder name in which the new data should be saved:"
									read DESTFOLDER
									echo "Got following entries:"
									echo "=> Source directory = $SOURCEDIR"
									echo "=> Frame number = $FRAMENUMBER"
									echo "=> Destination folder = $DESTFOLDER"
								fi
							fi
						fi
					fi
				fi
			fi
		fi
	fi
fi

# Specification of folder names depending on source and destination. Could be changed if needed.
DESTDIR="$BASEDIR$DESTFOLDER/"
OCCUPDIR="$DESTDIR""occup_""$IMAGE_SIZE""x""$IMAGE_SIZE"
HORIZONDIR="$DESTDIR""horizon_map_""$IMAGE_SIZE""x""$IMAGE_SIZE"
OCCLDIR="$DESTDIR""occlusion_""$IMAGE_SIZE""x""$IMAGE_SIZE"
OCCUPMASKDIR="$DESTDIR""occupmask_""$IMAGE_SIZE""x""$IMAGE_SIZE"
OCCLUDEDIMGDIR="$DESTDIR""occluded_imgs_""$IMAGE_SIZE""x""$IMAGE_SIZE"
EXTHORIZONDIR="$DESTDIR""extracted_horizon_""$IMAGE_SIZE""x""$IMAGE_SIZE"
TRANSDIR="$DESTDIR""translation_""$IMAGE_SIZE""x""$IMAGE_SIZE"
COMBMASKDIR="$DESTDIR""combined_mask_""$IMAGE_SIZE""x""$IMAGE_SIZE"
COMPRESSDIR="$DESTDIR""compressed_""$IMAGE_SIZE""x""$IMAGE_SIZE"
SPLITBASENAME="splitted_occup_""$IMAGE_SIZE""x""$IMAGE_SIZE"

# Serial execution of scripts that were explained at the top of this script.
echo "Create odometry file..."
python generate_odometry_file.py --folder $SOURCEDIR --save "$SOURCEDIR""odometry_t_mus-x_m-y_m-yaw_deg-yr_degs-v_ms.txt"
echo "Crop and resize original images..."
python cropAndResizeImage.py --inpath "$SOURCEDIR*_occupancy*" --outpath "$OCCUPDIR" --cropsize $CROP_SIZE --scalesize $IMAGE_SIZE --prescale $PRESCALE --suffix ".pgm"
python cropAndResizeImage.py --inpath "$SOURCEDIR*_horizon_map*" --outpath "$HORIZONDIR" --cropsize $CROP_SIZE --scalesize $IMAGE_SIZE --prescale $PRESCALE --suffix ".png"
echo "Generate occlusion maps..."
python generateOcclusionMap.py --inpath "$OCCUPDIR" --outpath "$OCCLDIR" --imsize $IMAGE_SIZE --occsteps $OCCUP_STEPS --overwrite $OVERWRITE_OCCLUSION
echo "Create occupancy masks..."
python createOccupancyMask.py --inpath "$OCCUPDIR" --outpath "$OCCUPMASKDIR" --occlpath "$OCCLDIR"  --overwrite $OVERWRITE_OCCUPMASK
echo "Create occlusion images..."
python create_occlusion_images.py --inpath "$OCCUPMASKDIR" --outpath "$OCCLUDEDIMGDIR" --occlpath "$OCCLDIR" --horizonpath "$HORIZONDIR"  --overwrite $OVERWRITE_OCCLIMGS
echo "Extract lanes and road map of horizon maps..."
python extract_line_street.py --inpath "$HORIZONDIR" --outpath "$EXTHORIZONDIR"
echo "Calculate image translations..."
python calc_image_translation.py --inpath "$OCCUPDIR" --outpath "$TRANSDIR" --odfile "$SOURCEDIR""odometry_t_mus-x_m-y_m-yaw_deg-yr_degs-v_ms.txt" --gridfile "$SOURCEDIR""gridmap_""$FRAMENUMBER""_grid_map_params.txt" --combFrames $FRAME_COMPOSING --overwrite "True"
echo "Combine masks..."
python combineMasks.py --mask1 "$EXTHORIZONDIR" --mask2 "$OCCLDIR" --outpath "$COMBMASKDIR" --mask1_suffix "_horizon_map_road.png" --mask2_suffix "_occupancy_occlusion.pgm" --out_suffix "_combined_mask.pgm" --overwrite $OVERWRITE_OCCLUSION

# Compression depends on whether the dataset should be split or not. 
if [ "$FRAME_COMPOSING" -eq 1 ]
then
	echo "Compress files..."
	python compressMoveMapDataset.py --occuppath "$OCCUPMASKDIR" --occlpath "$OCCLDIR" --roadpath "$EXTHORIZONDIR" --linespath "$EXTHORIZONDIR" --combpath "$COMBMASKDIR" --tfpath "$TRANSDIR" --outpath "$COMPRESSDIR" --seq "$SEQ_LENGTH" --step "$STEP_SIZE" --imsize "$IMAGE_SIZE" --combsuffix "_combined_mask.pgm" --tfsuffix "_translation.npz"
	echo "Test dataset..."
	python test_dataset.py --inpath "$COMPRESSDIR""/seq_???????.npz" --shape "$IMAGE_SIZE" "$IMAGE_SIZE" "$(( $SEQ_LENGTH * 5 ))" 
	echo "Create file list ""$DESTDIR""train_list_""$IMAGE_SIZE""x""$IMAGE_SIZE"".txt ..."
	find "$COMPRESSDIR" -name "seq_???????.npz" > "$DESTDIR""train_list_""$IMAGE_SIZE""x""$IMAGE_SIZE"".txt"
else 
	echo "Split dataset..."
	python split_frames.py --inpath "$OCCUPMASKDIR" --outpath "$DESTDIR" --basename $SPLITBASENAME --splits $FRAME_COMPOSING
	for i in `seq 0 $FRAME_COMPOSING`;
	do
		if [ "$i" -lt "$FRAME_COMPOSING" ]
		then
			echo "Compress files $i..."
			python compressMoveMapDataset.py --occuppath "$DESTDIR""$SPLITBASENAME""_""$i" --occlpath "$OCCLDIR" --roadpath "$EXTHORIZONDIR" --linespath "$EXTHORIZONDIR" --combpath "$COMBMASKDIR" --tfpath "$TRANSDIR" --outpath "$COMPRESSDIR" --seq "$SEQ_LENGTH" --step "$STEP_SIZE" --imsize "$IMAGE_SIZE" --combsuffix "_combined_mask.pgm" --tfsuffix "_translation.npz" --outsuffix "$i" --tfonly "False" --splitnumber $i --splitamount $FRAME_COMPOSING
		else 
			echo "Done"
		fi
	done
	echo "Test dataset..."
	python test_dataset.py --inpath "$COMPRESSDIR""/seq_???????#?.npz" --shape "$IMAGE_SIZE" "$IMAGE_SIZE" "$(( $SEQ_LENGTH * 5 ))" 
	echo "Create file list ""$DESTDIR""train_list_splitted_""$IMAGE_SIZE""x""$IMAGE_SIZE"".txt ..."
	find "$COMPRESSDIR" -name "seq_???????#?.npz" > "$DESTDIR""train_list_splitted_""$IMAGE_SIZE""x""$IMAGE_SIZE"".txt"
fi

echo "Create TFRecord..."
python convert_tfrecord.py --filename="$DESTFOLDER/""train_list_splitted_""$IMAGE_SIZE""x""$IMAGE_SIZE"".txt" --data_path="$BASEDIR" --dest_path="$DESTDIR"