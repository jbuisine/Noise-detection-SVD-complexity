import os, sys
import argparse
import numpy as np

# images imports
from PIL import Image
from ipfml.processing import segmentation
from ipfml import utils

# ml imports
import joblib

# project imports
from complexity.run.estimators import estimate, estimators_list
from features import get_features
import config

def write_progress(progress):
    barWidth = 180

    output_str = "["
    pos = barWidth * progress
    for i in range(barWidth):
        if i < pos:
           output_str = output_str + "="
        elif i == pos:
           output_str = output_str + ">"
        else:
            output_str = output_str + " "

    output_str = output_str + "] " + str(int(progress * 100.0)) + " %\r"
    print(output_str)
    sys.stdout.write("\033[F")

def main():

    parser = argparse.ArgumentParser(description="Compute 3 separates datasets based on scene complexity")

    parser.add_argument('--data', type=str, help='folder where scenes are available', required=True)
    parser.add_argument('--thresholds', type=str, help='file with scene list information and thresholds', required=True)
    parser.add_argument('--selected_zones', type=str, help='file which contains all selected zones of scene', required=True)  
    parser.add_argument('--feature', type=str, help='feature data choice', choices=config.features_choices_labels, required=True)
    parser.add_argument('--estimator', type=str, help='method to use to cluster zone of scene', choices=estimators_list, required=True)
    parser.add_argument('--model', type=str, help='kmeans model path', required=True)
    parser.add_argument('--mode', type=str, help='Kind of normalization level wished', choices=config.normalization_choices, required=True)
    parser.add_argument('--output', type=str, help="output prefix for generated datasets", required=True)

    args = parser.parse_args()

    p_data           = args.data
    p_thresholds     = args.thresholds
    p_selected_zones = args.selected_zones
    p_feature        = args.feature
    p_estimator      = args.estimator
    p_model          = args.model
    p_mode           = args.mode
    p_output         = args.output


     # 1. retrieve human_thresholds
    human_thresholds = {}

    # extract thresholds
    with open(p_thresholds) as f:
        thresholds_line = f.readlines()

        for line in thresholds_line:
            data = line.split(';')
            del data[-1] # remove unused last element `\n`
            current_scene = data[0]
            thresholds_scene = data[1:]

            # TODO : check if really necessary
            if current_scene != '50_shades_of_grey':
                human_thresholds[current_scene] = [ int(threshold) for threshold in  thresholds_scene ]

    # 2. get selected zones
    selected_zones = {}
    with(open(p_selected_zones, 'r')) as f:

        for line in f.readlines():

            data = line.split(';')
            del data[-1]
            scene_name = data[0]
            thresholds = data[1:]

            selected_zones[scene_name] = [ int(t) for t in thresholds ]

    # 3. first with load the kmeans model from scenes-complexity folder
    kmeans = joblib.load(p_model)

    # for each zone of scene, find the target label used from kmeans model
    scenes = os.listdir(p_data)

    output_dataset_folder = os.path.join(config.output_datasets, p_output)

    if not os.path.exists(output_dataset_folder):
        os.makedirs(output_dataset_folder)

    output_cluster_filename = os.path.join(output_dataset_folder, 'cluster_{}.{}')

    # 4. extract data
    for scene in sorted(scenes):

        if scene in selected_zones:
            print('\nBuilding data for scene: {}'.format(scene))
            scene_path = os.path.join(p_data, scene)

            images = sorted(os.listdir(scene_path))
            images_path = [ os.path.join(scene_path, img) for img in images if '.png' in img ]

            # get first image and for each zone associate label using kmeans
            img = Image.open(images_path[0])
            blocks = segmentation.divide_in_blocks(img, (200, 200))

            zones_label_cluster = []
            for b in blocks:
                data = estimate(p_estimator, b)
                label = kmeans.predict(data.reshape(1, -1))[0]
                zones_label_cluster.append(label)

            n_images = len(images_path)
            # now associate data with cluster per zone and write into new data file
            for img_index, image in enumerate(images_path):

                img_data = Image.open(image)
                blocks = segmentation.divide_in_blocks(img_data, (200, 200))

                samples_number = int(image.split('_')[-1].replace('.png', ''))

                for index, b in enumerate(blocks):
                    
                    # find classification label
                    threshold = human_thresholds[scene][index]

                    if samples_number > threshold:
                        input_label = 1
                    else:
                        input_label = 0

                    # compute data
                    input_data = get_features(p_feature, b)

                    if p_mode == 'svdn':
                        input_data = utils.normalize_arr_with_range(input_data)

                    # compute input line
                    line = str(input_label)

                    for value in input_data:
                        line += ';' + str(value)

                    line += '\n'

                    if index in selected_zones[scene]:
                        with open(output_cluster_filename.format(zones_label_cluster[index], 'train'), 'a') as f:
                            f.write(line)
                    else:
                        with open(output_cluster_filename.format(zones_label_cluster[index], 'test'), 'a') as f:
                            f.write(line)

                    write_progress((img_index + 1) / n_images)
            

            

if __name__ == "__main__":
    main()