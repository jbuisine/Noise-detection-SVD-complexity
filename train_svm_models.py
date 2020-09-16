# main imports
import numpy as np
import pandas as pd
import sys, os, argparse

# models imports
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

import joblib
import sklearn.svm as svm
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score

import models as mdl
import config


def main():

    parser = argparse.ArgumentParser(description="Train SKLearn model and save it into .joblib file for each cluster")

    parser.add_argument('--data', type=str, help='dataset folder with cluster files prefix')
    parser.add_argument('--choice', type=str, help='model choice from list of choices', choices=config.models_list)

    args = parser.parse_args()

    p_data   = args.data
    p_choice = args.choice

    ##########
    # Prepare output folders
    ##########
    current_output_models = os.path.join(config.output_models, os.path.split(p_data)[1])

    if not os.path.exists(current_output_models):
        os.makedirs(current_output_models)

    if not os.path.exists(config.output_results_folder):
        os.makedirs(config.output_results_folder)
    
    ##########
    # 0. find each cluster
    ##########

    clusters_datasets = sorted(os.listdir(p_data))
    clusters_datasets = [ e.split('.')[0] for e in clusters_datasets ]

    clusters_datasets = sorted(list(set(clusters_datasets)))

    print('Clusters to train are: {}'.format(clusters_datasets))

    for cluster_index, data_file in enumerate(clusters_datasets):

        print('---------------------------------------------------')
        print('Start training cluster n°{}'.format(cluster_index))
        print('---------------------------------------------------')
        ########################
        # 1. Get and prepare data
        ########################
        dataset_train = pd.read_csv(os.path.join(p_data, data_file + '.train'), header=None, sep=";")
        dataset_test = pd.read_csv(os.path.join(p_data, data_file + '.test'), header=None, sep=";")

        # default first shuffle of data
        dataset_train = shuffle(dataset_train)
        dataset_test = shuffle(dataset_test)

        # get dataset with equal number of classes occurences
        noisy_df_train = dataset_train[dataset_train.iloc[:, 0] == 1]
        not_noisy_df_train = dataset_train[dataset_train.iloc[:, 0] == 0]
        nb_noisy_train = len(noisy_df_train.index)
        nb_not_noisy_train = len(not_noisy_df_train.index)

        noisy_df_test = dataset_test[dataset_test.iloc[:, 0] == 1]
        not_noisy_df_test = dataset_test[dataset_test.iloc[:, 0] == 0]
        nb_noisy_test = len(noisy_df_test.index)
        nb_not_noisy_test = len(not_noisy_df_test.index)

        final_df_train = pd.concat([not_noisy_df_train, noisy_df_train])
        final_df_test = pd.concat([not_noisy_df_test, noisy_df_test])

        # shuffle data another time
        final_df_train = shuffle(final_df_train)
        final_df_test = shuffle(final_df_test)

        final_df_train_size = len(final_df_train.index)
        final_df_test_size = len(final_df_test.index)

        # use of the whole data set for training
        x_dataset_train = final_df_train.iloc[:,1:]
        x_dataset_test = final_df_test.iloc[:,1:]

        y_dataset_train = final_df_train.iloc[:,0]
        y_dataset_test = final_df_test.iloc[:,0]

        noisy_samples = nb_noisy_test + nb_noisy_train
        not_noisy_samples = nb_not_noisy_test + nb_not_noisy_train

        total_samples = noisy_samples + not_noisy_samples

        print('noisy:', noisy_samples)
        print('not_noisy:', not_noisy_samples)
        print('total:', total_samples)

        class_weight = {
            0: noisy_samples / float(total_samples),
            1: not_noisy_samples / float(total_samples)
        }

        print(class_weight)

        #######################
        # 2. Construction of the model : Ensemble model structure
        #######################

        print("---------------------------------------------------")
        print("Train dataset size:", final_df_train_size)
        model = mdl.get_trained_model(p_choice, x_dataset_train, y_dataset_train)

        ######################
        # 3. Test : dataset from .test dataset
        ######################

        y_train_model = model.predict(x_dataset_train)
        y_test_model = model.predict(x_dataset_test)
        print(y_test_model)

        train_accuracy = accuracy_score(y_dataset_train, y_train_model)
        test_accuracy = accuracy_score(y_dataset_test, y_test_model)

        train_f1 = f1_score(y_dataset_train, y_train_model)
        test_f1 = f1_score(y_dataset_test, y_test_model)

        train_auc = roc_auc_score(y_dataset_train, y_train_model)
        test_auc = roc_auc_score(y_dataset_test, y_test_model)

        ###################
        # 5. Output : Print and write all information in csv
        ###################

        print("Train dataset size:", final_df_train_size)
        print("Test dataset size:", final_df_test_size)

        print("Train ACC:", train_accuracy)
        print("Test ACC:", test_accuracy)
        print("Train F1:", train_f1)
        print("Test F1:", test_f1)
        print("Train AUC:", train_auc)
        print("Test AUC:", test_auc)

        ##################
        # 6. Save model : create path if not exists
        ##################

        joblib.dump(model, os.path.join(current_output_models, 'cluster_{}.joblib'.format(cluster_index)))

        ##################
        # 7. Save model : performance
        ##################
        results_filename = os.path.join(config.output_results_folder, os.path.split(p_data)[1])
        with open(results_filename, 'a') as f:
            f.write(
                '{};{};{};{};{};{};{};{}'
                .format(final_df_train_size, 
                    final_df_test_size, 
                    train_accuracy, 
                    test_accuracy, 
                    train_f1, 
                    test_f1, 
                    train_auc, 
                    test_auc)
                )

if __name__== "__main__":
    main()
