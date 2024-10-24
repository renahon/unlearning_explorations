import json
import os
from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset
import torch
import random
from PIL import Image


class FEMNIST(Dataset):
    """
    This dataset is derived from the Leaf repository
    (https://github.com/TalwalkarLab/leaf) pre-processing of the Extended MNIST
    dataset, grouping examples by writer. Details about Leaf were published in
    "LEAF: A Benchmark for Federated Settings" https://arxiv.org/abs/1812.01097.
    """

    def __init__(self, train=True, transform=None, target_transform=None, ):
        super(FEMNIST, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        train_clients, train_groups, train_data_temp, test_data_temp = read_data("Classification/data/femnist/train",
                                                                                 "Classification/data/femnist/test")
        if self.train:
            self.dic_users = {}
            train_data_x = []
            train_data_y = []
            for i in range(len(train_clients)):
                # if i == 100:
                #     break
                self.dic_users[i] = set()
                l = len(train_data_x)
                cur_x = train_data_temp[train_clients[i]]['x']
                cur_y = train_data_temp[train_clients[i]]['y']
                for j in range(len(cur_x)):
                    self.dic_users[i].add(j + l)
                    train_data_x.append(np.array(cur_x[j]).reshape(28, 28))
                    train_data_y.append(cur_y[j])
            self.data = train_data_x
            self.labels = train_data_y
        else:
            test_data_x = []
            test_data_y = []
            for i in range(len(train_clients)):
                cur_x = test_data_temp[train_clients[i]]['x']
                cur_y = test_data_temp[train_clients[i]]['y']
                for j in range(len(cur_x)):
                    test_data_x.append(np.array(cur_x[j]).reshape(28, 28))
                    test_data_y.append(cur_y[j])
            self.data = test_data_x
            self.labels = test_data_y

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)

    def get_client_dic(self):
        if self.train:
            return self.dic_users
        else:
            exit("The test dataset do not have dic_users!")
            
def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data



def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data


def femnist_replace(dataset, seed, percentage_limit_1=0.1, percentage_limit_2=0.02):
    dataset_dict = dataset.dic_users
    total_samples = len(dataset)
    
    # Calculer le nombre maximal d'échantillons pour chaque limite
    max_samples_1 = int(total_samples * percentage_limit_1)
    max_samples_2 = int(total_samples * percentage_limit_2)
    
    selected_users_stage_1 = []
    selected_users_stage_2 = []
    selected_samples_count_1 = 0
    selected_samples_count_2 = 0
    rng = np.random.RandomState(seed)
    
    # Mélanger les utilisateurs aléatoirement
    users = list(dataset_dict.keys())
    rng.shuffle(users)

    selected_indices_stage_1 = []
    selected_indices_stage_2 = []

    # Étape 1 : Sélection de 0.1% des données
    for user in users:
        user_samples_indices = dataset_dict[user]
        num_user_samples = len(user_samples_indices)
        
        # Si ajouter cet utilisateur dépasse la limite de 0.1%, on l'ignore
        if selected_samples_count_1 + num_user_samples < max_samples_1:
            selected_users_stage_1.append(user)
            selected_samples_count_1 += num_user_samples
            
            # Mélanger les indices des échantillons de l'utilisateur
            rng.shuffle(user_samples_indices)
            
            selected_indices_stage_1.append(user_samples_indices)
    
    # Combiner les indices de tous les utilisateurs pour la première étape
    selected_indices_stage_1 = np.hstack(selected_indices_stage_1).astype(int)
    rng.shuffle(users)
    # Étape 2 : Sélection de 0.02% des données à partir des utilisateurs sélectionnés
    for user in users:
        if user not in selected_users_stage_1:
            user_samples_indices = dataset_dict[user]
            num_user_samples = len(user_samples_indices)
            
            # Si ajouter cet utilisateur dépasse la limite de 0.02%, on l'ignore
            if selected_samples_count_2 + num_user_samples < max_samples_2:
                selected_users_stage_2.append(user)
                selected_samples_count_2 += num_user_samples
                
                # Mélanger les indices des échantillons de l'utilisateur
                rng.shuffle(user_samples_indices)
                
                selected_indices_stage_2.append(user_samples_indices)

    # Combiner les indices de tous les utilisateurs pour la deuxième étape
    selected_indices_stage_2 = np.hstack(selected_indices_stage_2).astype(int)

    print(f"Étape 1 - Nombre total de samples sélectionnés : {selected_samples_count_1}/{total_samples}")
    print(f"Étape 1 - Nombre d'utilisateurs sélectionnés : {len(selected_users_stage_1)}")
    print(f"Étape 2 - Nombre total de samples sélectionnés : {selected_samples_count_2}/{total_samples}")
    print(f"Étape 2 - Nombre d'utilisateurs sélectionnés : {len(selected_users_stage_2)}")

    return selected_indices_stage_1, selected_indices_stage_2

