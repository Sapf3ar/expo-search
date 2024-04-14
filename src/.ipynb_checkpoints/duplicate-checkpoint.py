import os
import torch
import numpy as np
import pandas as pd

from cross_encoder_model import CEModel

from PIL import Image
from torch.utils.data import Dataset


class DuplicateSecurity():
    def __init__(self,
                 searching_object_embedding: np.ndarray,
                 topk_objects_embeddings: dict[str, np.ndarray],
                 cross_encoder_model_weights_path: str
                ) -> None :
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.searching_object_embedding = searching_object_embedding
        self.topk_objects_embeddings = topk_objects_embeddings

        # self.model = CEModel().load_state_dict(torch.load(topk_objects_embeddings)).to(self.device)

    def find_duplicate(self) -> str:
        logits = {}
        for object_id,embedding in self.topk_objects_embeddings.items():
            logit = self.model.forward(self.searching_object_embedding, embedding)
            logits[object_id] = logit
        return


def generate_pairs(data, mode):
    positive_data = data[data.object_id.apply(lambda x: len(data[data.object_id == x]) > 1)]
    negative_data = data[data.object_id.apply(lambda x: len(data[data.object_id == x]) == 1)]
    
    first_image = []
    second_image = []
    groups = []
    for object_id in positive_data.object_id:
        id_data = positive_data[positive_data.object_id == object_id]
        for idx, (object_id,img_name,group) in enumerate(zip(id_data.object_id,id_data.img_name,id_data.group)):
            first_path = os.path.join(str(object_id),img_name)
            for object_id_,img_name_ in zip(id_data[idx+1:].object_id,id_data[idx+1:].img_name):
                second_path = os.path.join(str(object_id_),img_name_)
                first_image.append(first_path)
                second_image.append(second_path)
                groups.append(group)
                first_image.append(second_path)
                second_image.append(first_path)
                groups.append(group)
    
    new_data = []
    for f,s,g in zip(first_image,second_image,groups):
        new_data.append([f,s,g,1])

    first_image = []
    second_image = []
    groups = []
    for object_id,positive_group in zip(positive_data.object_id,positive_data.group):
        group_data = negative_data[negative_data.group == positive_group]
        for idx, (object_id,img_name,group) in enumerate(zip(id_data.object_id,id_data.img_name,id_data.group)):
            first_path = os.path.join(str(object_id),img_name)
            if mode == "train":
                k = 7
            else:
                k = 3
            for i in range(k):
                random_row = group_data.sample()
                second_path = os.path.join(str(random_row.object_id.item()),random_row.img_name.item())
                first_image.append(first_path)
                second_image.append(second_path)
                groups.append(group)
                first_image.append(second_path)
                second_image.append(first_path)
                groups.append(group)
    
    for f,s,g in zip(first_image,second_image,groups):
        new_data.append([f,s,g,0])
    
    new_data_df = pd.DataFrame(new_data, columns=["first_path", "second_path", "group", "label"])
    return new_data_df


class DuplicateDataset(Dataset):
    def __init__(self, csv_path, npy_path, mode, fold):
        assert mode in ["train", "eval"]
        self.npy_path = npy_path
        data = pd.read_csv(csv_path)

        if mode == "train":
            data = data[data.fold != fold]
            self.data = generate_pairs(data, mode)
        elif mode == 'eval':
            data = data[(data.fold == fold)]
            self.data = generate_pairs(data, mode)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        first_embedding_path = os.path.join(self.npy_path,row["first_path"].split(".")[0]+".npy")
        second_embedding_path = os.path.join(self.npy_path,row["second_path"].split(".")[0]+".npy")
        label = [0, 0]
        label[row["label"]] = 1

        sample = {}
        sample["first"] = torch.from_numpy(np.load(first_embedding_path))
        sample["second"] = torch.from_numpy(np.load(second_embedding_path))
        sample["labels"] = torch.Tensor(label)
        return sample



