import numpy as np
import faiss
import torch
import transformers
import pandas as pd 
import uuid
import sys

def get_group(result_df: pd.DataFrame) -> str:
    return result_df['group'].mode().values[0]

def append_to_faiss_index(
        test_df, 
        faiss_index, 
        image, 
        name, 
        group, 
        processor,
        description=None, 
        object_id=None, 
        img_name=None):
    
    # for pandas
    object_id = object_id if object_id else str(uuid.uuid4())
    img_name = img_name if img_name else str(uuid.uuid4())
    
    row = pd.DataFrame({'object_id': [object_id], 'name': [name], 'group': [group],
                        'description': [description], 'img_name': [img_name]})
    
    test_df = pd.concat([test_df, row], ignore_index=True)
    
    # for faiss
    img_embedding = (
        model.get_image_features(**processor([image], return_tensors="pt"))[0]
        .detach()
        .cpu()
        .numpy()
    )
    img_embedding = np.array([img_embedding])
    faiss_index.add(img_embedding)
    
    
    return test_df, faiss_index
    


# test_dataframe.to_csv('...')
# faiss.write_index(index, "faiss.index")


def get_relevant_objects(images, index, test_dataframe, model:transformers.CLIPModel, processor, k=30) -> pd.DataFrame:
    device = "cpu"
    img_embeddings = [(
            model.get_image_features(**processor([image], return_tensors="pt").to(device))[0]
            .detach()
            .cpu()
            .numpy()
        ) for image in images]
    
    search_embed = np.mean(np.array([img_embeddings]), axis=1)

    distances, indexes = index.search(search_embed, k)
    
    return test_dataframe.iloc[indexes[0]]





