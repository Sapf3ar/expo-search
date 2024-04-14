
import numpy as np
def get_relevant_objects(images, index, test_dataframe, model, processor, k=30):
    img_embeddings = [(
        model.get_image_features(**processor([image], return_tensors="pt").to("cuda"))[0]
        .detach()
        .cpu()
        .numpy()
    ) for image in images]
    
    search_embed = np.mean(np.array([img_embeddings]), axis=1)

    distances, indexes = index.search(search_embed, k)
    
    return test_dataframe.iloc[indexes[0]]




get_relevant_objects(
    images, 
    faiss_index, 
    test_dataframe, 
    model=model, 
    processor=processor
)

