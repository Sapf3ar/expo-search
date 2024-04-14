import cv2
from imagededup.methods import PHash


class DuplicatesFinder():
    def __init__(self, image_dir: str):
        self.phasher = PHash()
        self.encodings = self.phasher.encode_images(image_dir=image_dir, recursive=True)

    def find_duplicates(self, current_image, topk_image_files: list[str], max_distance_threshold: int = 2) -> str|None:
        '''
        topk_image_ids: лист путей вида object_id/img_name.*, которые выводятся пользователю
        '''
        current_encodings = {}
        current_hash = self.phasher.encode_image(image_file=False, image_array=current_image)
        current_encodings["current_image"] = current_hash
        for image_path in topk_image_files:
            if image_path in self.encodings:
                current_encodings[image_path] = self.encodings[image_path]

        duplicates = self.phasher.find_duplicates(encoding_map=current_encodings, max_distance_threshold=max_distance_threshold)
        if len(duplicates["current_image"]):
            return duplicates["current_image"][0].split("/")[-2]
        return None


if "name" == "__main__":
    dp = DuplicatesFinder("data/train")

    image = cv2.imread("data/train/24378251/26830885.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    topk_image_files = ["24378251/26830886.jpg"]

    print(f"duplicate found!\nduplicate id: {dp.find_duplicates(image, topk_image_files, max_distance_threshold=15)}")
