import albumentations as A
import albumentations.pytorch
def train_transform(image,keypoints,height,width):
    transform = A.Compose([
        A.Resize(height,width,p=1),
        #A.Rotate(45,p=0.5),
        #A.RandomBrightnessContrast(p=0.2),
       # A.Flip(p=0.3),
        #A.Normalize(p=1.0),
        albumentations.pytorch.transforms.ToTensorV2()
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False, angle_in_degrees=True))
    transformed=transform(image=image,keypoints=keypoints)
    return transformed

def val_transform(image,keypoints,height,width):
    transform = A.Compose([
        A.augmentations.transforms.Resize(height, width, p=1),
        A.pytorch.transforms.ToTensorV2()
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False, angle_in_degrees=True))
    image_trans, keypoints_trans = transform(image, keypoints)
    return image_trans, keypoints_trans

def test_transform(image,keypoints,height,width):
    transform = A.Compose([
        A.augmentations.transforms.Resize(height, width, p=1),
        A.pytorch.transforms.ToTensorV2()
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False, angle_in_degrees=True))
    image_trans, keypoints_trans = transform(image, keypoints)
    return image_trans, keypoints_trans