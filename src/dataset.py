import os
import torch
import torchvision.transforms as T
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader
from PIL import Image

# Pascal VOC Classes
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

class DistillationVOCDataset(VOCDetection):
    def __init__(self, root, year, image_set, transforms=None, teacher_transforms=None, fixed_size=(480, 480)):
        super().__init__(root, year=year, image_set=image_set, download=False)
        
        self.student_transforms = transforms
        self.teacher_transforms = teacher_transforms
        self.fixed_size = fixed_size # Target size (H, W)
        
        self.class_to_ind = dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))

    def __getitem__(self, index):
        img, target = super().__getitem__(index)

        w_orig, h_orig = img.size
        img_resized = img.resize(self.fixed_size)
        
        w_new, h_new = self.fixed_size
        w_scale = w_new / w_orig
        h_scale = h_new / h_orig

        # 5. Extract and Scale Bounding Boxes
        boxes = []
        labels = []
        objects = target['annotation']['object']
        if not isinstance(objects, list):
            objects = [objects]

        for obj in objects:
            bbox = obj['bndbox']
            # Scale the box coordinates
            xmin = float(bbox['xmin']) * w_scale
            ymin = float(bbox['ymin']) * h_scale
            xmax = float(bbox['xmax']) * w_scale
            ymax = float(bbox['ymax']) * h_scale
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_ind[obj['name']])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        target_dict = {"boxes": boxes, "labels": labels}

        student_img = img_resized
        if self.student_transforms:
            student_img = self.student_transforms(img_resized)

        # pass original image to teacher
        teacher_img = torch.zeros(1) 
        if self.teacher_transforms:
            teacher_img = self.teacher_transforms(img)

        return student_img, teacher_img, target_dict

def collate_fn(batch):
    student_imgs = []
    teacher_imgs = []
    targets = []

    for s_img, t_img, target in batch:
        student_imgs.append(s_img)
        teacher_imgs.append(t_img)
        targets.append(target)

    # Now this stack will work because all s_imgs are 480x480
    student_imgs = torch.stack(student_imgs, dim=0)
    
    if isinstance(teacher_imgs[0], torch.Tensor) and teacher_imgs[0].ndim > 1:
        try:
            teacher_imgs = torch.stack(teacher_imgs, dim=0)
        except:
            pass

    return student_imgs, teacher_imgs, targets

def get_loaders(batch_size=4):
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file_path))
    data_root = os.path.join(project_root, 'data')

    basic_transform = T.Compose([T.ToTensor()])

    train_ds_07 = DistillationVOCDataset(root=data_root, year='2007', image_set='trainval', transforms=basic_transform)
    train_ds_12 = DistillationVOCDataset(root=data_root, year='2012', image_set='trainval', transforms=basic_transform)
    val_ds = DistillationVOCDataset(root=data_root, year='2007', image_set='test', transforms=basic_transform)
    
    train_ds = torch.utils.data.ConcatDataset([train_ds_07, train_ds_12])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader
