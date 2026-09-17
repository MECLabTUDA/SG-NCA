import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

class ShapeSceneDataset(Dataset):
    def __init__(self, device="cpu", num_samples=100, img_size=64):
        self.num_samples = num_samples
        self.img_size = img_size
        self.class_map = {'background': 0, 'circle': 1, 'triangle': 2, 'rectangle': 3}
        self.device = device

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Initialize canvas and mask (using uint8 for OpenCV)
        img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        nodes_list = []

        # 1. Circle (The Hub)
        c_center = (np.random.randint(15, 45), np.random.randint(15, 45))
        c_radius = np.random.randint(6, 10)
        c_color = [int(c) for c in np.random.randint(100, 255, 3)]
        cv2.circle(img, c_center, c_radius, c_color, -1)
        cv2.circle(mask, c_center, c_radius, self.class_map['circle'], -1)
        nodes_list.append([1, c_center[0], c_center[1]]) # [class_id, x, y]

        # 2. Triangle
        t_center = (np.random.randint(10, 54), np.random.randint(10, 25))
        t_size = np.random.randint(7, 12)
        t_pts = np.array([[t_center[0], t_center[1] - t_size], 
                        [t_center[0] - t_size, t_center[1] + t_size], 
                        [t_center[0] + t_size, t_center[1] + t_size]], np.int32)
        t_color = [int(c) for c in np.random.randint(100, 255, 3)]
        cv2.fillPoly(img, [t_pts], t_color)
        cv2.fillPoly(mask, [t_pts], self.class_map['triangle'])
        nodes_list.append([2, t_center[0], t_center[1]])

        # 3. Rectangle
        r_w, r_h = np.random.randint(10, 18), np.random.randint(10, 18)
        r_x, r_y = np.random.randint(5, 45), np.random.randint(40, 54)
        r_color = [int(c) for c in np.random.randint(100, 255, 3)]
        cv2.rectangle(img, (r_x, r_y), (r_x + r_w, r_y + r_h), r_color, -1)
        cv2.rectangle(mask, (r_x, r_y), (r_x + r_w, r_y + r_h), self.class_map['rectangle'], -1)
        nodes_list.append([3, r_x + r_w//2, r_y + r_h//2])

        # Define Tensors
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask).long()
        node_tensor = torch.tensor(nodes_list, dtype=torch.float32)
        
        # Edges: Circle(0) -> Triangle(1) and Circle(0) -> Rectangle(2)
        edges = torch.tensor([[0, 1], [0, 2]], dtype=torch.long)

        return img_tensor.to(self.device), mask_tensor.to(self.device), node_tensor.to(self.device), edges.to(self.device)
