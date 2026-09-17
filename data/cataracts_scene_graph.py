""""
This is an example dataloader code for using the Cataract Scene Graph Dataset with Pytorch Geometric to generate Dynamic Scene Graphs from the Scene Graph Annotations.
It is based on the pyg InMemoryDataset class. It should be pointed at a directory with a subdirectory called "raw" of .json annotation files.
When called for the first time it will process the data into a .pt file in the subdirectory "processed". When changing anything about the data or dataloader, reprocessing does not happen automatically.

felix.holm@tum.de
"""

import json
import os
import torch
from torch_geometric.data import Data, InMemoryDataset

class CATSG(InMemoryDataset):
    """
    PyTorch Geometric dataset for the CAT-SG (Cataract Surgery Scene Graph) dataset.
    
    This dataset creates dynamic scene graphs from cataract surgery video annotations,
    supporting tasks like surgical workflow recognition, scene graph generation, and
    surgical technique recognition.
    
    The dataset processes JSON annotation files containing frame-level annotations of
    surgical tools, anatomical structures, and their interactions over time.
    
    Args:
        root (str): Root directory where the dataset should be stored.
        target_fps (int): Base FPS to sample at for DSG generation
        window_size (int): Number of temporal frames to include in each dynamic scene graph.
        dilation (int): Dilation for temporal sampling within the target frame rate. (e.g. dilation 5 at 5 fps = dilation 1 at 1 fps)
        data_dir (str): Directory containing the raw annotation files.
        num_classes (int, optional): Number of surgical step classes. Defaults to 19.
        temporal_undirected (bool, optional): Whether temporal edges should be undirected.
            Defaults to True.
        num_edge_classes (int, optional): Number of edge relation classes. Defaults to 8.
        transform (callable, optional): A function/transform that takes in a
            torch_geometric.data.Data object and returns a transformed version.
        pre_transform (callable, optional): A function/transform that takes in a
            torch_geometric.data.Data object and returns a transformed version.
            The data object will be transformed before being saved to disk.
        pre_filter (callable, optional): A function that takes in a
            torch_geometric.data.Data object and returns a boolean value,
            indicating whether the data object should be included in the final dataset.
    
    Attributes:
        step_names (List[str]): List of 19 surgical step names from the CATARACTS dataset.
        object_names (List[str]): List of 29 surgical objects (tools and anatomical structures).
        edge_classes (List[str]): List of 8 relation types plus spatial "Close to" relation.
    """
    def __init__(self, root, target_fps,  window_size, data_dir, dilation=1, num_classes=19, temporal_undirected=True, num_edge_classes=8, transform=None, pre_transform=None, pre_filter=None):
        
        self.frame_spacing = 30 // target_fps
        """
        frame_spacing (int): Parameter that determines how many frames to skip when sampling for a desired frame rate. frame_spacing = 30//target_fps (CATARACTS videos are approx. 30 fps originally). 
        Since CAT-SG annotations are at 5 fps, the frame numbers that are annotated are multiples of 6 (6,12,18 etc.) and therefore frame_spacing must be divisible by 6.
        """
        assert self.frame_spacing % 6 == 0


        self.number_of_classes = num_classes
        self.window_size = window_size
        self.dilation = dilation
        self.temporal_undirected = temporal_undirected
        self.data_dir = data_dir
        self.num_edge_classes = num_edge_classes

        self.step_names = [
            "Idle",
            "Toric Marking",
            "Implant Ejection",
            "Incision",
            "Viscodilatation",
            "Capsulorhexis",
            "Hydrodissection",
            "Nucleus Breaking",
            "Phacoemulsification",
            "Vitrectomy",
            "Irrigation/Aspiration",
            "Preparing Implant",
            "Manual Aspiration",
            "Implantation",
            "Positioning",
            "OVD Aspiration",
            "Suturing",
            "Sealing Control",
            "Wound Hydratation",
        ]
        self.object_names = [
            "Pupil",                            #0
            "Surgical Tape",                    #1
            "Hand",                             #2
            "Eye Retractors",                   #3
            "Iris",                             #4
            "Skin",                             #5
            "Cornea",                           #6
            "Hydro. Cannula",                   #7
            "Visc. Cannula",                    #8
            "Cap. Cystotome",                   #9
            "Rycroft Cannula",                  #10
            "Bonn Forceps",                     #11
            "Primary Knife",                    #12
            "Ph. Handpiece",                    #13
            "Lens Injector",                    #14
            "I/A Handpiece",                    #15
            "Secondary Knife",                  #16
            "Micromanipulator",                 #17
            # "I/A Handpiece Handle",           #18
            "Cap. Forceps",                     #19
            # "R. Cannula Handle",              #20
            # "Ph. Handpiece Handle",           #21
            # "Cap. Cystotome Handle",          #22
            # "Sec. Knife Handle",              #23
            # "Lens Injector Handle",           #24
            "Suture Needle",                    #25
            "Needle Holder",                    #26
            "Charleux Cannula",                 #27
            # "Primary Knife Handle",           #28
            "Vitrectomy Handpiece",             #29
            "Mendez Ring",                      #30
            "Marker",                           #31
            # "Hydrodissection Cannula Handle"  #32
            "Troutman Forceps",                 #33
            "Cotton",                           #34
            "Iris Hooks",                       #35
            "Vannas Scissors",                  #???
        ]
        self.edge_classes = [
            "Close to", # represents the spatial or geometric relation between two objects
            "Holding",
            "Activation",
            "Pushing",
            "Pulling",
            "Cutting",
            "Inserting",
            "Retracting"
        ]

        super().__init__(
            root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return "data.pt"

    # def download(self):
    #     # Download to `self.raw_dir`.
    #     pass

    def process(self):
        """
        Processes raw annotation files into PyTorch Geometric Data objects.
        
        This method:
        1. Loads JSON annotation files from the raw directory
        2. Creates temporal scene graphs by aggregating frames within a sliding window
        3. Builds node features from object information (type, position, size, time)
        4. Constructs edge indices and attributes for geometric and semantic relations
        5. Adds temporal edges between same objects across different time steps
        6. Saves processed data as PyTorch tensors
        
        The resulting Data objects contain:
        - x: Node features [num_nodes, 5] (object_type, time, pos_x, pos_y, size)
        - edge_index: Edge connectivity [2, num_edges]
        - edge_attr: Edge attributes [num_edges, num_edge_classes+1] (one-hot encoded)
        - y: Surgical step label [1, num_classes] (one-hot encoded)
        - pos: Node positions [num_nodes, 2] (x, y coordinates)
        """
        data_list = []

        for file_ in self.raw_file_names:
            # Extract file components
            file_name = file_.split(".")[0]
            fold = file_name[0:-2]
            video_id = file_name[-2:]

            # Load annotation dictionary from JSON file
            with open(os.path.join(self.raw_dir, file_), "r") as f:
                data_dict = json.load(f)
            
            frames = list(data_dict.keys())
            frames.sort(key=lambda x: int(x))

            for i, frame in enumerate(frames):
                # Get surgical step label and convert to one-hot encoding
                if data_dict[frame]["step"] == "Hydrodissetion":
                    data_dict[frame]["step"] = "Hydrodissection"  # fix typo in annotation files
                try:
                    y = self.step_names.index(data_dict[frame]["step"])
                except ValueError as e:
                    print(f"Unknown step in file {file_name} fold {fold} video {video_id} frame {frame}: {data_dict[frame]['step']}")
                    raise e
                y = [0 if j != y else 1 for j in range(self.number_of_classes)]
                y = torch.tensor(y, dtype=torch.float).unsqueeze(0)

                # Initialize containers for temporal aggregation
                list_of_x = []
                list_of_edge_indices = []
                list_of_entity_dicts = []
                list_of_edge_attrs = []
                list_of_pos = []

                shift = 0  # Node index shift for temporal concatenation

                # Build dynamic scene graph over temporal window
                for w_idx in range(self.window_size):
                    index_step = w_idx * self.dilation * (self.frame_spacing // 6)
                    if i - index_step < 0:
                        continue
                    f_idx = frames[i - index_step]

                    # Extract entities and relations for current frame
                    entities = [self.object_names.index(obj) for obj in data_dict[f_idx]["entities"]]
                    geometric_relations = [
                        [self.object_names.index(obj[0]), self.object_names.index(obj[1])]
                        for obj in data_dict[f_idx]["geometric_relations"]
                    ]

                    if not entities or not geometric_relations:
                        print(f"Skipping empty graph: {file_name} {fold} {video_id} {f_idx}")
                        continue

                    semantic_relations = [
                        [self.object_names.index(obj[0]), self.object_names.index(obj[2])]
                        for obj in data_dict[f_idx]["semantic_relations"]
                    ]
                    
                    relations = geometric_relations + semantic_relations
                    
                    # Create edge attributes with one-hot encoding
                    edge_attr = torch.zeros((len(relations), len(self.edge_classes) + 1))
                    edge_attr[:len(geometric_relations), 0] = 1  # Geometric relations are type 0
                    
                    # Encode semantic relations
                    for j, rel in enumerate(data_dict[f_idx]["semantic_relations"]):
                        edge_attr[len(geometric_relations) + j, self.edge_classes.index(rel[1])] = 1

                    # Extract spatial information
                    pos = data_dict[f_idx]["pos"]
                    size = data_dict[f_idx]["size"]

                    # Create node features [object_type, time, pos_x, pos_y, size]
                    entities_tensor = torch.tensor([[x] for x in entities], dtype=torch.float)
                    pos_x = torch.tensor([[p[0]] for p in pos], dtype=torch.float)
                    pos_y = torch.tensor([[p[1]] for p in pos], dtype=torch.float)
                    t = torch.tensor([[w_idx]] * len(entities), dtype=torch.long)
                    s = torch.tensor([[s] for s in size], dtype=torch.float)

                    x = torch.cat((entities_tensor, t, pos_x, pos_y, s), dim=1)
                    pos = torch.tensor(pos, dtype=torch.float)
                    
                    # Create entity mapping for edge construction
                    entity_dict = {v: i + shift for i, v in enumerate(entities)}
                    edge_index = torch.tensor([
                        [entity_dict[relation[0]], entity_dict[relation[1]]]
                        for relation in relations
                    ], dtype=torch.long).t().contiguous()

                    # Add temporal edges between consecutive frames
                    if list_of_x:
                        prev_entity_dict = list_of_entity_dicts[-1]
                        temp_edge_index = torch.empty((2, 0), dtype=torch.long)
                        
                        for ent in entities:
                            if ent in prev_entity_dict:
                                # Forward temporal edge
                                temp_edge = torch.tensor([
                                    [entity_dict[ent], prev_entity_dict[ent]]
                                ], dtype=torch.long).t()
                                temp_edge_index = torch.cat((temp_edge_index, temp_edge), dim=1)
                                
                                # Backward temporal edge (if undirected)
                                if self.temporal_undirected:
                                    temp_edge_back = torch.tensor([
                                        [prev_entity_dict[ent], entity_dict[ent]]
                                    ], dtype=torch.long).t()
                                    temp_edge_index = torch.cat((temp_edge_index, temp_edge_back), dim=1)
                        
                        if temp_edge_index.shape[1] > 0:
                            list_of_edge_indices.append(temp_edge_index)
                            # Create temporal edge attributes
                            num_temp_edges = temp_edge_index.shape[1]
                            temp_edge_attrs = torch.zeros((num_temp_edges, self.num_edge_classes + 1))
                            temp_edge_attrs[:, -1] = 1  # Set last bit for temporal edges
                            list_of_edge_attrs.append(temp_edge_attrs)

                    # Store frame data
                    list_of_x.append(x)
                    list_of_edge_indices.append(edge_index)
                    list_of_entity_dicts.append(entity_dict)
                    list_of_edge_attrs.append(edge_attr)
                    list_of_pos.append(pos)
                    shift += len(entities)

                # Skip if no valid frames in window
                if not list_of_x:
                    print(f"Skipping empty temporal graph: {file_name} {fold} {video_id} {frame}")
                    continue

                # Concatenate all temporal data
                x = torch.cat(list_of_x, dim=0)
                edge_index = torch.cat(list_of_edge_indices, dim=1)
                edge_attr = torch.cat(list_of_edge_attrs, dim=0)
                pos = torch.cat(list_of_pos, dim=0)

                # Create PyTorch Geometric Data object
                data = Data(
                    x=x,
                    y=y,
                    edge_index=edge_index,
                    pos=pos,
                    edge_attr=edge_attr
                )
                data_list.append(data)

        # Apply filters and transforms
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Save processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])