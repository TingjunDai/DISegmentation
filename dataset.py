import os
import random
import numpy as np
from PIL import Image, ImageEnhance

import jittor as jt
from jittor.dataset import Dataset
from jittor import transform
from config import Config
from utils import path_to_image
from tqdm import tqdm

from image_proc import preproc

_class_labels_TR_sorted = 'Airplane, Ant, Antenna, Archery, Axe, BabyCarriage, Bag, BalanceBeam, Balcony, Balloon, Basket, BasketballHoop, Beatle, Bed, Bee, Bench, Bicycle, BicycleFrame, BicycleStand, Boat, Bonsai, BoomLift, Bridge, BunkBed, Butterfly, Button, Cable, CableLift, Cage, Camcorder, Cannon, Canoe, Car, CarParkDropArm, Carriage, Cart, Caterpillar, CeilingLamp, Centipede, Chair, Clip, Clock, Clothes, CoatHanger, Comb, ConcretePumpTruck, Crack, Crane, Cup, DentalChair, Desk, DeskChair, Diagram, DishRack, DoorHandle, Dragonfish, Dragonfly, Drum, Earphone, Easel, ElectricIron, Excavator, Eyeglasses, Fan, Fence, Fencing, FerrisWheel, FireExtinguisher, Fishing, Flag, FloorLamp, Forklift, GasStation, Gate, Gear, Goal, Golf, GymEquipment, Hammock, Handcart, Handcraft, Handrail, HangGlider, Harp, Harvester, Headset, Helicopter, Helmet, Hook, HorizontalBar, Hydrovalve, IroningTable, Jewelry, Key, KidsPlayground, Kitchenware, Kite, Knife, Ladder, LaundryRack, Lightning, Lobster, Locust, Machine, MachineGun, MagazineRack, Mantis, Medal, MemorialArchway, Microphone, Missile, MobileHolder, Monitor, Mosquito, Motorcycle, MovingTrolley, Mower, MusicPlayer, MusicStand, ObservationTower, Octopus, OilWell, OlympicLogo, OperatingTable, OutdoorFitnessEquipment, Parachute, Pavilion, Piano, Pipe, PlowHarrow, PoleVault, Punchbag, Rack, Racket, Rifle, Ring, Robot, RockClimbing, Rope, Sailboat, Satellite, Scaffold, Scale, Scissor, Scooter, Sculpture, Seadragon, Seahorse, Seal, SewingMachine, Ship, Shoe, ShoppingCart, ShoppingTrolley, Shower, Shrimp, Signboard, Skateboarding, Skeleton, Skiing, Spade, SpeedBoat, Spider, Spoon, Stair, Stand, Stationary, SteeringWheel, Stethoscope, Stool, Stove, StreetLamp, SweetStand, Swing, Sword, TV, Table, TableChair, TableLamp, TableTennis, Tank, Tapeline, Teapot, Telescope, Tent, TobaccoPipe, Toy, Tractor, TrafficLight, TrafficSign, Trampoline, TransmissionTower, Tree, Tricycle, TrimmerCover, Tripod, Trombone, Truck, Trumpet, Tuba, UAV, Umbrella, UnevenBars, UtilityPole, VacuumCleaner, Violin, Wakesurfing, Watch, WaterTower, WateringPot, Well, WellLid, Wheel, Wheelchair, WindTurbine, Windmill, WineGlass, WireWhisk, Yacht'
class_labels_TR_sorted = _class_labels_TR_sorted.split(', ')
config = Config()

class ISNetDataset(Dataset):

    def __init__(self, datasets, is_train=True):
        super().__init__()
        self.is_train = is_train
        self.load_all = config.load_all
        valid_extensions = ['.png', '.jpg', '.PNG', '.JPG', '.JPEG']
        
        self.img_transform = jt.transform.Compose([
            jt.transform.Resize(config.size),
            jt.transform.ToTensor(),
            jt.transform.ImageNormalize([0.5, 0.5, 0.5], [1, 1, 1])
        ])
        self.gt_transform = jt.transform.Compose([
            jt.transform.Resize(config.size),
            jt.transform.ToTensor()
        ])
        dataset_root = os.path.join(config.data_root_dir, config.task)
        # datasets can be a list of different datasets for training on combined sets.
        self.image_paths = []
        for dataset in datasets.split('+'):
            image_root = os.path.join(dataset_root, dataset, 'im')
            self.image_paths += [os.path.join(image_root, p) for p in os.listdir(image_root) if any(p.endswith(ext) for ext in valid_extensions)]
        self.label_paths = []
        for p in self.image_paths:
            for ext in valid_extensions:
                ## 'im' and 'gt' may need modifying
                p_gt = p.replace('/im/', '/gt/')[:-(len(p.split('.')[-1])+1)] + ext
                file_exists = False
                if os.path.exists(p_gt):
                    self.label_paths.append(p_gt)
                    file_exists = True
                    break
            if not file_exists:
                print('Not exists:', p_gt)
        
        if len(self.label_paths) != len(self.image_paths):
            set_image_paths = set([os.path.splitext(p.split(os.sep)[-1])[0] for p in self.image_paths])
            set_label_paths = set([os.path.splitext(p.split(os.sep)[-1])[0] for p in self.label_paths])
            print('Path diff:', set_image_paths - set_label_paths)
            raise ValueError(f"There are different numbers of images ({len(self.label_paths)}) and labels ({len(self.image_paths)})")
        
        if self.load_all:
            self.images_loaded, self.labels_loaded = [], []
            # for image_path, label_path in zip(self.image_paths, self.label_paths):
            for image_path, label_path in tqdm(zip(self.image_paths, self.label_paths), total=len(self.image_paths)):
                _image = path_to_image(image_path, size=config.size, color_type='rgb')
                _label = path_to_image(label_path, size=config.size, color_type='gray')
                self.images_loaded.append(_image)
                self.labels_loaded.append(_label)

    def __getitem__(self, index):
        if self.load_all:
            image = self.images_loaded[index]
            label = self.labels_loaded[index]
        else:
            image = path_to_image(self.image_paths[index], size=config.size, color_type='rgb')
            label = path_to_image(self.label_paths[index], size=config.size, color_type='gray')
        if self.is_train:
            image, label = preproc(image, label, preproc_methods=config.preproc_methods)

        image, label = self.img_transform(image), self.gt_transform(label)
        if self.is_train:
            return image, label
        else:
            return image, label, self.label_paths[index]

    def __len__(self):
        return len(self.image_paths)


class UDUNDataset(Dataset):

    def __init__(self, datasets, is_train=True):
        super().__init__()
        self.is_train = is_train
        self.load_all = config.load_all
        valid_extensions = ['.png', '.jpg', '.PNG', '.JPG', '.JPEG']
        
        self.img_transform = jt.transform.Compose([
            jt.transform.Resize(config.size),
            jt.transform.ImageNormalize([124.55 / 255.0, 118.90 / 255.0, 102.94 / 255.0], [56.77 / 255.0, 55.97 / 255.0, 57.50 / 255.0]),
            jt.transform.ToTensor()
        ])
        self.gt_transform = jt.transform.Compose([
            jt.transform.Resize(config.size),
            jt.transform.ToTensor()
        ])
        
        if is_train:
            self.trunk_transform = jt.transform.Compose([
                jt.transform.Resize(config.size),
                jt.transform.ToTensor()
            ])
            self.struct_transform = jt.transform.Compose([
                jt.transform.Resize(config.size),
                jt.transform.ToTensor()
            ])
            
        dataset_root = os.path.join(config.data_root_dir, config.task)
        # datasets can be a list of different datasets for training on combined sets.
        self.image_paths = []
        for dataset in datasets.split('+'):
            image_root = os.path.join(dataset_root, dataset, 'im')
            self.image_paths += [os.path.join(image_root, p) for p in os.listdir(image_root) if any(p.endswith(ext) for ext in valid_extensions)]
        self.label_paths = []
        if self.is_train:
            self.struct_paths = []
            self.trunk_paths = []
            for p in self.image_paths:
                for ext in valid_extensions:
                    p_st = p.replace('/im/', '/struct-origin/')[:-(len(p.split('.')[-1])+1)] + ext
                    file_exists = False
                    if os.path.exists(p_st):
                        self.struct_paths.append(p_st)
                        file_exists = True
                        break
                if not file_exists:
                    print('struct not exists:', p_st)
                for ext in valid_extensions:
                    p_tr = p.replace('/im/', '/trunk-origin/')[:-(len(p.split('.')[-1])+1)] + ext
                    file_exists = False
                    if os.path.exists(p_tr):
                        self.trunk_paths.append(p_tr)
                        file_exists = True
                        break
                if not file_exists:
                    print('trunk not exists:', p_tr)
                
        for p in self.image_paths:
            for ext in valid_extensions:
                p_gt = p.replace('/im/', '/gt/')[:-(len(p.split('.')[-1])+1)] + ext
                file_exists = False
                if os.path.exists(p_gt):
                    self.label_paths.append(p_gt)
                    file_exists = True
                    break
            if not file_exists:
                print('gt not exists:', p_gt)
        
        if (len(self.label_paths) != len(self.image_paths)):
            set_image_paths = set([os.path.splitext(p.split(os.sep)[-1])[0] for p in self.image_paths])
            set_label_paths = set([os.path.splitext(p.split(os.sep)[-1])[0] for p in self.label_paths])
            print('Path diff:', set_image_paths - set_label_paths)
            raise ValueError(f"There are different numbers of images ({len(self.label_paths)}) and labels ({len(self.image_paths)})")
        
        if self.is_train:
            if (len(self.struct_paths) != len(self.image_paths)) or (len(self.trunk_paths) != len(self.image_paths)):
                raise ValueError(f"There are different numbers of structs or trunks compared with images")
        
        if self.load_all:
            self.images_loaded, self.labels_loaded = [], []
            # for image_path, label_path in zip(self.image_paths, self.label_paths):
            for image_path, label_path in tqdm(zip(self.image_paths, self.label_paths), total=len(self.image_paths)):
                _image = path_to_image(image_path, size=config.size, color_type='rgb')
                _label = path_to_image(label_path, size=config.size, color_type='gray')
                self.images_loaded.append(_image)
                self.labels_loaded.append(_label)
            if self.is_train:
                self.struct_loaded, self.trunk_loaded = [], []
                for struct_path, trunk_path in tqdm(zip(self.struct_paths, self.trunk_paths), total=len(self.struct_paths)):
                    _struct = path_to_image(struct_path, size=config.size, color_type='gray')
                    _trunk = path_to_image(trunk_path, size=config.size, color_type='gray')
                    self.struct_loaded.append(_struct)
                    self.trunk_loaded.append(_trunk)
                

    def __getitem__(self, index):
        if self.load_all:
            image = self.images_loaded[index]
            label = self.labels_loaded[index]
            if self.is_train:
                struct = self.struct_loaded[index]
                trunk = self.trunk_loaded[index]
        else:
            image = path_to_image(self.image_paths[index], size=config.size, color_type='rgb')
            label = path_to_image(self.label_paths[index], size=config.size, color_type='gray')
            if self.is_train:
                struct = path_to_image(self.struct_paths[index], size=config.size, color_type='gray')
                trunk = path_to_image(self.trunk_paths[index], size=config.size, color_type='gray')
            
        if self.is_train:
            image, label, trunk, struct = preproc(image, label, trunk=trunk, struct=struct, preproc_methods=config.preproc_methods)
            trunk, struct = self.trunk_transform(trunk), self.struct_transform(struct)
        image, label = self.img_transform(image), self.gt_transform(label)
        
        if self.is_train:
            return image, label, trunk, struct
        else:
            return image, label, self.label_paths[index]

    def __len__(self):
        return len(self.image_paths)



class BiRefNetDataset(Dataset):

    def __init__(self, datasets, is_train=True):
        super().__init__()
        self.is_train = is_train
        self.load_all = config.load_all
        valid_extensions = ['.png', '.jpg', '.PNG', '.JPG', '.JPEG']
        
        if self.is_train and config.auxiliary_classification:
            self.cls_name2id = {_name: _id for _id, _name in enumerate(class_labels_TR_sorted)}
        self.img_transform = jt.transform.Compose([
            jt.transform.Resize(config.size),
            jt.transform.ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            jt.transform.ToTensor()
        ])
        self.gt_transform = jt.transform.Compose([
            jt.transform.Resize(config.size),
            jt.transform.ToTensor()
        ])
        dataset_root = os.path.join(config.data_root_dir, config.task)
        # datasets can be a list of different datasets for training on combined sets.
        self.image_paths = []
        for dataset in datasets.split('+'):
            image_root = os.path.join(dataset_root, dataset, 'im')
            self.image_paths += [os.path.join(image_root, p) for p in os.listdir(image_root) if any(p.endswith(ext) for ext in valid_extensions)]
        self.label_paths = []
        for p in self.image_paths:
            for ext in valid_extensions:
                ## 'im' and 'gt' may need modifying
                p_gt = p.replace('/im/', '/gt/')[:-(len(p.split('.')[-1])+1)] + ext
                file_exists = False
                if os.path.exists(p_gt):
                    self.label_paths.append(p_gt)
                    file_exists = True
                    break
            if not file_exists:
                print('Not exists:', p_gt)
        
        if len(self.label_paths) != len(self.image_paths):
            set_image_paths = set([os.path.splitext(p.split(os.sep)[-1])[0] for p in self.image_paths])
            set_label_paths = set([os.path.splitext(p.split(os.sep)[-1])[0] for p in self.label_paths])
            print('Path diff:', set_image_paths - set_label_paths)
            raise ValueError(f"There are different numbers of images ({len(self.label_paths)}) and labels ({len(self.image_paths)})")
        
        if self.load_all:
            self.images_loaded, self.labels_loaded = [], []
            self.class_labels_loaded = []
            # for image_path, label_path in zip(self.image_paths, self.label_paths):
            for image_path, label_path in tqdm(zip(self.image_paths, self.label_paths), total=len(self.image_paths)):
                _image = path_to_image(image_path, size=config.size, color_type='rgb')
                _label = path_to_image(label_path, size=config.size, color_type='gray')
                self.images_loaded.append(_image)
                self.labels_loaded.append(_label)
                self.class_labels_loaded.append(
                    self.cls_name2id[label_path.split('/')[-1].split('#')[3]] if self.is_train and config.auxiliary_classification else -1
                )

    def __getitem__(self, index):
        if self.load_all:
            image = self.images_loaded[index]
            label = self.labels_loaded[index]
            class_label = self.class_labels_loaded[index] if self.is_train and config.auxiliary_classification else -1
        else:
            image = path_to_image(self.image_paths[index], size=config.size, color_type='rgb')
            label = path_to_image(self.label_paths[index], size=config.size, color_type='gray')
            class_label = self.cls_name2id[self.label_paths[index].split('/')[-1].split('#')[3]] if self.is_train and config.auxiliary_classification else -1
        if self.is_train:
            image, label = preproc(image, label, preproc_methods=config.preproc_methods)

        image, label = self.img_transform(image), self.gt_transform(label)
        if self.is_train:
            return image, label, class_label
        else:
            return image, label, self.label_paths[index]

    def __len__(self):
        return len(self.image_paths)
    
    

class MVANetDataset(Dataset):

    def __init__(self, datasets, is_train=True):
        super().__init__()
        self.is_train = is_train
        self.load_all = config.load_all
        valid_extensions = ['.png', '.jpg', '.PNG', '.JPG', '.JPEG']
        
        self.img_transform = jt.transform.Compose([
            jt.transform.Resize(config.size),
            jt.transform.ToTensor(),
            jt.transform.ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.gt_transform = jt.transform.Compose([
            jt.transform.Resize(config.size),
            jt.transform.ToTensor()
        ])
        dataset_root = os.path.join(config.data_root_dir, config.task)
        # datasets can be a list of different datasets for training on combined sets.
        self.image_paths = []
        for dataset in datasets.split('+'):
            image_root = os.path.join(dataset_root, dataset, 'im')
            self.image_paths += [os.path.join(image_root, p) for p in os.listdir(image_root) if any(p.endswith(ext) for ext in valid_extensions)]
        self.label_paths = []
        for p in self.image_paths:
            for ext in valid_extensions:
                ## 'im' and 'gt' may need modifying
                p_gt = p.replace('/im/', '/gt/')[:-(len(p.split('.')[-1])+1)] + ext
                file_exists = False
                if os.path.exists(p_gt):
                    self.label_paths.append(p_gt)
                    file_exists = True
                    break
            if not file_exists:
                print('Not exists:', p_gt)
        
        if len(self.label_paths) != len(self.image_paths):
            set_image_paths = set([os.path.splitext(p.split(os.sep)[-1])[0] for p in self.image_paths])
            set_label_paths = set([os.path.splitext(p.split(os.sep)[-1])[0] for p in self.label_paths])
            print('Path diff:', set_image_paths - set_label_paths)
            raise ValueError(f"There are different numbers of images ({len(self.label_paths)}) and labels ({len(self.image_paths)})")
        
        if self.load_all:
            self.images_loaded, self.labels_loaded = [], []
            # for image_path, label_path in zip(self.image_paths, self.label_paths):
            for image_path, label_path in tqdm(zip(self.image_paths, self.label_paths), total=len(self.image_paths)):
                _image = path_to_image(image_path, size=config.size, color_type='rgb')
                _label = path_to_image(label_path, size=config.size, color_type='gray')
                self.images_loaded.append(_image)
                self.labels_loaded.append(_label)

    def __getitem__(self, index):
        if self.load_all:
            image = self.images_loaded[index]
            label = self.labels_loaded[index]
        else:
            image = path_to_image(self.image_paths[index], size=config.size, color_type='rgb')
            label = path_to_image(self.label_paths[index], size=config.size, color_type='gray')
        if self.is_train:
            image, label = preproc(image, label, preproc_methods=config.preproc_methods)

        image, label = self.img_transform(image), self.gt_transform(label)
        if self.is_train:
            return image, label
        else:
            return image, label, self.label_paths[index]

    def __len__(self):
        return len(self.image_paths)
    

def get_data_loader(datasets, batch_size, others=None, shuffle=True, is_train=True):
    if config.model == 'BiRefNet':
        dataset = BiRefNetDataset(datasets, is_train).set_attrs(batch_size=batch_size, shuffle=shuffle)
    elif config.model == 'ISNet' or config.model == 'ISNet_GTEncoder':
        dataset = ISNetDataset(datasets, is_train).set_attrs(batch_size=batch_size, shuffle=shuffle)
    elif config.model == 'UDUN':
        dataset = UDUNDataset(datasets, is_train).set_attrs(batch_size=batch_size, shuffle=shuffle)
    elif config.model == 'MVANet':
        dataset = MVANetDataset(datasets, is_train).set_attrs(batch_size=batch_size, shuffle=shuffle)
    return dataset