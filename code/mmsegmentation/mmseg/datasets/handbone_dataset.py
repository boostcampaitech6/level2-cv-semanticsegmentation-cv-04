import os

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
      
      
@DATASETS.register_module()
class XRayDataset(BaseSegDataset):
    
    METAINFO = dict(
        classes=('finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
            'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
            'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
            'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
            'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
            'Triquetrum', 'Pisiform', 'Radius', 'Ulna'
        ),
        palette=[
            (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
            (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
            (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
            (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
            (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
            (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176)
        ]
    )
    
    def __init__(self, split, **kwargs):
        self.split = split
        
        super().__init__(**kwargs)
    
    def load_data_list(self):
        
        pngs = {
            os.path.relpath(os.path.join(root, fname), start=self.data_root + f"{self.split}/DCM/")
            for root, _dirs, files in os.walk(self.data_root + f"{self.split}/DCM/")
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".png"
        }
        
        jsons = {
            os.path.relpath(os.path.join(root, fname), start=self.data_root + f"{self.split}/outputs_json/")
            for root, _dirs, files in os.walk(self.data_root + f"{self.split}/outputs_json/")
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".json"
        }
        
        pngs = sorted(pngs)
        jsons = sorted(jsons)
        
        data_list = []
        for i, (img_path, ann_path) in enumerate(zip(pngs, jsons)):
            data_info = dict(
                img_path=os.path.join(self.data_root + f"{self.split}/DCM/", img_path),
                seg_map_path=os.path.join(self.data_root + f"{self.split}/outputs_json/", ann_path),
            )
            data_list.append(data_info)
            
        return data_list