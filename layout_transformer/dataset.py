import numpy as np
import torch
from torchvision.datasets.mnist import MNIST
from torch.utils.data.dataset import Dataset
from PIL import Image, ImageDraw, ImageOps
import json

from tqdm import tqdm
from pathlib import Path
from typing import List, Optional, Tuple,Dict
from numpy import ndarray

from utils import trim_tokens, gen_colors


class Padding(object):
    def __init__(self, max_length, vocab_size):
        self.max_length = max_length
        self.bos_token = vocab_size - 3
        self.eos_token = vocab_size - 2
        self.pad_token = vocab_size - 1

    def __call__(self, layout):
        # grab a chunk of (max_length + 1) from the layout

        chunk = torch.zeros(self.max_length+1, dtype=torch.long) + self.pad_token
        # Assume len(item) will always be <= self.max_length:
        chunk[0] = self.bos_token
        chunk[1:len(layout)+1] = layout
        chunk[len(layout)+1] = self.eos_token

        x = chunk[:-1]
        y = chunk[1:]
        return {'x': x, 'y': y}


class MNISTLayout(MNIST):

    def __init__(self, root, train=True, download=True, threshold=32, max_length=None):
        super().__init__(root, train=train, download=download)
        self.vocab_size = 784 + 3  # bos, eos, pad tokens
        self.bos_token = self.vocab_size - 3
        self.eos_token = self.vocab_size - 2
        self.pad_token = self.vocab_size - 1

        self.threshold = threshold
        self.data = [self.img_to_set(img) for img in self.data]
        self.max_length = max_length
        if self.max_length is None:
            self.max_length = max([len(x) for x in self.data]) + 2  # bos, eos tokens
        self.transform = Padding(self.max_length, self.vocab_size)

    def __len__(self):
        return len(self.data)

    def img_to_set(self, img):
        fg_mask = img >= self.threshold
        fg_idx = fg_mask.nonzero(as_tuple=False)
        fg_idx = fg_idx[:, 0] * 28 + fg_idx[:, 1]
        return fg_idx

    def render(self, layout):
        layout = trim_tokens(layout, self.bos_token, self.eos_token, self.pad_token)
        x_coords = layout % 28
        y_coords = layout // 28
        # valid_idx = torch.where((y_coords < 28) & (y_coords >= 0))[0]
        img = np.zeros((28, 28, 3)).astype(np.uint8)
        img[y_coords, x_coords] = 255
        return Image.fromarray(img, 'RGB')

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) tokens from the data
        layout = self.transform(self.data[idx])
        return layout['x'], layout['y']


class JSONLayout(Dataset):
    def __init__(self, json_path, max_length=None, precision=8):
        with open(json_path, "r") as f:
            data = json.loads(f.read())

        images, annotations, categories = data['images'], data['annotations'], data['categories']
        self.size = pow(2, precision)

        self.categories = {c["id"]: c for c in categories}
        self.colors = gen_colors(len(self.categories))

        self.json_category_id_to_contiguous_id = {
            v: i + self.size for i, v in enumerate([c["id"] for c in self.categories.values()])
        }

        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

        self.vocab_size = self.size + len(self.categories) + 3  # bos, eos, pad tokens
        self.bos_token = self.vocab_size - 3
        self.eos_token = self.vocab_size - 2
        self.pad_token = self.vocab_size - 1

        image_to_annotations = {}
        for annotation in annotations:
            image_id = annotation["image_id"]

            if not (image_id in image_to_annotations):
                image_to_annotations[image_id] = []

            image_to_annotations[image_id].append(annotation)

        self.data = []
        for image in images:
            image_id = image["id"]
            height, width = float(image["height"]), float(image["width"])

            if image_id not in image_to_annotations:
                continue

            ann_box = []
            ann_cat = []
            for ann in image_to_annotations[image_id]:
                x, y, w, h = ann["bbox"]
                ann_box.append([x, y, w, h])
                ann_cat.append(self.json_category_id_to_contiguous_id[ann["category_id"]])

            # Sort boxes
            ann_box = np.array(ann_box)
            ind = np.lexsort((ann_box[:, 0], ann_box[:, 1]))
            ann_box = ann_box[ind]
       
            ann_cat = np.array(ann_cat)
            ann_cat = ann_cat[ind]

            # Discretize boxes
            ann_box = self.quantize_box(ann_box, width, height)

            # Append the categories
            layout = np.concatenate([ann_cat.reshape(-1, 1), ann_box], axis=1)

            # Flatten and add to the dataset
            self.data.append(layout.reshape(-1))

        self.max_length = max_length
        if self.max_length is None:
            self.max_length = max([len(x) for x in self.data]) + 2  # bos, eos tokens
        self.transform = Padding(self.max_length, self.vocab_size)

    def quantize_box(self, boxes, width, height):

        # range of xy is [0, large_side-1]
        # range of wh is [1, large_side]
        # bring xywh to [0, 1]
        boxes[:, [2, 3]] = boxes[:, [2, 3]] - 1
        boxes[:, [0, 2]] = boxes[:, [0, 2]] / (width - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]] / (height - 1)
        boxes = np.clip(boxes, 0, 1)

        # next take xywh to [0, size-1]
        boxes = (boxes * (self.size - 1)).round()

        return boxes.astype(np.int32)

    def __len__(self):
        return len(self.data)

    def render(self, layout):
        img = Image.new('RGB', (256, 256), color=(255, 255, 255))
        draw = ImageDraw.Draw(img, 'RGBA')
        layout = layout.reshape(-1)
        layout = trim_tokens(layout, self.bos_token, self.eos_token, self.pad_token)
        layout = layout[: len(layout) // 5 * 5].reshape(-1, 5)
        box = layout[:, 1:].astype(np.float32)
        box[:, [0, 1]] = box[:, [0, 1]] / (self.size - 1) * 255
        box[:, [2, 3]] = box[:, [2, 3]] / self.size * 256
        box[:, [2, 3]] = box[:, [0, 1]] + box[:, [2, 3]]

        for i in range(len(layout)):
            x1, y1, x2, y2 = box[i]
            cat = layout[i][0]
            col = self.colors[cat-self.size] if 0 <= cat-self.size < len(self.colors) else [0, 0, 0]
            draw.rectangle([x1, y1, x2, y2],
                           outline=tuple(col) + (200,),
                           fill=tuple(col) + (64,),
                           width=2)

        # Add border around image
        img = ImageOps.expand(img, border=2)
        return img

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) tokens from the data
        layout = torch.tensor(self.data[idx], dtype=torch.long)
        layout = self.transform(layout)
        return layout['x'], layout['y']


class PPTLayout(Dataset):
    def __init__(self,datapath:Path,max_length:int=162,precision:int=8) -> None:
        """
        pptdata.txt: 
        seqs are serparated by empty line
        """
        with open(datapath,"r") as fin:
            print(f"Reading data from {datapath}")
            pptdata = fin.readlines()

        # category analysis
        topK_cates = 3
        kept_cates,unkept_cates = self.category_analysis(pptdata,topK_cates)   
        # seq and obj analysis
        self.seq_obj_analysis(pptdata)
        # cleaning
        # rm total seq if containing unwanted category
        # cleaning too long seq(>trunc_len obj)
        trunc_len = (max_length-2)//5 # rm <bos> and <eos>
        seq_data = self.clean_data(pptdata,unkept_cates,trunc_len)

        self.size = quant_size = pow(2,precision)
        self.categories = kept_cates
        cate2tok = {
            cate:id+quant_size for id,cate in enumerate(kept_cates)
        }
        print("Cate2Tok:",cate2tok)
        self.data = self.convert2seq(seq_data,cate2tok,quant_size)
        self.max_length = max_length

        self.vocab_size = self.size + len(self.categories) + 3 
        self.bos_token = self.vocab_size - 3
        self.eos_token = self.vocab_size - 2
        self.pad_token = self.vocab_size - 1
        self.colors = gen_colors(len(self.categories))

        self.transform = Padding(self.max_length, self.vocab_size)

    
    def render(self, layout):
        img = Image.new('RGB', (256, 256), color=(255, 255, 255))
        draw = ImageDraw.Draw(img, 'RGBA')
        layout = layout.reshape(-1)
        layout = trim_tokens(layout, self.bos_token, self.eos_token, self.pad_token)
        layout = layout[: len(layout) // 5 * 5].reshape(-1, 5)
        box = layout[:, 1:].astype(np.float32)
        box[:, [0, 1]] = box[:, [0, 1]] / (self.size - 1) * 255
        box[:, [2, 3]] = box[:, [2, 3]] / self.size * 256
        box[:, [2, 3]] = box[:, [0, 1]] + box[:, [2, 3]]

        for i in range(len(layout)):
            x1, y1, x2, y2 = box[i]
            cat = layout[i][0]
            col = self.colors[cat-self.size] if 0 <= cat-self.size < len(self.colors) else [0, 0, 0]
            draw.rectangle([x1, y1, x2, y2],
                           outline=tuple(col) + (200,),
                           fill=tuple(col) + (64,),
                           width=2)

        # Add border around image
        img = ImageOps.expand(img, border=2)
        return img

    def __getitem__(self, index) -> Tuple[ndarray,ndarray]:
        layout = torch.tensor(self.data[index], dtype=torch.long)
        layout = self.transform(layout)
        return layout['x'], layout['y']
    
    def __len__(self)->int:
        return len(self.data)
    
    def category_analysis(self,pptdata:List[str], topK:int)->Tuple[List[str],List[str]]:
        # category analysis
        print("Start Category analysis...")
        cate_data = [line.split()[0] for line in pptdata if line!='\n']
        total_obj_num = len(cate_data)
        catenum = {cate:0 for cate in cate_data}
        for cate in cate_data:
            catenum[cate] +=1
        # cateratio in descending number order
        des_cate_keys = sorted(catenum,key=lambda k:catenum[k],reverse=True)
        cateratio = {cate : catenum[cate]/total_obj_num 
                    for cate in des_cate_keys
                }
        accum = 0.
        print("Category\tRatio\tTopKaccum")
        for cate in des_cate_keys:
            accum += cateratio[cate]
            print(f"{cate} | {cateratio[cate]:.3f} | {accum:.3f}")
        print()
        print(f"Only keep {topK} most categories...")
        kept_cates = des_cate_keys[:topK]
        print(f"Only keep {kept_cates}...")
        unkept_cates = des_cate_keys[topK:]
        print(f"Unkeep {unkept_cates}...")
        return kept_cates,unkept_cates
    
    def seq_obj_analysis(self,pptdata:List[str]):
        print("Start Seq and Obj Number analysis...")
        seq_data = []
        objlen_data = []
        for seq in tqdm("".join(pptdata).split("\n\n")):
            seq = seq.split()
            seq_data.append(seq)
            objlen_data.append(len(seq)//5)
        print(f"Number of Seqs:{len(seq_data)}")
        print(f"MaxNumber of objs in a seq:{max(objlen_data)}")

    def clean_data(self,pptdata:List[str],unkept_cates:List[str],trunc_len:int)->List[List[str]]:
        print("Start data cleaning...")
        seq_data = [seq for seq in ''.join(pptdata).split("\n\n")]
        print(f"Before cleaning, Number of Seqs is {len(seq_data)}")
        seq_data = [seq for seq in seq_data if not any(u_cate in seq for u_cate in unkept_cates)]
        print(f"After cleaning unkept category{unkept_cates}\nNumber of Seqs is {len(seq_data)}")
        seq_data = [seq.split() for seq in seq_data]
        seq_data = [seq for seq in seq_data if len(seq) < 5 * trunc_len]
        print(f"After cleaning too long(numobj > {trunc_len}) seqs\nNumber of Seqs is {len(seq_data)}")
        return seq_data

    def convert2seq(self,seq_data:List[List[str]], cate2tok:Dict[str,int],quant_size:int)->List[ndarray]:
        print("Start preparing seqs...")
        seqs = []
        seq_num = len(seq_data)
        print(f"Total Number of Seqs: {seq_num}")
        for seq in tqdm(seq_data):
            seq = np.array(seq).reshape(-1,5)
            cates = np.array([cate2tok[cate] for cate in seq[:,0].tolist()])
            bbox = seq[:,1:].astype(float)
            # convert from x2,y2 to wh
            bbox[:,[2,3]] -= bbox[:,[0,1]] 
            bbox.clip(0,1)
            # quantize
            bbox = (bbox * quant_size).astype(int)
            # sorting from left2right and top2down
            idx = np.lexsort((bbox[:,0],bbox[:,1]))
            cates = cates[idx]
            bbox = bbox[idx]
            # flatten the seq
            seq = np.concatenate((cates.reshape(-1,1),bbox),axis=1).flatten()
            seqs.append(seq)
        print(f"Runs for sanity check...")
        objlen_data = [len(seq)//5 for seq in seqs]
        print(f"MaxNumber of obj:{max(objlen_data)}")
        print(f"First 10 seqs...")
        print("\n".join([repr(seq) for seq in seqs[:10]]))
        return seqs 


import typer
app = typer.Typer()
@app.command()
def ppt_test(datapath:Path):
    dataset = PPTLayout(datapath)
    print("Finish tesing")
    print(f"dataset is {dataset}")

if __name__ == "__main__":
    app()