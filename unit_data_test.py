import typer
from pathlib import Path
import os
import json

app = typer.Typer()

def isleaf_dir(abs_path:Path):
    return any(f.endswith(".json") for f in os.listdir(abs_path))

def traverse(cur_abs_path:Path):
    if isleaf_dir(cur_abs_path):
        print(f"{cur_abs_path}:")
        json_files = [f for f in os.listdir(cur_abs_path) if f.endswith(".json")]
        for json_file in json_files:
            json_path = cur_abs_path/json_file
            print(f"{json_path}:")
            with open(json_path,"r") as fin:
                data = json.load(fin)[0]
                slides = data['slides']
                for slide in slides:
                    shapes = slide['shapes']
                    seq = []
                    for shape in shapes:
                        cate = shape['Type']
                        pos0 = shape['pos0']
                        pos1 = shape['pos1']
                        print(cate,*pos0,*pos1)
                    print()
    else:
        for childdir in os.listdir(cur_abs_path):
            childdir_path = cur_abs_path/childdir
            if os.path.isdir(childdir_path):
                traverse(cur_abs_path/childdir)

@app.command()
def run(dir_path:Path):
    traverse(dir_path)
    
from .layout_transformer.dataset import PPTLayout

@app.command()
def pptdata_test(datapath:Path):
    dataset = PPTLayout(datapath)
    print("Finish Testing!")
    print(f"Dataset is {dataset}")


if __name__ == "__main__":
    app()


