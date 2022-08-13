import typer
from typing import List, Optional

import os,subprocess,shlex
from subprocess import Popen, PIPE
from pathlib import Path
import shutil

import requests, tarfile
from io import BytesIO
import urllib

app = typer.Typer()

def typer_run(cmd:str,debug:bool=False,*args,**kwargs):
    cmdl = shlex.split(cmd) 
    typer.echo(f"running cmd is {cmdl}")
    if not debug:
        subprocess.run(cmdl,*args,**kwargs)
        typer.echo(f"Finish running!")

@app.command()
def hold(time:str='7d'):
    typer.echo(f"hold the server for {time}")
    cmd = f'sleep {time}'
    typer_run(cmd)

def azcopy_setup():
    """
    download, extract and cp azcopy to current working dir
    """
    url = "https://aka.ms/downloadazcopy-v10-linux"
    typer.echo(f"downloading azcopy from {url}...")
    r = requests.get(url)
    fileobj = BytesIO(r.content) # which could be utilize later to iter_content
    tarfile.open(fileobj=fileobj).extractall()
    azcopy_dir = Path([f for f in os.listdir() if 'azcopy_linux' in f][0])
    shutil.copy(azcopy_dir/'azcopy',".")

def fish_setup():
    cmds = ["sudo apt-add-repository ppa:fish-shell/release-3", 
            "sudo apt update","sudo apt install fish -y"
        ]
    for cmd in cmds:
        typer_run(cmd)

def tmux_setup():
    cmds = ["git clone https://github.com/gpakosz/.tmux.git",
            "ln -s -f .tmux/.tmux.conf",
            "cp .tmux/.tmux.conf.local .",
        ]
    cwd = os.path.expanduser("~") # which is the ~
    for cmd in cmds:
        typer_run(cmd,cwd=cwd)
    with open(f"{cwd}/.tmux.conf.local","a") as fout:
        # set up fish and vim mode
        typer.echo("setting up fish and vi mode for tmux...")
        fout.writelines(["set-option -g default-shell /usr/bin/fish\n",
                        "setw -g mode-keys vi" ])

@app.command()
def tool_setup(only_azcopy:bool=False):
    typer.echo("Starting azcopy setting...")
    azcopy_setup()
    if only_azcopy :
        typer.echo("only azcopy is set up...")
    else:
        typer.echo("Start fish setting...")
        fish_setup()
        typer.echo("Start tmux setting...")
        tmux_setup()

sas_d = {
    "bingdatawu2premium":"sv=2020-08-04&ss=b&srt=sco&sp=rwdlacix&se=2023-03-31T18:21:49Z&st=2022-04-24T10:21:49Z&spr=https&sig=ICaizAliYjo%2B7IIXfLz92LoWLOHTURweWvHhuW1w88k%3D",
    "zeliuus":"sv=2020-10-02&st=2022-05-26T07%3A00%3A25Z&se=2023-06-27T07%3A00%3A00Z&sr=c&sp=racwdxlt&sig=vH2BBEAo%2BbOLzcgIp1uG6%2Bvcj6P%2Fu1f%2F%2FagK%2Fyzj3Sk%3D"
}

def get_datapath(data:str):
    k12 = ['caltech101','flowers102'] # not exhausted
    if data=="imagenet1000":
        datapath = Path("reim_laydrop_jch")/data
    elif data in k12:
        datapath = Path("t-zhxie/data")/data
    return datapath

@app.command()
def data_download(data:str,debug:bool=False):
    blob = "bingdatawu2premium"
    container = Path("fwd-data")
    sas=sas_d[blob]
    data_path = container/get_datapath(data)
    cmd = f"./azcopy cp https://{blob}.blob.core.windows.net/{data_path}/?{sas} . --recursive"
    typer_run(cmd,debug=debug)

@app.command()
def ckp_download(ckpurl:str,des:Path=Path("."),debug:bool=False):
    """
    from ckpurl download ckp to des
    """
    purl_res = urllib.parse.urlparse(ckpurl)
    blob = purl_res.netloc.split(".")[0]
    sas = sas_d[blob]
    cmd = f"./azcopy cp {ckpurl}?{sas} {des}"
    typer_run(cmd,debug=debug)


if __name__ == "__main__":
    app()
