import copy
import csv
import os
import re
import pickle

import jax
import numpy as np
import jax.numpy as jnp
from itertools import product



def print_tree_shapes(params):
    """
    Prints the shapes of the parameters in a tree structure.

    Args:
        params (dict): A dictionary of parameters.
    """
    print("Tree shape: ", jax.tree.map(lambda x: x.shape, params))

def create_directory(dirs):
    directory_name = ""
    for i,dir in enumerate(dirs):
        directory_name += dirs[i] + "/"
        try:
            os.mkdir(directory_name)
        except:
            pass

def picklit(obj,path,name):
    create_directory([f"../../{path}/"])
    filename = f"../../{path}/{name}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def picklin(path,name):
    with open(f'../../{path}/{name}.pkl', 'rb') as f:
        return pickle.load(f)
    
def print_header():
    l1 = f" epoch | accuracy | cost | runtime "
    l2 =  " --------------------------------- "
    print(f"\n{l1}\n{l2}")

def spaced_string(header,item):
    lh = len(header)
    li = len(str(item))
    s1 = (lh - li)//2
    s2 = lh-(s1+li)
    string = s1*" "+str(item)+s2*" "
    return string

def print_progress(epoch,model,runtime,nextline):
    headers = [" epoch "," accuracy "," cost "," runtime "]
    data = [epoch,f"{model.accs[-1]:.3}",f"{model.costs[-1]:.3}",f"{runtime:.3}", f"{model.name}"]

    strings = [spaced_string(headers[i],data[i]) for i,head in enumerate(headers)]
    line = "|".join(strings)
    # line = f"{string1}| {model.accs[-1]:.3} | {model.costs[-1]:.3} | {runtime:.3}"
    if nextline==True:
        print(line)
    else:
        print(line,end="\r")

def print_dict(dct,keys=None):
    """
    Prints the contents of a dictionary.
    
    Args:
        dct (dict): The dictionary to print.
    """
    if type(dct)!=dict:
        dct = dct.__dict__

    if keys is None: keys = list(dct.keys())
    space = max([len(key) for key in keys]) + 1
    for key in keys:
        if "__" not in key: 
            try:
                val = dct[key].shape
            except:
                try:
                    val = jax.tree.map(lambda x: x.shape, dct[key])
                except:
                    try:
                        val = dct[key].__name__
                    except:
                        val = dct[key]
            gap = space - len(key)
            if type(dct[key])==float:
                print(f"  {key}{' '*gap}: {val}")
            else:
                print(f"  {key}{' '*gap}: {val}")