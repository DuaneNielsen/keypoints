from pathlib import Path
from argparse import ArgumentParser
import shutil

"""
Deletes all folders with small tensorboard run files
"""

parser = ArgumentParser('delete small runs')
parser.add_argument('--minsize', type=int, default=18000)
parser.add_argument('--list', type=bool, action='store_true')
args = parser.parse_args()

files = list(Path('.').glob('*/events*.*'))
rundirs = {}

for file in files:
    if file.parent in rundirs:
        max_size_so_far = rundirs[file.parent].stat().st_size
        if file.stat().st_size > max_size_so_far:
            rundirs[file.parent] = file
    else:
        rundirs[file.parent] = file

for parent, file in rundirs.items():

    if file.stat().st_size < args.minsize:
        print(file.parent, file.name, file.stat().st_size)
        if args.list:
            pass
        else:
            try:
                shutil.rmtree(str(file.parent))
            except:
                print(f"OS didn't let us delete {str(file.parent)}")