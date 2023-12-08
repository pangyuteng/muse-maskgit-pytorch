
import pandas as pd
from pathlib import Path

def gen_tl():
    directory = '/mnt/hd2/data/DeepLesion/Images_png'
    path_list = [{'png_path':str(x)} for x in Path(directory).rglob('*.png')]
    df = pd.DataFrame(path_list)
    df.to_csv('pngs.csv',index=False)



gen_tl()