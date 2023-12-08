

def cache_png_file_paths():
    directory = '/mnt/scratch/data/DeepLesion/Images_png'
    path_list = [{'png_path':str(x)} for x in Path(directory).rglob('*.png')]
    df = pd.DataFrame(path_list)
    df.to_csv('pngs.csv',index=False)



docker run -it --runtime=nvidia -u $(id -u):$(id -g) -v /mnt:/mnt -w $PWD pangyuteng/muse-maskgit-pytorch:latest bash
