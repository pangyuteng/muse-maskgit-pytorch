
docker run -it --runtime=nvidia -u $(id -u):$(id -g) -v /mnt:/mnt -w $PWD pangyuteng/muse-maskgit-pytorch:latest bash
