# CZII_kaggle_3d_segmentation
this is kaggle competition CZII. task is 3d segmentation with micro proteins


## kaggle environment with Docker

```sh
docker compose build

# using bash
docker compose run --rm kaggle bash 

# using jupyter lab
docker compose up 
```

## execute script

```sh
python experiments/sample/run.py exp=001
python experiments/sample/run.py exp=base
```