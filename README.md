Replication of

# [Unsupervised Learning of Object Landmarks through Conditional Image Generation](http://www.robots.ox.ac.uk/~vgg/research/unsupervised_landmarks/)

[Tomas Jakab*](http://www.robots.ox.ac.uk/~tomj), [Ankush Gupta*](http://www.robots.ox.ac.uk/~ankush), Hakan Bilen, Andrea Vedaldi (* equal contribution).
Advances in Neural Information Processing Systems (NeurIPS) 2018.

in Pytorch by Duane

### installing

requires python 3.6

```
git clone https://github.com/duanenielsen/keypoints
python3 -m venv ~/.venv/keypoints
. ~/.venv/keypoints/activate
cd keypoints
pip3 install .
```

if you are running RTX card and want to use mixed precision you will need to install nvidia apex

http://github.com/NVIDIA/apex

follow the instructions

####install celeba dataset



### running

pong example with 16 bit precision

```
python3 transporter.py --run_id 2 --config configs/transporter_pong_grey.yaml
```


### basic command usage

```
python3 keypoints.py --run_id 1 --config configs/keypoints.yaml
```

### useful command line switches

run on a specific cuda device

```
--device cuda:1
```

disable apex (note, no optimization, ie: full 32 bit mode is  is O0 NOT 00)

```
--opt_level O0
```

