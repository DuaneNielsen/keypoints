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

if you are running RTX card and want to use mixed precision you will need to install NVIDIA apex

http://github.com/NVIDIA/apex

follow the instructions on their readme to install

you can disable mixed precision from command line later if it's too much hassle for you

####install celeba dataset

faces requires celeba dataset, download https://drive.google.com/open?id=0B7EVK8r0v71pWEZsZE9oNnFzTm8


### running

pong example with 16 bit precision

```
python3 transporter.py --run_id 2 --config configs/transporter_pong_grey.yaml
```

if you dont have RTX card, or can't be bothered with mixed precision you can disable, but you may need to adjust minibatch size, use the flags

```
--opt_level O0 --batch_size 16
```

if you get GPU memory errors, reduce batch size until it fits on your card


### basic command usage

```
python3 keypoints.py --run_id 1 --config configs/keypoints.yaml
```

### useful command line switches

run on a specific cuda device

```
--device cuda:1
```

disable apex (note, full 32 bit mode (no optimization) is O0 NOT 00)

```
--opt_level O0
```

display the run live, update display every 100 minibatches

```
--display --display_freq 100
```

