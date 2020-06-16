# Monotone operator equilibrium networks

*Code to replicate the experiments in [the paper](https://arxiv.org/abs/2006.08591) by Ezra Winston and [Zico Kolter](http://zicokolter.com). More detailed tutorial coming soon.*

Monotone operator equilibrium networks (MONs) are a form of Deep Equilibrium Model ([Bai et al. 2019](https://arxiv.org/abs/1909.01377)) which guarantee convergence to a unique fixed-point. 

Unlike traditional deep networks which explicitly iterate some number of non-linear layers, deep equilibrium models directly solve for the fixed point of an "infinitely deep" weight-tied network. But how can we guarantee that such a fixed point exists, and that our fixed-point solver will converge to it? In practice, deep equilibrium models require extensive tuning in order to obtain stable convergence. 

## The MON framework
We recast the problem of finding the network fixed point as a form of _monotone operator splitting problem_, which can be solved using operator splitting methods such as _forward-backward_ or _Peaceman-Rachford_ splitting.

In fact, both solving for the network fixed-point and analytical backpropagation through the fixed point can be cast as operator splitting problems. Implementations of these methods applied to both problems can be found in `splitting.py`.

These methods will be guaranteed to converge to a solution provided the network weights obey a certain monotone constraint. We can directly parameterize networks such that this condition holds. Implementations for fully-connected, convolutional, and multi-scale convolutional networks can be found in `mon.py`. 

## Results on image benchmarks

MONs perform well on several image classification benchmarks. For example, the table shows performance of several instantiations of MON on CIFAR-10. We compare to the performance of Neural ODEs ([Chen et al. 2018](https://arxiv.org/abs/1806.07366)) and Augmented Neural ODEs ([Dupont et al. 2019](https://arxiv.org/abs/1904.01681)), the only other implicit-depth models like MON which guarantee existence of a network fixed-point.   
 
| Method                 | Model size | Acc.       |
|---------------------------------------|------------------------|---------------------------|
| Neural ODE             | 172K                   | 53.7%                  |
| Aug. Neural ODE | 172K                   | 60.6%                  |
| MON (ours\)                    |                        |                           |
| ---Single conv.           | 172K                   | **74.1%**      |
| ---Multi-tier           | 170K                   | 71.8%                  |
| ---Single conv. lg with data aug. | 854K                   | 82.5%             |
| ---Multi-tier lg with data aug. | 1M                     | **89.7%** |

The code to train these models and all those described in the results section of the paper is given in `examples.ipynb`.



## Requirements
Compatible with python 3.5+ and known to work with pytorch 1.4, torchvision 0.5, and numpy 1.18. Can install with `pip install -r requirements.txt`.


