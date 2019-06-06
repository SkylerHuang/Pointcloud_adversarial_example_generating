### Simple Pointcloud adversarial example generate

![avatar](https://github.com/SkylerHuang/Pointcloud_adversarial_example_generating/blob/master/adversarial_example.png)



### Introduction

This code was based on Pointnet(https://github.com/charlesq34/pointnet) and Pointnet++(https://github.com/charlesq34/pointnet2). And use simple point coordinate perturbation to generate adversarial examples.

Main idea :

Randomly generate a small noise as train parameter, add to the original point cloud

Setting a target label call y_target

Through gradient back propagation to update noisy,

In order to make the adversarial sample  same as the original sample,we need add a constraint to Limit the magnitude of the noise.

So we define a cost function:
$$
Loss = \frac{1}{2}||y-\vec y_{target}||^2_2+\lambda D(x,x')
$$
Here x is original point cloud, x' is perturbation point cloud, D is the distance of them.lambda is a trade-off between adversarial loss and perturbation magnitude.

### Pretrain model

For classification tasks based on Modelnet40:

pointnet:  https://drive.google.com/file/d/1ecfkbyGG3rLrNxhBA60iA125WhiTgQi9/view?usp=sharin



pointnet++:

https://drive.google.com/file/d/1gx5s72TaxWnMVLKEh5iWpqko0w3JeDEq/view?usp=sharing

Download it and put it in log

### Dataset

you must download   <a href="https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip" target="_blank">(Modelnet datasets) </a> and move it to data dir

### Usage

--need_label is the class you will attack

```python
python2 train.py --need_label 0 --target_label 5
```

then you will get a file which contain the original point cloud and perturbation vector.

You can add the two together to obtain adversairal example.



### Next work

* Other type adversarial attack

  



