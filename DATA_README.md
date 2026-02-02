**基础用法**

***数据处理***

先将数据整理到一个csv中，包含``video_path`` 和``split``两列。然后调用``build_dataset.py``将数据转换成我们需要的metadata格式。也可以不包含``split``，在``build_datase.py``中手动更改``split`` 是``training`` 或者 ``validation``

对于每一个video data，参考``build_dataset.py``里面对应一个action path。在npz文件里的``actions`` key中即可
对于目前的模型，只支持长度为49的video chunk

***训练命令***

参考``run.sh``:

``load``字段可以用对应wandb的runid，例如``ehwb3z4b``或者也可以直接使用对应的ckpt路径。ckpt会存在output当中，然后``outputs/checkpoint_links/run_id``存有对应wandb run的ckpt软连接。 记得加上``ehwb3z4b:model``其中model代表只load model而不是其他optimizer啥的state

``dataset.metadata_path``就用上面``build_dataset``生成的metadata的csv路径。metadata中的各个字段可以参考``build_dataset.csv``.

``name``就是这个run的wandb name

``experiments.tasks``默认是``['training']`` ，在跑eval例如rollout的时候可以改成``['validation']``

建议可以在ft candidate data的时候把eval set的metadata直接粘贴在训练的metadata.csv后面，注意表明split为validation，即可在训练中自动进行eval


***rollout***

如果需要记录rollout candidate data中每一个data的error，可以前往 ``algorithms/wan/wan_t2v.py``, Line 610-621, 注释掉那一段代码里面可以修改log每一个sample的mse error的路径

***neural net***

目前github remote里面的``algorithms/wan/modules/action_encoder.py``里面对于action的1D Conv实现有一定问题，kernel size 3，stride 4导致有的actions没有读进来。本地版本里面修改了相应实现改为两个stride 2的Conv。可以看情况进行修改。

