# Variational Zero-shot Multispectral Pansharpening (IEEE TGRS 2024)

<p align="center">
    Xiangyu Rui, <a href="https://github.com/xiangyongcao">Xiangyong Cao</a>, <a href="https://github.com/YiningLi-ai">Yining Li</a>, <a href="https://gr.xjtu.edu.cn/web/dymeng">Deyu Meng</a>
</p>

<p align="center">

[Paper](https://ieeexplore.ieee.org/document/10744593)

**very easy to implement**
  
<img src="./imgs/m.png" align="center"> 

## Dataset
download benchmark WV2, WV3, QB **testing** datasets from [PanCollection](https://liangjiandeng.github.io/PanCollection.html) to your local files. (NO training data is required)

* .mat format could be used to directly run the code.
* .hd format is also available. Then, please change the data load way in PSDip.py and PSDip_f.py.

Please also change the data dir in line 89 in PSDip.py or line 86 in PSDip_f.py.

## Run the code
Please run ``python3 PSDip.py -sensor <sensor> -init`` for reduced resolution experiments.

run ``python3 PSDip_f.py -init`` for full resolution experiments.

e.g. ``python3 PSDip.py -sensor WV2 -init``

The restored HRMS image will be directly saved in 'results/..' files. 

## Connections
<a href="mailto:xyrui.aca@gmail.com">xyrui.aca@gmail.com</a> 

