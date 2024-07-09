# Variational Zero-shot Multispectral Pansharpening

[arxiv]()

## Dataset
download benchmark WV2, WV3, QB datasets from [PanCollection](https://liangjiandeng.github.io/PanCollection.html) to your local files.

* .mat format could be used to directly run the code.
* .hd format is also available. Then, please change the data load way in PSDip.py and PSDip_f.py.

Please also change the data dir in PSDip.py and PSDip_f.py.

## Run the code
Please run ``PSDip.py -sensor <sensor>`` for reduced resolution experiments.

run ``PSDip_f.py`` for full resolution experiments.

e.g. ``PSDip.py -sensor WV2``

The restored HRMS image will be directly saved in 'results/..' files. 

## Connections
<a href="mailto:xyrui.aca@gmail.com">xyrui.aca@gmail.com</a> 

