# CAT: Beyond Efficient Transformer for Content-Aware Anomaly Detection in Event Sequences
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg?style=plastic)
![PyTorch 1.2](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?style=plastic)
![cuDNN 7.3.1](https://img.shields.io/badge/cudnn-7.3.1-green.svg?style=plastic)
![License CC BY-NC-SA](https://img.shields.io/badge/license-CC_BY--NC--SA--green.svg?style=plastic)

This is the origin Pytorch implementation of CAT in the following paper: 
[CAT: Beyond Efficient Transformer for Content-Aware AnomalyDetection in Event Sequences].

<p align="center">
<img src=".\img\Architecture.PNG" height = "300" alt="" align=center />
<br><br>
<b>Figure 1.</b> The architecture of CAT.
</p>

## Requirements

- Python 3.6
- matplotlib == 3.1.1
- numpy == 1.19.4
- pandas == 0.25.1
- scikit_learn == 0.21.3
- torch == 1.8.0

Dependencies can be installed using the following command:
```bash
pip install -r requirements.txt
```

## Data

The log datasets used in the paper can be found in the repo [loghub](https://github.com/logpai/loghub).
In this repository, an small sample of the HDFS dataset is proposed for a quick hands-up.

For generating the Log template files, please refer to the official implementation repo of [logparser](https://github.com/logpai/logparser).


## Usage

The simplest way of running CAT is to run `python main_cat.py --data HDFS`.


## Contact
If you have any questions, feel free to contact Shengming Zhang through Email (shengming.zhang@rutgers.edu) or Github issues.
