# FluxEV: A Fast and Effective Unsupervised Framework for Time-Series Anomaly Detection
The code is for our paper ["FluxEV: A Fast and Effective Unsupervised Framework for Time-Series Anomaly Detection"](https://dl.acm.org/doi/10.1145/3437963.3441823) 
and this paper has been accepted by WSDM 2021.

## Requirements
* numpy
* numba
* scipy
* pandas
* scikit-learn
* matplotlib (plot)
* more-itertools (plot)

## Datasets
1. KPI. <http://iops.ai/dataset_detail/?id=10>
2. Yahoo. <https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70>

## Instructions
`preprocessing.py`: 
* Fill the missing points for KPI dataset.

`spot_pipe.py`: 
* SPOT function is modified to be a part of FluxEV for streaming detection;
* MOM(Method of Moments) is added as one of parameter estimation methods;
* For the original code, please refer to [SPOT (Streaming Peaks-Over-Threshold)](https://github.com/Amossys-team/SPOT)

`eval_methods.py`: 
* The adjustment strategy is consistent with AIOps Challenge, [KPI Anomaly Detection Competition](http://iops.ai/competition_detail/?competition_id=5&flag=1).
* The original evaluation script is available at [iops](https://github.com/iopsai/iops/blob/master/evaluation/evaluation.py).

`main.py`: 
* Implement streaming detection of FluxEV.

## Run
```
python main.py --dataset=KPI
```

```
python main.py --dataset=Yahoo
```

## Citation
```
@inproceedings{li2021fluxev,
  title={FluxEV: A Fast and Effective Unsupervised Framework for Time-Series Anomaly Detection},
  author={Li, Jia and Di, Shimin and Shen, Yanyan and Chen, Lei},
  booktitle={Proceedings of the 14th ACM International Conference on Web Search and Data Mining},
  pages={824--832},
  year={2021}
}
```

