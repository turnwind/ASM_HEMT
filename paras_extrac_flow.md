## 提取流程
### DC Modeling

在ICCAP中，DC建模相对复杂，其中穿插着CV等其它建模

本节旨在梳理5个DC相关的plot所要提取的参数

#### Transfer_lin.mdm

**step_1**
| Paras | Min | Ini | Max|
|:------|:-------:|:------:|------:|
| Voff | | -2 | |
| Nfactor | 0| 1.5 |5.0 |
| U0 |  1m| 65.4 |500m |
| UA |  1.0n| 29.36n |500.0n |

**step_3**
| Paras | Min | Ini | Max|
|:------|:-------:|:------:|------:|
| Voff | | -2 | |
| Nfactor | 0| 1.5 |5.0 |
| U0 |  1m| 65.4 |500m |
| UA |  1.0n| 29.36n |500.0n |

#### Input.mdm

**step_2**
| Paras | Min | Ini | Max|
|:------|:-------:|:------:|------:|
| Igsdio | | 101.02m ||
| Njgs |  | 2.679| |
| Igddio |  | 12.9 | |
| Njgd |  |3.120  | |
| Rshg | 0 | 1.00m | |

#### Transfer_sub.mdm

**step_4**
| Paras | Min | Ini | Max|
|:------|:-------:|:------:|------:|
| Eta0 | 0 | 103.3m |200m |
| Vdscale | 0 | 5.644|10 |
| Cdscd |  | 0 | |

#### Transfer.mdm

**step_5**
| Paras | Min | Ini | Max|
|:------|:-------:|:------:|------:|
| Rsc | 0 | 460u | |
| Rdc |  |1.215m  | |
| U0 |  1m| 65.4 |500m |
| UA |  1.0n| 29.36n |500.0n |
| UTE |-10  |-598.6m  |0 |
| RTH0 |  | 24 | |
| LAMBDA |  | 2.132| |
| Vsat | 1.0K | 256.3K |500K |


#### Output.mdm

**step_6**
| Paras | Min | Ini | Max|
|:------|:-------:|:------:|------:|
| RTH0 |  | 24 | |
| LAMBDA |  | 2.132| |
| Vsat | 1.0K | 256.3K |500K |
| Eta0 | 0 | 103.3m |200m |
| Vdscale | 0 | 5.644|10 |
| Cdscd |  | 0 | |
| Rdc |  |1.215m  | |
| Tbar |  | 11.40n| |