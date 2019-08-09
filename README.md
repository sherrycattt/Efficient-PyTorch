# Efficient-PyTorch
My best practice of training large dataset using PyTorch.

# Speed overview
By following the tips, we can reach achieve **~730 images/second** with PyTorch when training ResNet-50 on ImageNet. According to benchmark reported on [Tensorflow](https://www.tensorflow.org/performance/benchmarks) and [MXNet](https://github.com/apache/incubator-mxnet/tree/master/example/image-classification), the performance is still competitive.

```
Epoch: [0][430/5005]    Time 0.409 (0.405)      Data 626.6 (728.0)      Loss 6.8381 (6.9754)    Error@1 100.000 (99.850) Error@5 99.609 (99.259)
Epoch: [0][440/5005]    Time 0.364 (0.404)      Data 704.2 (727.9)      Loss 6.8506 (6.9725)    Error@1 100.000 (99.851) Error@5 99.609 (99.258)
Epoch: [0][450/5005]    Time 0.350 (0.403)      Data 730.7 (727.3)      Loss 6.8846 (6.9700)    Error@1 100.000 (99.847) Error@5 99.609 (99.258)
Epoch: [0][460/5005]    Time 0.357 (0.402)      Data 716.8 (727.4)      Loss 6.9129 (6.9680)    Error@1 100.000 (99.849) Error@5 99.609 (99.256)
Epoch: [0][470/5005]    Time 0.346 (0.401)      Data 740.8 (727.4)      Loss 6.8574 (6.9657)    Error@1 100.000 (99.850) Error@5 98.828 (99.249)
Epoch: [0][480/5005]    Time 0.425 (0.400)      Data 601.8 (727.3)      Loss 6.8467 (6.9632)    Error@1 100.000 (99.849) Error@5 99.609 (99.239)
Epoch: [0][490/5005]    Time 0.358 (0.399)      Data 715.2 (727.2)      Loss 6.8319 (6.9607)    Error@1 100.000 (99.848) Error@5 99.609 (99.232)
Epoch: [0][500/5005]    Time 0.347 (0.399)      Data 737.4 (726.9)      Loss 6.8426 (6.9583)    Error@1 99.609 (99.843)  Error@5 98.047 (99.220)
Epoch: [0][510/5005]    Time 0.346 (0.398)      Data 740.5 (726.7)      Loss 6.8245 (6.9561)    Error@1 100.000 (99.839) Error@5 99.609 (99.211)
Epoch: [0][520/5005]    Time 0.350 (0.452)      Data 730.7 (724.0)      Loss 6.8270 (6.9538)    Error@1 99.609 (99.834)  Error@5 97.656 (99.193)
Epoch: [0][530/5005]    Time 0.340 (0.450)      Data 752.9 (724.4)      Loss 6.8149 (6.9516)    Error@1 100.000 (99.832) Error@5 98.047 (99.183)
```

# Key Points of Efficiency 
Now most frameworks adapt `CUDNN` as their backends. Without special optimization, the inference time is similiar across frameworks. To optimize training time, we focus on other points such as 

## Data Loader
The default combination `datasets.ImageFolder` + `data.DataLoader` is not enough for large scale classification. According to my experience, even I switch to Samsung 960 Pro (read 3.5 GB/s, write 2.0 GB/s), whole training pipeline still suffers at disk I/O.

The reason causing is the slow reading of discountiuous small chunks. You should have experienced one or two times, for example, type `ls` command under original ImageNet validation folder. To optimize, we need to compress small JPEG images into a large binary file. TensorFlow has its own `TFRecord` and MXNet uses `recordIO`. Beside these two, there are many other options like `hdf5`, `pth`, `n5`, `lmdb` etc. Here I will choose `lmdb`, because of its super effienceny. 

### LMDB 
LMDB is a json-like, but in binary stream key-value storage. In my design, the format of converted LMDB is defined as follow.

key | value 
--- | ---
img-id1 | (jpeg_raw1, label1)
img-id2 | (jpeg_raw2, label2)
img-id3 | (jpeg_raw3, label3)
... | ...
img-idn | (jpeg_rawn, labeln)
`__keys__` | [img-id1, img-id2, ... img-idn]
`__len__` | n

As for details of reading/writing, please refer to [code](tv_data_loader.py).


### Convert `DatasetsFolder` to `LMDB`
```bash
python create_lmdb.py --data_dir ~/torch_data/ --name train --dataset ILSVRC --out_dir ~/torch_data-lmdb
```

### ImageFolderLMDB
The usage of `ImageFolderLMDB` is identical to `torchvision.datasets`. 

```python
from tools.tv_data_loader import ImageFolderLMDB
from torch.utils.data import DataLoader
dst = ImageFolderLMDB(path, transform, target_transform)
loader = DataLoader(dst, batch_size=64)
```


## Data Parallel (On-going)
The default data parallel of PyTorch, powerd by `nn.DataParallel` is in-efficienct! 

