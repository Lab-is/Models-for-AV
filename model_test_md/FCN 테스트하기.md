# FCN 테스트하기

모든 모델에서 epoch를 10으로 맞추고 학습시킴

# IoU란?

> **IoU(Intersection over Union, 교차점 집합체)는 예측된 경계 상자와 정답 경계 상자 사이의 중첩 정도를 지정하는 0에서 1 사이의 숫자.** 보통 mean average precision을 계산하는데 사용함.
- 출처 : 머신러닝 블로그, [https://towardsdatascience.com/iou-a-better-detection-evaluation-metric-45a511185be1](https://towardsdatascience.com/iou-a-better-detection-evaluation-metric-45a511185be1)
> 

![Untitled](/model_test_md/FCN%20%ED%85%8C%EC%8A%A4%ED%8A%B8%ED%95%98%EA%B8%B0/Untitled.png)

즉 전체 픽셀 중에서 클래스 값이 겹치는 픽셀 수의 비율을 IoU라고 볼 수 있으며, 얼마나 segmentation 성능이 좋은지 평가하는 척도로 사용할 수 있다.

# Unet과 Segnet 테스트

## Intro

라이브러리 내에 있는 evaluate_model을 활용하여 

클래스 개수를 많이 설정했기에 유효한 클래스 내에서 성능을 평가하는 데 있어 봐야할 척도는 **frequency_weighted_IU**이다.

### vgg_unet → 0.709782

```python
2022-07-15 06:37:08.635490: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:39] Overriding allow_growth setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.
loaded weights  /tmp/vgg_unet_1.00005
0it [00:00, ?it/s]2022-07-15 06:37:11.693825: W tensorflow/core/common_runtime/bfc_allocator.cc:343] Garbage collection: deallocate free memory regions (i.e., allocations) so that we can re-allocate a larger region to avoid OOM due to memory fragmentation. If you see this message frequently, you are running near the threshold of the available device memory and re-allocation may incur great performance overhead. You may try smaller batch sizes to observe the performance impact. Set TF_ENABLE_GPU_GARBAGE_COLLECTION=false if you'd like to disable this feature.
101it [00:19,  5.32it/s]
{'frequency_weighted_IU': 0.7097825978962239, 'mean_IU': 0.17239401516467234, 'class_wise_IU': array([0.9402354 , 0.53302012, 0.0564702 , 0.93279325, 0.73467599,
       0.8002994 , 0.31354082, 0.28472422, 0.26927778, 0.12163762,
       0.04598814, 0.1391575 , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ])}
```

### unet_mini → 0.730876

```python
2022-07-15 06:57:41.162094: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:39] Overriding allow_growth setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.
loaded weights  /tmp/vgg_unet_mini_1.00010
101it [00:29,  3.40it/s]
{'frequency_weighted_IU': 0.7308766751119348, 'mean_IU': 0.1781819784270099, 'class_wise_IU': array([0.91980752, 0.73194236, 0.04420023, 0.85812857, 0.53708521,
       0.82243875, 0.21284222, 0.33812509, 0.38118221, 0.16109102,
       0.15014513, 0.18847105, 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ])}
```

### unet → 0.805472

```python
2022-07-15 07:08:26.511894: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:39] Overriding allow_growth setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.
loaded weights  /tmp/unet1.00010
101it [00:18,  5.48it/s]
{'frequency_weighted_IU': 0.8054728038019441, 'mean_IU': 0.209977327464878, 'class_wise_IU': array([0.94073797, 0.78641363, 0.15023278, 0.93375852, 0.75803843,
       0.8813807 , 0.31801774, 0.47752288, 0.58163822, 0.20358787,
       0.05451275, 0.21347834, 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ])}
```

### resnet50_unet → 0.811940

```python
2022-07-15 07:10:03.160550: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:39] Overriding allow_growth setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.
loaded weights  /tmp/resnet50_unet.00010
101it [00:14,  6.98it/s]
{'frequency_weighted_IU': 0.8119405470730613, 'mean_IU': 0.23963543776675547, 'class_wise_IU': array([0.92189846, 0.76373645, 0.13533429, 0.9663021 , 0.85388055,
       0.76352607, 0.40990873, 0.47733476, 0.7192379 , 0.39965969,
       0.57363791, 0.20460623, 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ])}
```

### mobilenet_unet → 0.872027

```python
2022-07-15 07:14:46.721846: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:39] Overriding allow_growth setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.
loaded weights  /tmp/mobilenet_unet.00010
101it [00:15,  6.59it/s]
{'frequency_weighted_IU': 0.8720273857196631, 'mean_IU': 0.2715637701890002, 'class_wise_IU': array([0.94105217, 0.85864259, 0.13633014, 0.96114402, 0.84338634,
       0.89273382, 0.40275988, 0.66297377, 0.81635081, 0.53555031,
       0.79433912, 0.30165013, 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ])}
```

### segnet → 0.597346

```python
2022-07-15 07:18:07.621649: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:39] Overriding allow_growth setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.
loaded weights  /tmp/segnet.00010
101it [00:15,  6.36it/s]
{'frequency_weighted_IU': 0.5973464293723921, 'mean_IU': 0.1463757606497467, 'class_wise_IU': array([0.91773161, 0.52287038, 0.0135031 , 0.73425847, 0.2754709 ,
       0.72508776, 0.27118343, 0.33196756, 0.25514701, 0.12451399,
       0.11707287, 0.10246575, 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ])}
```

### vgg_segnet → 0.780580

```python
2022-07-15 07:19:20.419770: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:39] Overriding allow_growth setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.
loaded weights  /tmp/vgg_segnet.00010
101it [00:13,  7.38it/s]
{'frequency_weighted_IU': 0.7805806072242188, 'mean_IU': 0.19961270284382446, 'class_wise_IU': array([0.92066114, 0.73411369, 0.00323938, 0.93109188, 0.68815444,
       0.85767466, 0.27096937, 0.41488834, 0.39868537, 0.15591116,
       0.47043788, 0.14255377, 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ])}
```

### resnet50_segnet → 0.843303

```python
2022-07-15 07:29:33.460601: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:39] Overriding allow_growth setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.
loaded weights  /tmp/resnet50_segnet.00010
101it [00:16,  6.08it/s]
{'frequency_weighted_IU': 0.8433035066702697, 'mean_IU': 0.24393821135920363, 'class_wise_IU': array([0.92840344, 0.82333439, 0.0884593 , 0.96171379, 0.81669775,
       0.88270003, 0.35527331, 0.48254576, 0.65642657, 0.46800658,
       0.60597325, 0.24861217, 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ])}
```

### moblienet_segnet → 0.895545

```python
2022-07-15 07:29:29.134080: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:39] Overriding allow_growth setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.
loaded weights  /tmp/mobilenet_segnet.00010
101it [00:15,  6.41it/s]
{'frequency_weighted_IU': 0.8590877571129518, 'mean_IU': 0.2530016233382121, 'class_wise_IU': array([0.91852141, 0.8397151 , 0.03750361, 0.96048497, 0.85322056,
       0.89554529, 0.40767495, 0.62306522, 0.76876723, 0.3413064 ,
       0.67442648, 0.26981749, 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ])}
```