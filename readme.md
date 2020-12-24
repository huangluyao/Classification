# Classification

Research on classification of deep learning

## Installation

This project has the following dependencies:

- Numpy 

  ```
  sudo pip install numpy
  ```

- OpenCV Python

  ```
   sudo apt-get install python-opencv
  ```

- pytorch 

  ```
   pip install torch==1.1.0 torchvision==0.3.0
  ```

- dqtm

  ```
  sudo pip install dqtm
  ```

## Datasets

 Datasets should under the database directory, which structure is as follows:

```
-─psa_top_rgb_cls
    ├─train
    │  ├─0good
    │  ├─1shift
    │  ├─2empty
    │  └─3double
    └─val
        ├─0good
        ├─1shift
        ├─2empty
        └─3double
```

## Third-party resources

- [Albumentations](https://albumentations.ai/) are used for data augmentation

## Train

- Training the model

  ```
  python train.py --config configs/PSCTop_MobileNetv3.json
  ```

  

- Knowledge Distillation

  ```
  python distilling.py --config configs/distilling.json
  ```

## Results

- The best score of MobileNetv3 on F1Score was 99.43% with **HorizontalFlip**
- The best score of EfficientNet on F1Score was 100% with **ColorJitter**
- The best score of MobileNetv3 on F1Score was 100% by **knowldege distillation**

#### Four tests on MobileNetv3

| 数据增强                 | 网络模型    | 输入尺寸 | 模型最高精度（epoch60） | 模型最高精度（epoch60） | 模型最高精度（epoch60） | 模型最高精度（epoch61） | 均值   | 方差(乘以10的4次方) | 参数                                                         |
| ------------------------ | ----------- | -------- | ----------------------- | ----------------------- | ----------------------- | ----------------------- | ------ | ------------------- | ------------------------------------------------------------ |
| ColorJitter              | MobileNetV3 | 224      | 0.9898                  | 0.9921                  | 0.9876                  | 0.9910                  | 0.9901 | 0.0369              | ColorJitter brightness:0.2 contrast:0.2 saturation:0.2 hue:0.2 prob:0.3 |
| HorizontalFlip           | MobileNetV3 | 224      | 0.9894                  | 0.9894                  | 0.9864                  | 0.9943                  | 0.9899 | 0.1061              | type:HorizontalFlip prob:0.5                                 |
| RandomCrop               | MobileNetV3 | 224      | 0.9841                  | 0.9932                  | 0.9887                  | 0.9910                  | 0.9892 | 0.1506              | type:Resize size:[230, 230] prob:1  type:RandomCrop size:[224, 224] prob:1 |
| Rotate                   | MobileNetV3 | 224      | 0.9886                  | 0.9894                  | 0.9899                  | 0.9887                  | 0.9891 | 0.0036              | prob:0.2   limit:5 interpolation:linear border_mode:constant value:0 |
| ChannelShuffle           | MobileNetV3 | 224      | 0.9931                  | 0.9887                  | 0.9853                  | 0.9865                  | 0.9884 | 0.1191              | ChannelShuffle prob:0.2                                      |
| GaussNoise               | MobileNetV3 | 224      | 0.9838                  | 0.9887                  | 0.9865                  | 0.9910                  | 0.9875 | 0.0938              | GaussNoise  mean:0.5   prob:0.2                              |
| Blur                     | MobileNetV3 | 224      | 0.9887                  | 0.9884                  | 0.9854                  | 0.9832                  | 0.9864 | 0.0681              |                                                              |
| ChannelDropout           | MobileNetV3 | 224      | 0.9828                  | 0.9910                  | 0.9854                  | 0.9854                  | 0.9861 | 0.1189              | ChannelDropout prob:0.3 fill_value:0.9                       |
| RandomGridShuffle        | MobileNetV3 | 224      | 0.9883                  | 0.9887                  | 0.9805                  | 0.9862                  | 0.9859 | 0.1429              | type:RandomGridShuffle prob:0.2 grid:[1, 2]  type:RandomGridShuffle prob:0.2 grid:[2, 1] |
| CenterCrop               | MobileNetV3 | 224      | 0.9772                  | 0.9921                  | 0.9820                  | 0.9876                  | 0.9847 | 0.4196              | type:Resize size:[230, 230] prob:1 type:CenterCrop size:[224, 224] prob:1 |
| CoarseDropout            | MobileNetV3 | 224      | 0.9815                  | 0.9876                  | 0.9831                  | 0.9864                  | 0.9846 | 0.0799              | prob:0.2 max_height:15 max_width:15 max_holes:5              |
| Equalize                 | MobileNetV3 | 224      | 0.9751                  | 0.9932                  | 0.9875                  | 0.9827                  | 0.9846 | 0.5865              | prob:0.2                                                     |
| GaussianBlur             | MobileNetV3 | 224      | 0.9849                  | 0.9898                  | 0.9770                  | 0.9864                  | 0.9845 | 0.2971              | GaussianBlur blur_limit:[1, 3] sigma_limit:0 prob:0.2        |
| HueSaturationValue       | MobileNetV3 | 224      | 0.9887                  | 0.9898                  | 0.9721                  | 0.9864                  | 0.9843 | 0.6776              | HueSaturationValue prob:0.2 hue_shift_limit:5 sat_shift_limit:5 val_shift_limit:5 |
| RandomBrightnessContrast | MobileNetV3 | 224      | 0.9814                  | 0.9909                  | 0.9808                  | 0.9825                  | 0.9839 | 0.2216              | RandomBrightnessContrast prob:0.2 brightness_limit:0.1 contrast_limit:0.1 |
| MultiplicativeNoise      | MobileNetV3 | 224      | 0.975                   | 0.9921                  | 0.9808                  | 0.9864                  | 0.9836 | 0.5370              | MultiplicativeNoise prob:0.3 multiplier:[0.9, 1.1]           |
| FancyPCA                 | MobileNetV3 | 224      | 0.9865                  | 0.9909                  | 0.9713                  | 0.9841                  | 0.9832 | 0.7111              | prob:0.3 alpha:0.1                                           |
| GridDistortion           | MobileNetV3 | 224      | 0.9785                  | 0.9808                  | 0.9843                  | 0.9865                  | 0.9825 | 0.1271              | GridDistortion prob:0.2 num_steps:5 distort_limit:0.3        |
| FromFloat                | MobileNetV3 | 224      | 0.9836                  | 0.9822                  | 0.9796                  |                         | 0.9818 | 0.0416              | FromFloat  prob:0.3 max_value:0.95                           |
| RandomFlip               | MobileNetV3 | 224      | 0.9779                  | 0.9843                  | 0.9765                  | 0.9865                  | 0.9813 | 0.2343              | prob:0.3                                                     |
| GlassBlur                | MobileNetV3 | 224      | 0.9838                  | 0.9826                  | 0.9735                  | 0.9831                  | 0.9808 | 0.2374              | GlassBlur  sigma:0.7 max_delta:4 iterations:2 prob:0.2       |
| -                        | MobileNetV3 | 224      | 0.9743                  | 0.9898                  | 0.9743                  | 0.9842                  | 0.9807 | 0.5861              |                                                              |
| CLAHE                    | MobileNetV3 | 224      | 0.9834                  | 0.9823                  | 0.9761                  | 0.9804                  | 0.9806 | 0.1020              | CLAHE  clip_limit:10 prob:0.2                                |
| Downscale                | MobileNetV3 | 224      | 0.985                   | 0.9776                  | 0.9737                  | 0.9820                  | 0.9796 | 0.2431              | scale_min:0.25 scale_max:0.25 prob:0.2                       |
| ElasticTransform         | MobileNetV3 | 224      | 0.9722                  | 0.9854                  | 0.9794                  | 0.9806                  | 0.9794 | 0.2974              | prob:0.2                                                     |

#### Augmentation results

| 增强手段                 | 前五出现次数 | 第一名出现次数 | 倒第五出现次数 | 总次数 | 前五名出现概率 | 第一名出现概率 | 倒五出现概率 |
| ------------------------ | ------------ | -------------- | -------------- | ------ | -------------- | -------------- | ------------ |
| ColorJitter              | 4            |                |                | 4      | 100%           | 0%             | 0%           |
| RandomCrop               | 3            | 1              |                | 4      | 75%            | 25%            | 0%           |
| Equalize                 | 2            | 1              | 1              | 4      | 50%            | 0%             | 25%          |
| GaussNoise               | 2            |                |                | 4      | 50%            | 0%             | 0%           |
| HorizontalFlip           | 2            | 1              |                | 4      | 50%            | 25%            | 0%           |
| Rotate                   | 2            | 1              |                | 4      | 50%            | 25%            | 0%           |
| Blur                     | 1            |                |                | 4      | 25%            | 0%             | 0%           |
| CenterCrop               | 1            |                | 1              | 4      | 25%            | 0%             | 25%          |
| ChannelShuffle           | 1            | 1              |                | 4      | 25%            | 25%            | 0%           |
| HueSaturationValue       | 1            |                | 1              | 4      | 25%            | 0%             | 25%          |
| MultiplicativeNoise      | 1            |                | 1              | 4      | 25%            | 0%             | 25%          |
| ChannelDropout           |              |                |                | 4      | 0%             | 0%             | 0%           |
| CLAHE                    |              |                | 2              | 4      | 0%             | 0%             | 50%          |
| CoarseDropout            |              |                |                | 4      | 0%             | 0%             | 0%           |
| Downscale                |              |                | 3              | 4      | 0%             | 0%             | 75%          |
| ElasticTransform         |              |                | 2              | 4      | 0%             | 0%             | 50%          |
| FancyPCA                 |              |                | 1              | 4      | 0%             | 0%             | 25%          |
| FromFloat                |              |                | 2              | 4      | 0%             | 0%             | 50%          |
| GaussianBlur             |              |                |                | 4      | 0%             | 0%             | 0%           |
| GlassBlur                |              |                | 2              | 4      | 0%             | 0%             | 50%          |
| GridDistortion           |              |                | 1              | 4      | 0%             | 0%             | 25%          |
| RandomBrightnessContrast |              |                | 1              | 4      | 0%             | 0%             | 25%          |
| RandomFlip               |              |                |                | 3      | 0%             | 0%             | 0%           |
| RandomGridShuffle        |              |                |                | 4      | 0%             | 0%             | 0%           |
| 无                       |              |                | 2              | 4      | 0%             | 0%             | 50%          |
| 总计                     | 20           | 5              | 20             | 100    |                |                |              |