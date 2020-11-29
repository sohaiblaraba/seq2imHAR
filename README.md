# Human Action Recognition using Sequence to Image (Seq2Im) sequence transformation
A repository related to the papers:
* Laraba, S., Brahimi, M., Tilmanne, J., & Dutoit, T. (2017). 3D skeleton‐based action recognition by representing motion capture sequences as 2D‐RGB images. Computer Animation and Virtual Worlds, 28(3-4), e1782.[[1](https://onlinelibrary.wiley.com/doi/abs/10.1002/cav.1782)]
* Laraba, S., Tilmanne, J., & Dutoit, T. (2019, September). Leveraging Pre-trained CNN Models for Skeleton-Based Action Recognition. In International Conference on Computer Vision Systems (pp. 612-626). Springer, Cham. [[2](https://link.springer.com/chapter/10.1007/978-3-030-34995-0_56)]  

<center>
<p>
  <img src="https://github.com/sohaiblaraba/seq2imHAR/blob/main/data/examples/image36.gif" width="300" />
  <img src="https://github.com/sohaiblaraba/seq2imHAR/blob/main/data/examples/image37.jpg" width="256" /> 
</p>
</center>

## Requirements
* pytorch >= 1.1 (models to be updated to fit current version)
* itertools
* functools



## Usage
### Transforming sequences into images
In the case of one file, do the following:
```
$ python seq2im.py --file path/to/file.txt --output path/to/output.jpg
```
In the case of a folder containing a list of sequences:
```
$ python seq2im.py --folder path/to/folder --output path/to/output_folder
```
#### Accepted formats:
* '.txt': a text file containing x, y and z coordinates of all joints in one line per frame, separated with spaces. Example: 21 joints x 5 frames
```
212.8507 807.9225 3581.9950 245.7799 687.7249 3633.8970 242.5592 624.7278 3656.0550 232.3292 432.6636 3714.7670 218.1153 172.5972 3785.5470 344.9145 574.0387 3588.6550 421.1396 384.7547 3627.7780 191.8999 318.9077 3540.7560 129.8675 317.0390 3542.5770 110.9304 611.1551 3716.9620 100.8750 428.6715 3742.5930 118.6992 342.8923 3556.5420 116.5352 317.3672 3550.7410 269.9234 167.8491 3730.0050 310.5748 -146.2299 3818.3830 342.8889 -433.2140 3943.2580 295.1482 -501.7325 3968.8960 162.7660 174.5395 3778.4710 197.5115 -136.0554 3904.3810 244.2706 -420.1995 4054.1470 204.0377 -476.3960 4094.2940
214.8817 808.6480 3590.7340 247.2661 687.8656 3638.0780 243.9015 625.0034 3660.1670 233.2372 433.3165 3718.6870 218.4443 173.7018 3789.2340 345.5990 574.3399 3592.3590 420.6736 385.2528 3629.2460 183.3682 333.4976 3557.2930 128.8160 320.2999 3564.1630 114.5627 615.1996 3726.2740 104.4773 432.6954 3751.8380 121.1366 346.3704 3565.7290 119.1443 319.6799 3569.8230 269.9170 168.7128 3732.6680 308.4871 -140.9414 3820.3770 345.3195 -426.0727 3942.2710 303.6116 -478.2606 3968.7930 163.4845 175.8385 3783.0830 196.1725 -133.1571 3907.9890 242.3866 -419.1891 4061.9860 193.8665 -476.8415 3967.5190
214.8725 808.6450 3590.7220 247.1501 688.5785 3638.3300 243.7884 625.6233 3660.5080 233.1048 433.7439 3719.3510 218.2214 173.8454 3790.6350 345.0240 573.9973 3592.5930 420.9870 385.7001 3629.5590 184.3068 332.2636 3556.6400 130.6769 320.3038 3554.1300 114.5870 615.2725 3726.7620 104.4976 432.7629 3752.3020 121.0331 346.4065 3566.3190 115.5924 315.5987 3561.4960 269.4088 168.8617 3734.1450 308.6371 -144.8276 3823.3870 342.4788 -430.0128 3954.4830 305.1564 -487.5182 3971.7720 163.5039 175.9672 3784.3300 196.4338 -133.3533 3909.3210 242.7123 -419.3571 4064.7090 194.0295 -477.0178 3970.3650
214.4592 807.7186 3586.4120 246.7303 688.3090 3636.3030 243.4830 625.3507 3658.5660 233.1042 433.4322 3717.5820 218.6054 173.4429 3788.8450 344.6368 574.1953 3591.2380 420.9057 385.4574 3628.9440 189.6139 324.7738 3543.8570 127.4719 319.1971 3541.4200 114.0878 615.4307 3724.2010 103.9814 432.9014 3749.6650 120.3022 346.4750 3563.8700 111.8385 313.6773 3548.1460 269.7302 168.7230 3732.9350 308.8691 -146.3469 3823.1790 342.5348 -431.5508 3952.7950 298.2559 -500.0779 3996.8490 163.9304 175.1923 3781.6980 196.7638 -134.6997 3909.8000 242.5556 -420.7411 4062.1390 232.5329 -481.8147 4126.2120
```

* '.skeleton': skeleton sequences from the [NTU RGB+D dataset](https://github.com/shahroudy/NTURGB-D).
* 'v3d'
* You can add custom formats yourself.

### Training 
```
$ python train.py --data path/to/dataset --mode retrain_deep --epoch 1 --batch 15 --save models/
```
* --data: path of your dataset. The dataset has contain two folders 'train' and 'val', where each folder cantains a list of sub-folders containing the image files. Each sub-folder corresponds to a class. See example bellow.  
```
dataset __ train __ A001  
       |        |  |__ file_a001_001.jpg  
       |        |  |__ file_a001_002.jpg  
       |        |  ...  
       |        |  |__ file_a001_999.jpg  
       |        |__ A002  
       |        |  |__ file_a002_001.jpg  
       |        |  |__ file_a002_002.jpg  
       |        |  ...  
       |        |  |__ file_a002_999.jpg  
       |        |..  
       |        |__ A999  
       |           |__ file_a999_001.jpg  
       |           |__ file_a999_002.jpg  
       |           ...  
       |           |__ file_a999_999.jpg  
       |___ val __ A001  
                |  |__ file_a001_001.jpg  
                |  |__ file_a001_002.jpg  
                |  ...  
                |  |__ file_a001_999.jpg  
                |__ A002  
                |  |__ file_a002_001.jpg  
                |  |__ file_a002_002.jpg  
                |  ...  
                |  |__ file_a002_999.jpg  
                |..  
                |__ A999  
                   |__ file_a999_001.jpg  
                   |__ file_a999_002.jpg  
                   ...  
                   |__ file_a999_999.jpg  
```

* --mode: selecting the training protocol:
  * retrain_deep: for transfer learning using a deep retraining 
  * retrain_shallow: for a transfer learning using a shallow retraining
  * scratch: for training from scratch
* --epoch: number of epochs
* --batch: batch size
* --save: path to the folder where the models and stats will be saved.

## Testing the trained model
Coming soon
