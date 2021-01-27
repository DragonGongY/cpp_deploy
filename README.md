## 介绍
采用paddle的c++部署分割网络模型，并采用mkl在cpu端进行加速
![demo image](output/test.png)

### 使用方法

- **编译**

  修改scripts下的build.sh,如果有gpu测打开gpu，没有则使用mkl进行推理加速。
  
  输入:sh ./scripts/build.sh

- **预测**

  输入：./build/demo/segmentor --model_dir=/path/to/model/ --image=/path/to/image
  
       ./build/demo/segmentor --model_dir=/path/to/model/ --image_list=/path/to/images/path
      
- **数据保存**

  默认会将所有图片预测的结果保存到根目录下的output文件夹下