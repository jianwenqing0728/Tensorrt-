# Cuda CuDnn Tensorrt OpenCV  Yolo C++ Python模型部署加速

## 项目

- **使用Tensorrt对Yolo系列(其他模型同理)进行部署C++、Python和C#等工程项目**
- **本仓库提供使用到的全部代码和配置环境**

## 环境说明（Windows）

- [Cuda 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)

- [CuDnn 9.1.1.17](https://developer.nvidia.com/cudnn-downloads?target_os=Windows&target_arch=x86_64&target_version=Agnostic&cuda_version=11)

- [Tensorrt](https://developer.nvidia.com/tensorrt/download) 8.6.1.6  

- [OpenCV](https://github.com/opencv/opencv/tags) 4.9.0

- [Cmake 3.29](https://cmake.org/files/)

- [Yolov 1](https://github.com/ultralytics/ultralytics)-9

- MicroSoft Visual [Studio](https://visualstudio.microsoft.com/zh-hans/downloads/) 2017/2019/2022

  

## 环境配置vs

### Cuda安装

Win+R，输入nvidia-smi，查看适配当前NVIDIA的最高CUDA版本

<img src="F:\Machine-Learning\Deep-Learning\Tensorrt部署软件包\nvidia-smi.png" alt="nvidia-smi"  />

进入[Cuda 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)，根据查到的CUDA版本下载低于这个版本的CUDA安装包（不能高于查到的最高版本），Windows,x86_64,11,exe(local)。安装之前请安装好MicroSoft Visual Studio2017/2019/2022，根据提示选择自定义安装。

### CuDnn安装

进入[CuDnn 9.1.1.17](https://developer.nvidia.com/cudnn-downloads?target_os=Windows&target_arch=x86_64&target_version=Agnostic&cuda_version=11)，根据安装的Cuda版本适配CuDnn（很重要）,windows,x86_64,Tarball,11,生成wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-9.1.1.17_cuda11-archive.zip

下载并解压好CuDnn压缩包，将CuDnn的bin、include、lib，3个文件夹复制到CUDA的安装路径下（默认安装路径）C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8

### Tensorrt安装

进入[Tensorrt](https://developer.nvidia.com/tensorrt/download) 8.6.1.6  ，根据安装的Cuda版本适配Tensorrt（很重要），下载并解压好，将TensorRT的bin、include、lib，3个文件夹复制到CUDA的安装路径下（默认安装路径）C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8

### Yolo部署

#### 方法一：

1. 进入[Yolov 1](https://github.com/ultralytics/ultralytics)-9，`pip install ultralytics`下载好yolo项目，阅读[ultralytics官方文档](https://docs.ultralytics.com/)，`yolo mode=export model=yolov8n.pt format=onnx dynamic=False`生成ONNX文件
2. 打开仓库TensoRT-Yolo-trt-Qing仓库文件夹，进入CMD，执行`python v8_transform.py yolov8n.onnx`，生成yolov8n.transd.onnx
3. 打开Tensorrt文件夹，进入bin文件，将yolov8n.transd.onnx文件复制到此路径，进入CMD，执行`trtexec --onnx=yolov8n.transd.onnx --saveEngine=yolov8n_fp16.trt --fp16`，生成yolov8n_fp16.trt。
4. 将yolov8n_fp16.trt文件复制到仓库TensoRT-Yolo-Qing中，在仓库根目录新建build文件，修改CMakeLists.txt文件（很重要），修改下面3句
5. `set(OpenCV_DIR "D:\\OpenCV4.9.0\\opencv\\build\\x64\\vc16\\lib")` ，将此路径修改为opencv安装路径，注意：指定到**opencv\\build\\x64\\vc16\\lib**（切记）,若指定到build，CMake会因为OpenCV版本问题报错， `Found OpenCV Windows Pack but it has no binaries compatible with your configuration`，无法匹配build下的配置文件。
6. `set(TRT_DIR "D:\\TensorRT-8.6.1.6")`  ，将此路径修改为TensorRT路径
7. `set(CMAKE_CUDA_ARCHITECTURES 86)`，根据GPU算力修改，具体参数查询ChatGPT。此处一定要注意，否则可能会由于cuda版本较高，cmake报错
8. 在buid路径下进行cmd执行cmake CMakeList.txt，生成vs项目文件，用vs打开项目，点yolov8右键生成。查看Debug文件夹下，将生成exe文件，将trt文件和图片放在Debug文件夹，执行exe文件即可测试结果
9. 在vs中设置项目输出为dll即可生成dll，提供其他程序调用

#### 方法二：

python gen_wts.py通过pt文件转wts文件，wts生成engine文件，具体参考https://github.com/xiaocao-tian/yolov8_tensorrt，https://zhuanlan.zhihu.com/p/628984182，可能遇到的问题，和解决方法同方法一和常见问题，和方法一的使用区别不大，选一个学习即可。

## 常见问题

cmake失败找不到opencv或`Found OpenCV Windows Pack but it has no binaries compatible with your configuration`，请检查opencv环境变量Path  `D:\OpenCV4.9.0\opencv\build\x64\vc16\bin`，`D:\OpenCV4.9.0\opencv\build\x64\vc16\lib，D:\OpenCV4.9.0\opencv\build`。若find_package(OpenCV REQUIRED)由于版本问题失败，尝试指定set指定opencv路径.`no binaries compatible with your configuration`是因为cmake无法处理opencv/build/OpenCVConfig.cmake文件，OpenCVConfig.cmake是用于路径指定。因此在环境变量中需要指定到更具体的路径。或者在CMakeList.txt中指定到更加具体的bin、lib路径。

cmake失败找不到cuda，检查cuda安装是否成功，cuda环境变量是否正确。cmd执行nvcc-version，若输出cuda版本信息，则安装正确，环境变量CUDA_PATH：`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8`，CUDA_PATH_V11_8：`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8`

cmake失败找不到tensorrt，检查tensorrt版本是否适配

cmake失败，CMakeCUDACompilerId.cu执行失败，在cmake过程中过测试一个cu文件能否执行成功，这是由于vs版本过高，cuda未跟上，例如cuda11未跟上vs2022,因此有3个方法：1、降低vs版本到vs2017\2019，2、提高cuda版本到cuda12，3、进入CUDA安装路径修改cuda的host_config.h文件（关键），*`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include\crt\host_config.h`，MSC_VER是vs版本码，将版本检查MSC_VER数值改大，如下代码*

*`#if _MSC_VER < 1910 || _MSC_VER >= 2300*`

`*#error -- unsupported Microsoft Visual Studio version! Only the versions between 2017 and 2022 (inclusive) are supported! The nvcc flag '-allow-unsupported-compiler' can be used to override this version check; however, using an unsupported host compiler may cause compilation failure or incorrect run time execution. Use at your own risk.*`

`*#elif _MSC_VER >= 1910 && _MSC_VER < 2300*`

`*#pragma message("support for this version of Microsoft Visual Studio has been deprecated! Only the versions between 2017 and 2022 (inclusive) are supported!")`*

## 参考资料

https://github.com/Monday-Leo/Yolov5_Tensorrt_Win10

https://github.com/xiaocao-tian/yolov8_tensorrt

https://github.com/wang-xinyu/tensorrtx

cuda\cudnn\tensorrt配置避坑指南：https://blog.csdn.net/weixin_43249548/article/details/133563043

cuda\cudnn安装：https://blog.csdn.net/jhsignal/article/details/111401628