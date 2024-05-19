# Guide-rknn

## Install dependencies on rockchip platform (in my case orange pi5 plus)

Basic pakages:

```
sudo apt-get update
sudo apt-get install -y python3 python3-dev python3-pip gcc
sudo apt-get install -y python3-opencv
sudo apt-get install -y python3-numpy
sudo apt-get install git
sudo apt-get install wget
sudo apt-get install python3-setuptools
```

**FOR 3588**

[version rknn-toolkit] (https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit-lite2/packages)


```
#choose own version

wget https://github.com/rockchip-linux/rknpu2/blob/master/runtime/RK3588/Linux/librknn_api/aarch64/librknnrt.so

sudo mv librknnrt.so /usr/lib/librknnrt.so

# choose correct version rknn-toolkit (look on your python3 version , in my case 3.10)
python3 --version

wget https://github.com/airockchip/rknn-toolkit2/blob/master/rknn-toolkit-lite2/packages/rknn_toolkit_lite2-2.0.0b0-cp310-cp310-linux_aarch64.whl

pip3 install rknn_toolkit_lite2-2.0.0b0-cp310-cp310-linux_aarch64.whl

sudo ln -s librknnrt.so librknn_api.so

```

**To test it:**

Download repsitory

`https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit-lite2/examples/resnet18`

and run `test.py`

## Inference yolo-model

**COMING SOON**



