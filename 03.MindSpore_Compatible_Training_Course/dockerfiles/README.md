# 本地构建镜像
在具有Docker环境的Linux服务器上按照以下步骤构建镜像。

## 1、文件准备（根据实际需求进行版本修改）
### CANN包下载
以CANN 8.2.RC1为例:
```bash
# 下载链接获取: https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.2.RC1
TOOLKIT_URL="https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.2.RC1/Ascend-cann-toolkit_8.2.RC1_linux-aarch64.run"
KERNEL_URL="https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.2.RC1/Ascend-cann-kernels-910b_8.2.RC1_linux-aarch64.run"
wget  --no-check-certificate --header="Referer: https://www.hiascend.com/"  ${TOOLKIT_URL}
wget  --no-check-certificate --header="Referer: https://www.hiascend.com/"  ${KERNEL_URL}
```

## 2、将下载文件放置本仓dockerfiles同级目录
文件目录如下所示：
```txt
├── dockerfile_unified
├── Ascend-cann-kernels-910b_8.2.RC1_linux-aarch64.run
└── Ascend-cann-toolkit_8.2.RC1_linux-aarch64.run
```

## 3、执行下述命令构建镜像
```shell
docker build -f dockerfile_unified --tag ms2.7.0-cann8.2rc1:v1 .
```

## 4、执行下述命令保存镜像
```shell
docker save -o ms2_7_0_cann8_2rc1_v1.tar ms2.7.0-cann8.2rc1:v1
```

## 5、基于镜像创建容器
```bash
image_name="ms2.7.0-cann8.2rc1:v1"
container_name="testdocker"
workdir="/home/ma-user/shared"
docker run -itd -u root --ipc=host --network=bridge --privileged=true \
--workdir ${workdir} \
--name ${container_name} \
--device=/dev/davinci0 \
--device=/dev/davinci1 \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
--device=/dev/davinci4 \
--device=/dev/davinci5 \
--device=/dev/davinci6 \
--device=/dev/davinci7 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/common \
-v /usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/driver \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /etc/vnpu.cfg:/etc/vnpu.cfg \
-v /home/aicc:/home/aicc \
-v /home/shared:/home/ma-user/shared \ 
-p 1023:22 \
-p 8007:8000 \
$image_name /bin/bash
# default port: SSH 22 vLLM 8000 Gradio 7860 ollama 8080
# 端口号需要根据当前端口占用情况修改下 sudo lsof -i 1023 或者 sudo netstat -tulnp | grep ':1023'
```

## 6、进入容器
```bash
docker exec -it ${container_name} /bin/bash
```

# 在华为云ModelArts平台使用镜像
若需要在ModelArts平台使用上述构建的镜像, 可先将镜像上传至**容器镜像服务(SWR)**, 再在**ModelArts平台注**册该镜像.

## 上传镜像到华为云
登录华为云[容器镜像服务SWR](https://console.huaweicloud.com/swr/?locale=zh-cn&agencyId=739a0f3116764c61adeb9e2d8fc8557f&region=cn-southwest-2#/swr/dashboard)平台, 在左侧**总览**页面, 按照页面指引:
1. 创建组织
2. 创建镜像(客户端上传)

这里以mindspore-courses组织为例, 将镜像上传至西南区镜像仓
```bash
docker login -u cn-southwest-2@xxx -p "your_password" swr.cn-southwest-2.myhuaweicloud.com # 从页面获取登录命令
sudo docker tag ms2.7.0-cann8.2rc1:v1 swr.cn-southwest-2.myhuaweicloud.com/mindspore-courses/ms2.7.0-cann8.2rc1:v1
sudo docker push swr.cn-southwest-2.myhuaweicloud.com/mindspore-courses/ms2.7.0-cann8.2rc1:v1
```
## 在ModelArts平台注册镜像

进入[ModelArts镜像管理](https://console.huaweicloud.com/modelarts/?locale=zh-cn&agencyId=739a0f3116764c61adeb9e2d8fc8557f&region=cn-southwest-2#/images)页面, 点击右上角**注册镜像**, 按以下格式填写:
1. 镜像源信息: swr.cn-southwest-2.myhuaweicloud.com/mindspore-courses/ms2.7.0-cann8.2rc1:v1 (此镜像源仅做示范), 
2. 架构: ARM
3. 类型: ASCNED
4. 规格: ASCEND_SNT9B

注册完成后, 在创建Notebook实例时, 选择**自定义镜像下**的该镜像即可.

## 注意事项
每个区域都需要单独上传镜像至容器镜像服务SWR, 并在对应区域的ModelArts平台注册该镜像.


