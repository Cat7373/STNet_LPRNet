# STNet + LPRNet
STNet 进行空间校正 + LPRNet 进行车牌识别

目前在 CCPD 数据集下，各类作为一个整体，以 half 模式运行测试，最终 macc 约为 93.5%

## 使用
1. 下载并安装 [Conda](https://www.anaconda.com/products/individual)
2. 运行下列命令安装依赖

    ```sh
    conda create --name py38 python=3.8
    conda activate py38  # 建议加到 .bashrc 中，不然每次都要先执行

    conda install -c pytorch pytorch
    conda install -c conda-forge accimage opencv onnx
    conda install tqdm tensorboard numpy
    conda install -c anaconda flask
    pip install thop
    ```
3. 下载 [CCPD](https://github.com/detectRecog/CCPD) 数据集，放到任意位置
4. 运行下列命令进行训练

    ```sh
    python train.py --source-dir /data/CCPD2019
    ```
5. 运行下列命令进行测试

    ```sh
    python test.py --source-dir /data/CCPD2019 --weights weights/final.pt
    ```

    * 如果之前训练时中断了，使用 --weights 加载之前的模型继续训练
    * 如果内存充足，建议启用 --cache-images 参数来加快训练，这大约需要 5.5GiB 的内存
    * 过小的 batch-size 会引起不稳定，梯度震荡比较严重
    * 过大的 batch-size 会引起收敛速度慢，需要更多的 epochs 才能得到同样的精度
    * 不建议手动指定 --workers
    * 由于现在 ST 和 LPR 是一起训练的，所以多少还有些玄学的东西，如果到 50 epochs，macc 还没上 0.8，就干脆弃了重炼一炉吧
6. 运行下列命令进行运行下列命令提供 Web API

    ```sh
    python web.py --weights weights/final.pt
    ```
7. 剪枝

    ```sh
    python prune.py --source-dir /data/CCPD2019 --weights weights/final.pt
    ```

    * 效果极差，不建议使用
7. ONNX

    ```sh
    python export.py --weights weights/final.pt
    ```

    * 目前的 pytorch 无法正确的导出 STNet，因此实际上它是不可用的

## Web API
### 识别图片中的车牌
> 请先将车牌部分切割出来再提交，否则基本无法识别

* POST /api/ml/st_lpr
* 请求
    * img 被识别的图片，采用常规 upload 文件的方式上传
    * rect 可选，用逗号分割的坐标点列表
        * 设 r1 为区域 1，r2 为区域 2，依此类推
        * r1x1,r1y1,r1x2,r1y2,r2x1,r2y1,r2x2,r2y2,r3x1,r3y1,r3x2,r3y2...
        * 会通过这组坐标从图片中切割出车牌部分
* 响应
    * 识别到的车牌内容列表

## 目录结构
```
/data 数据加载器
|-- ccpd.py 用于遍历 source-dir 目录中的 CCPD 数据集，提供基本的数据解析
|-- dataset.py 基于 CCPD 数据集制作的 Pytorch Dataset 实现
/model 模型
|-- lprnet.py LPRNet 实现
|-- st.py STNet 实现
/runs
|-- /api WebAPI 的日志文件
|-- /cache 缓存文件
|-- /exp* 训练时的数据输出目录，每次启动会自动自增 + 1
    |-- weights 输出的模型文件
        |-- last.pt 目前最新一次完成的 epoch 保存的数据
        |-- best.pt 训练过程中，测试得到最高 acc 的模型
        |-- final.pt 跑完所有 epoch 后保存的最终模型文件，注意，不应使用此文件继续训练，请使用 last.pt 继续训练
/weights
|-- final.pt 已训练好的模型，方便快速开始测试、产品部署，但不应用其继续训练
```

## Links
* [原仓库](https://github.com/xuexingyu24/License_Plate_Detection_Pytorch)
    * 模型和原始代码来自这里
* [YOLO5](https://github.com/ultralytics/yolov5)
    * 代码中大部分改动都照抄的这边
* [Pytorch Tutorials (ST)](https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html)
    * 注意，此文写于 pytorch 1.3 发布之前
* [Pytorch Tutorials (Flask)](https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html)
* [Pytorch Tutorials (Prune)](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)

## TODO
* [ ] 分别训练 STNet 和 LPRNet
* [ ] 双行车牌支持
