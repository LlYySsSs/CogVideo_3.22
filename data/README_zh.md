# 汽船威利号--视频生成数据集

[Read in English](./README.md)

本数据集从迪士尼公司的《汽船威利号》中裁剪了 `69` 个视频。

+ 视频长度为`6`秒。
+ 视频帧率为`30`帧。
+ 视频分辨率为 `640 x 380`。
+ 视频均为黑白视频，非彩色视频。

## 数据集格式

```
.
├── README.md
├── metadata.csv
├── prompt.txt
├── videos
└── videos.txt
```

其中，`prompt.txt` 包含了每个视频的描述文件，均小于 `100` 个英语单词。
在使用`T5`作为 text encoder的视频生成模型（例如 [CogVideoX-5B-I2V](https://huggingface.co/THUDM/CogVideoX-5b-I2V)
和 [CogVideoX-5B](https://huggingface.co/THUDM/CogVideoX-5b) ）可以直接使用。

该模型可以用作图片生成视频和文字生成视频两种任务。

## 开源协议

本数据集采用 [Apache-2.0](LICENSE) 开源。
