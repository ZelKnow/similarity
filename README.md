# 文本相似度计算

本项目实现了[向量空间模型](https://en.wikipedia.org/wiki/Vector_space_model)，用以求文档间的相似度。并提供了多进程方法以提高处理速度。

使用环境：```python3.8+```

## 项目结构

```
similarity             //
|-.git                 //
|-.gitignore           //
|-LICENSE              //
|-README.md            //
|-data                 //
| |-199801_clear.txt   // 源数据
| |-stopwords.txt      // 停用词
|-poetry.lock          //
|-pyproject.toml       //
|-util                 //
| |-__init__.py        //
| |-dataloader.py      // 读取数据辅助程序
|-vsm.py               // 向量空间模型主程序

```