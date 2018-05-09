# 数据挖掘第二次作业

## 实践方法
- 基础的LFM
- 考虑偏置量的LFM
- 考虑评论文本和偏置量的LFM

## 语言
- Python
- C++

## 提交文件目录结构
|文件|说明|
|--|-----|
|`./data`|预测结果数据文件，内只含`test.dat`|
|`./LFM-CPP`|以C++语言实现的考虑偏置量的LFM模型|
|`LFM.py`|考虑偏置量的LFM模型python版本 |
|`LFM_with_review.py`| 考虑评论文本和偏置量的LFM模型python版本 |
|`code_and_report.ipynb`|jupyter notebook文件，详尽记录了实验的全过程，不同模型的代码编写，以及数据预处理等 |
|`Report.pdf`|作业报告|
|`README`|this|


## 关于C++代码的运行
`./LFM-CPP`文件中已包含了makefile文件，故而可以直接make，生成最后可执行文件`LFM`。
执行时，按照如下格式：
>`$ LFM -n iteration_times -a alpha > redirection_file`

## 关于报告
此篇pdf形式的报告，直接由jupyter notebook导出，故而其中包含大量代码与部分运行结果。关于实验结果和实验心得的记录，仅在文档的最后一两页。为节省老师的阅读时间，前面的部分可以直接忽略。关于实践过程，已在上面大致描述。非常抱歉迟交了此次作业，请老师海涵。
