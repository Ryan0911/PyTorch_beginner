# Tensor

1. tensor_initialization.py: 說明四種的 tensor 初始化方法
   > - 直接生成 tensor
   > - 通過 numpy 轉 tensor
   > - 通過已有的 tensor 來生成新的 tensor
   > - 指定數據維度生成 tensor
2. tensor_attributes.py: 查詢維度與類型
   > - 了解如何查詢 tensor 的維度、數據類型和儲存的設備(CPU/GPU)
3. tensor_operation.py: tensor 的運算方式
   > - https://pytorch.org/docs/stable/torch.html 原始文檔有很多運算方式
   > - tensor 的索引與切片
   > - tensor 的拼接
   > - tensor 的乘積和矩陣乘法
   > - 自動賦值運算 (不推薦使用)
4. tensorAndNumpy.py:
   > - tensor 轉 numpy array
   > - numpy array 轉 tensor
   > - 修改值時會互相影響
