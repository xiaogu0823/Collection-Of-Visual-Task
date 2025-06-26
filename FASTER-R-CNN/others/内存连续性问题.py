import torch

# 创建一个张量
x = torch.randn(2, 3)

# 转置张量，使其变得非内存连续
y = x.t()

# 尝试直接对 y 进行 reshape 操作，这通常会失败
# try:
#     z = y.reshape(6)  # 这会抛出 RuntimeError
# except RuntimeError as e:
#     print(e)  # 输出错误信息，指出需要内存连续的张量

# 正确的做法是先调用 .contiguous()
y_contiguous = y.contiguous()
z = y_contiguous.reshape(6)  # 现在这可以成功执行
