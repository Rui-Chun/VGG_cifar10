# 模块导入
import numpy as np
import matplotlib.pyplot as plt

# 创建一个8x6大小的figure，并设置每英寸80个像素点
plt.figure(figsize=(10, 6), dpi=80)

# 创建在1x1的位置创建一个subplot
plt.subplot(121)

X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
C, S = np.cos(X), np.sin(X)

# 绘制cosine函数，颜色是蓝色，line宽为1.0，line类型是实线
plt.plot(X, C, color='blue', linewidth=1.0, linestyle='-')
plt.title("AAAAAAA")

# 绘制sine函数，颜色是绿色，line宽为1.0，line类型是实线
plt.subplot(122)
plt.plot(X, S, color='green', linewidth=1.0, linestyle='-')

# 设置x轴和y轴范围
plt.xlim(-4.0, 4.0)
plt.ylim(-1.0, 1.0)

# # 设置x轴下标和y轴下标
# plt.xticks(np.linspace(-4, 4, 9, endpoint=True))
# plt.yticks(np.linspace(-1, 1, 5, endpoint=True))

# 保存图片，每英寸72个点
# 如果此处报错，在当前路径下创建figures文件夹
# savefig("../figures/exercice_2.png", dpi=72)

# 在屏幕上显示结果
plt.show()