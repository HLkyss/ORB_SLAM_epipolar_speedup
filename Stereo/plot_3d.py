import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 假设这是三组实验数据
# 数据格式：[单目区大小, 轨迹误差]
data_fov1 = np.array([[10, 0.2], [20, 0.15], [30, 0.1], [40, 0.05]])
data_fov2 = np.array([[10, 0.25], [20, 0.2], [30, 0.15], [40, 0.1]])
data_fov3 = np.array([[10, 0.3], [20, 0.25], [30, 0.2], [40, 0.15]])

# 创建一个三维图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 设置不同的FOV值
fovs = [1, 2, 3]
data = [data_fov1, data_fov2, data_fov3]

# 绘制每组数据
for i, d in enumerate(data):
    monovision = d[:, 0]
    error = d[:, 1]
    fov = np.full_like(monovision, fovs[i])
    ax.plot(monovision, fov, error, label=f'FOV {fovs[i]}')

# 添加标签和标题
ax.set_xlabel('Monovision Size')
ax.set_ylabel('FOV')
ax.set_zlabel('Trajectory Error')
ax.set_title('Trajectory Error vs Monovision Size for Different FOVs')
ax.legend()

# 显示图形
plt.show()

