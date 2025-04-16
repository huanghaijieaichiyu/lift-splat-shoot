"""检查NuScenes Box类的实现"""
import inspect
from nuscenes.utils.data_classes import Box

# 将结果写入文件
with open("box_check_results.txt", "w") as f:
    # 打印Box类的源码
    f.write(inspect.getsource(Box.__init__) + "\n")
    f.write("\nBox类velocity参数的默认值:\n")
    f.write(str(inspect.signature(Box.__init__)) + "\n")

    # 尝试创建Box对象
    try:
        box1 = Box([0, 0, 0], [1, 1, 1], None)
        f.write("\n不带velocity创建成功\n")
    except Exception as e:
        f.write(f"\n不带velocity创建失败: {e}\n")

    try:
        box2 = Box([0, 0, 0], [1, 1, 1], None, velocity=None)
        f.write("velocity=None创建成功\n")
    except Exception as e:
        f.write(f"velocity=None创建失败: {e}\n")

    try:
        box3 = Box([0, 0, 0], [1, 1, 1], None, velocity=[0, 0, 0])
        f.write("velocity=list创建成功\n")
    except Exception as e:
        f.write(f"velocity=list创建失败: {e}\n")

    try:
        box4 = Box([0, 0, 0], [1, 1, 1], None, velocity=(0, 0, 0))
        f.write("velocity=tuple创建成功\n")
    except Exception as e:
        f.write(f"velocity=tuple创建失败: {e}\n")
