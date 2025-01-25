# pylint: disable=import-error  # Project structure requires dynamic path handling
# pylint: disable=wrong-import-position  # Path setup must come before local imports
"""应用入口文件"""
import os
import sys
import asyncio

# ===== 2. 初始化配置 =====
# 获取当前文件所在目录的绝对路径
if "__file__" in globals():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.normpath(os.path.join(current_dir, ".."))
else:
    # 在 Jupyter Notebook 环境中
    current_dir = os.getcwd()
    current_dir = os.path.join(current_dir, "..")
    root_dir = os.path.normpath(os.path.join(current_dir))

current_dir = os.path.normpath(current_dir)
sys.path.append(current_dir)

from Module.AppCore.app_manager import AppManager
from Module.AppCore.ui_manager import UIManager

def main():
    """主函数"""
    # 在程序启动前添加以下代码
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # 初始化应用管理器
    app_manager = AppManager(current_dir)

    # 初始化UI管理器
    ui_manager = UIManager(app_manager)

    # 启动应用
    ui_manager.launch()


if __name__ == "__main__":
    main()
