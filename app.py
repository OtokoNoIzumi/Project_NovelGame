# pylint: disable=import-error  # Project structure requires dynamic path handling
# pylint: disable=wrong-import-position  # Path setup must come before local imports
"""应用入口文件"""
import os
import sys
import argparse
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

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--share", action="store_true", help="Enable sharing")
    parser.add_argument("--use-local", action="store_true",
                      help="use local configuration files (default: False)")
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    # 在程序启动前添加以下代码
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # 初始化应用管理器
    app_manager = AppManager(current_dir, use_local=args.use_local)

    # 初始化UI管理器
    ui_manager = UIManager(app_manager)

    # 启动应用
    # 直接传递启动参数到launch方法
    ui_manager.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )


if __name__ == "__main__":
    main()
