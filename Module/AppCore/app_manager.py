"""应用管理器模块,负责管理应用状态和初始化"""
from typing import Dict, Any, List
import os
import sys
from google import genai
import gradio as gr

from Module.Components.config import Settings, get_file_path
from Module.Components.state_manager import StateManager
from Module.AppCore.response_processor import ResponseProcessor


class AppManager:
    """应用管理器类,管理应用状态和初始化"""

    def __init__(self, current_dir: str = None, use_local: bool = True, model_name: str = "gemini-2.0-flash-001"):
        """初始化应用管理器

        Args:
            current_dir: 当前目录路径,如果为None则自动检测
            use_local: 是否优先使用本地配置文件
        """
        # 初始化目录
        self.current_dir = self._init_directories(current_dir)

        # 加载配置
        self.settings = Settings(current_dir=self.current_dir, use_local=use_local)

        # 初始化LLM客户端
        self.model_name = model_name
        self.llm_client = genai.Client(api_key=self.settings.api_key)

        # 初始化状态管理器
        self.state_manager = StateManager(self._create_initial_state(), self.settings.config)

        # 初始化响应处理器
        self.response_processor = ResponseProcessor(
            state_manager=self.state_manager,
            llm_client=self.llm_client,
            model_name=self.model_name,
            settings=self.settings
        )

    def _init_directories(self, current_dir: str = None) -> str:
        """初始化目录配置"""
        if current_dir is None:
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
        return current_dir

    def _create_initial_state(self) -> Dict:
        """创建初始游戏状态"""
        initial_state = {
            "gr_version": gr.__version__,
            "story_chapter": "起因",
            "story_chapter_stage": 1,
            "inventory": {},
            "character_state": {}
        }

        # 深度拷贝确保完全隔离
        for attr in self.settings.config["state_attributes"]:
            initial_state["character_state"][attr] = {
                "state": "",  # 主状态
            }
            # 添加额外属性
            for extra_attr in self.settings.config.get("extra_state_attributes", []):
                initial_state["character_state"][attr][extra_attr] = -1

        return initial_state

    def create_initial_chat_data(self) -> Dict:
        """创建初始聊天数据"""
        return {
            "current_id": 0,
            "history": []
        }

    def update_state(self, state: Dict, chat_data: Dict) -> List[Any]:
        """更新状态"""
        # 更新状态管理器中的状态
        self.state_manager.state = state
        return ([state, chat_data["history"]]
                if self.settings.config["show_chat_history"]
                else state)

    def clear_state(self, state: Dict, chat_data: Dict) -> List[Any]:
        """清除状态"""
        chat_data['current_id'] = 0
        chat_data['history'] = []
        state.clear()
        state.update(self._create_initial_state())
        # 更新状态管理器中的状态
        self.state_manager.state = state
        self.state_manager.state_history = self._create_initial_state()
        return self.update_state(state, chat_data)

    def undo_state(self, state: Dict, chat_data: Dict) -> List[Any]:
        """撤销状态"""
        if chat_data["current_id"] > 0:
            chat_data["history"] = [
                msg for msg in chat_data["history"]
                if msg["idx"] != chat_data["current_id"]
            ]
            chat_data["current_id"] -= 1

        # 重置状态到上一个状态
        self.state_manager.reset_state()
        # 更新状态管理器中的状态
        state = self.state_manager.state
        return self.update_state(state, chat_data)

    def get_launch_kwargs(self, launch_kwargs: dict = None) -> Dict[str, Any]:
        """获取增强后的启动参数"""
        base_kwargs = {
            "server_name": "0.0.0.0",
            "server_port": 7860,
            "share": False,
            "debug": True
        }

        # 合并传入参数
        if launch_kwargs:
            base_kwargs.update(launch_kwargs)

        # SSL处理逻辑保持不变
        if not base_kwargs.get("share"):
            ssl_cert = get_file_path("localhost+1.pem", current_dir=self.current_dir)
            ssl_key = get_file_path("localhost+1-key.pem", current_dir=self.current_dir)
            if os.path.exists(ssl_cert) and os.path.exists(ssl_key):
                base_kwargs.update({
                    "ssl_certfile": ssl_cert,
                    "ssl_keyfile": ssl_key
                })
            else:
                print("SSL证书已禁用（Share模式使用ngrok的自动HTTPS）")

        return base_kwargs
