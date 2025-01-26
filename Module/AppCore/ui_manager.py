"""UI管理器模块,负责管理Gradio界面"""
from typing import Dict, Any, List, Generator
import gradio as gr
from Module.AppCore.app_manager import AppManager


class UIManager:
    """UI管理器类,管理Gradio界面"""

    def __init__(self, app_manager: AppManager):
        """初始化UI管理器

        Args:
            app_manager: 应用管理器实例
        """
        self.app_manager = app_manager
        self.demo = None

    def create_interface(self) -> gr.Blocks:
        """创建Gradio界面"""
        with gr.Blocks(theme="soft") as demo:
            # 创建会话级状态
            session_state = gr.State(self.app_manager._create_initial_state())
            session_chat_data = gr.State(self.app_manager.create_initial_chat_data())

            # 1. 创建界面组件
            chatbot = gr.ChatInterface(
                self.respond,
                title=self.app_manager.settings.config["title"],
                type="messages",
                chatbot=gr.Chatbot(
                    placeholder="输入 【开始】 开始进行创作",
                    height="80vh",
                    show_share_button=True,
                    editable="user",
                    show_copy_all_button=True,
                    type="messages",
                ),
                additional_inputs=[
                    gr.Checkbox(value=True, label="Use system message"),
                    gr.Checkbox(value=True, label="Add Extra message"),
                    gr.Checkbox(value=True, label="Auto analysis state"),
                    gr.Textbox(
                        value=self.app_manager.settings.config.get("explored_jobs", {}).get("default", ""),
                        label="ignore job"
                    ),
                    session_state,  # 添加会话状态作为隐藏输入
                    session_chat_data,  # 添加聊天数据作为隐藏输入
                ],
            )

            # 2. 创建状态显示组件
            outputs = []
            with gr.Accordion("查看故事状态", open=False):
                state_output = gr.JSON(value=self.app_manager._create_initial_state())
                outputs.append(state_output)

            if self.app_manager.settings.config["show_chat_history"]:
                with gr.Accordion("查看历史对话", open=False):
                    history_output = gr.JSON(value=[])
                    outputs.append(history_output)

            # 3. 绑定事件处理
            chatbot.chatbot.change(
                fn=self.update_state,
                inputs=[session_state, session_chat_data],
                outputs=[*outputs, session_state, session_chat_data]
            )
            chatbot.chatbot.clear(
                fn=self.clear_state,
                inputs=[session_state, session_chat_data],
                outputs=[*outputs, session_state, session_chat_data]
            )
            chatbot.chatbot.undo(
                fn=self.undo_state,
                inputs=[session_state, session_chat_data],
                outputs=[*outputs, session_state, session_chat_data]
            )
            chatbot.chatbot.retry(
                fn=self.undo_state,
                inputs=[session_state, session_chat_data],
                outputs=[*outputs, session_state, session_chat_data]
            )

            self.demo = demo
            return demo

    def update_state(self, session_state: Dict, session_chat_data: Dict) -> tuple:
        """更新会话状态"""
        outputs = self.app_manager.update_state(
            state=session_state,
            chat_data=session_chat_data
        )
        return (*outputs, session_state, session_chat_data)

    def clear_state(self, session_state: Dict, session_chat_data: Dict) -> tuple:
        """清除会话状态"""
        outputs = self.app_manager.clear_state(
            state=session_state,
            chat_data=session_chat_data
        )
        return (*outputs, session_state, session_chat_data)

    def undo_state(self, session_state: Dict, session_chat_data: Dict) -> tuple:
        """撤销会话状态"""
        outputs = self.app_manager.undo_state(
            state=session_state,
            chat_data=session_chat_data
        )
        return (*outputs, session_state, session_chat_data)

    def respond(
        self,
        message: str,
        history: List[Any],
        use_system_message: bool,
        add_extra_message: bool,
        auto_analysis_state: bool,
        ignore_job: str,
        session_state: Dict,
        session_chat_data: Dict,
    ) -> Generator[str, None, None]:
        """处理用户输入并生成响应"""

        if self.app_manager.settings.config.get("enable_safe_check", False):
            safe_check = self.app_manager.response_processor.safe_check(message)
            if not safe_check:
                yield "抱歉，内容包含不当内容，无法支持。"
                return

        # 预处理
        message, is_control = self.app_manager.response_processor.pre_process(
            message,
            history,
            add_extra_message,
            ignore_job,
            state=session_state,
            chat_data=session_chat_data
        )

        # 主处理 - 流式输出LLM响应
        final_response = None
        got_content = False
        for chunk in self.app_manager.response_processor.main_process(
            message,
            use_system_message,
            state=session_state,
            chat_data=session_chat_data
        ):
            if not chunk:  # 如果是空响应
                if not got_content:  # 如果还没有收到过内容
                    yield "正在生成回复，请稍候..."
                continue
            got_content = True
            final_response = chunk
            yield chunk

        # 如果整个过程都没有收到内容
        if not got_content:
            yield "抱歉，生成响应失败，请重试。"
            return

        # 如果有响应且需要后处理
        if final_response and not is_control and auto_analysis_state:
            # 后处理 - 分析状态变化并获取完整响应
            final_response = self.app_manager.response_processor.post_process(
                final_response,
                is_control,
                auto_analysis_state,
                state=session_state,
                chat_data=session_chat_data
            )
            yield final_response

    def launch(self, **launch_kwargs) -> None:
        """启动Gradio界面"""
        if self.demo is None:
            self.create_interface()

        # 合并默认参数和传入参数
        final_kwargs = self.app_manager.get_launch_kwargs(launch_kwargs)
        self.demo.launch(**final_kwargs)
