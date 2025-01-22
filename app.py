# pylint: disable=import-error  # Project structure requires dynamic path handling
# pylint: disable=wrong-import-position  # Path setup must come before local imports
"""
For more information on `huggingface_hub` Inference API support
please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
"""
import os
import sys
import json
from typing import Dict, Any, List, Tuple
import gradio as gr
from google import genai
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

from Module.Components.config import get_file_path, Settings
from Module.Components.state_manager import StateManager
from Module.Common.scripts.llm.gemini_sdk import (
    types,
    get_safety_settings,
    format_content,
    get_content_config
)
from Module.Common.scripts.llm.utils.schema_response import ContentAnalyzer
from Module.Common.scripts.common.debug_utils import log_and_print

# 使用方式
settings = Settings(current_dir=current_dir)

gemini_client = genai.Client(api_key=settings.api_key)

MODEL_NAME = "gemini-2.0-flash-exp"


def gemini_generate_with_schema(
    client: Any,
    input_text: str,
    response_schema: Dict,
    system_prompt: str = ""
) -> Any:
    """使用Gemini生成带schema的响应

    Args:
        client: Gemini客户端实例
        input_text: 输入文本
        response_schema: 响应schema定义
        system_prompt: 系统提示词

    Returns:
        生成的响应内容
    """
    # log_and_print(f"gemini_generate_with_schema: {input_text}")
    return client.models.generate_content(
        model=MODEL_NAME,
        contents=[format_content("user", input_text)],
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=response_schema,
            safety_settings=get_safety_settings(),
        ),
    )


def game_response_formatter(response: Any) -> Dict[str, Any]:
    """格式化游戏响应

    Args:
        response: 原始响应内容

    Returns:
        格式化后的响应字典
    """
    log_and_print("game_response_formatter:\n", response.text)
    updates = json.loads(response.text)

    # Deduplicate character state updates by attribute
    state_updates = updates.get('stateUpdates', [])
    seen_attrs = {}

    for i, update in enumerate(state_updates):
        attr = update['attribute']
        if attr in seen_attrs:

            log_and_print("attribute_update_game_response_formatter:\n", attr)

            extra_state_attribute = settings.config.get("extra_state_attributes", [])[0]
            prev_idx = seen_attrs[attr]
            prev_update = state_updates[prev_idx]

            # Compare extra_state if available
            if extra_state_attribute in update and extra_state_attribute in prev_update:
                if update[extra_state_attribute] > prev_update[extra_state_attribute]:
                    state_updates[prev_idx] = update
            # Otherwise keep the later update
            else:
                state_updates[prev_idx] = update

            state_updates[i] = None
        else:
            seen_attrs[attr] = i

    # Remove None entries
    updates['stateUpdates'] = [u for u in state_updates if u is not None]

    return {
        'inventory': updates.get('itemUpdates', []),
        'character_state': updates.get('stateUpdates', [])
    }


def game_context_formatter(current_data: Dict[str, Any], content: str) -> str:
    """格式化游戏上下文

    Args:
        current_data: 当前游戏数据
        content: 内容文本

    Returns:
        格式化后的上下文字符串
    """
    extra_state_attribute = settings.config.get("extra_state_attributes", [])
    extra_state_hint = ""
    if extra_state_attribute:
        extra_state_hint = (
            f"- 为了阅读顺畅，故事内容里不会明示出现下列属性{extra_state_attribute}的信息，"
            "但请根据故事内容和状态初始值，分析出变化来；"
            f"每个角色状态的变化，都必然伴随着{extra_state_attribute}的变化，在这个故事里，变化值都是在初始值的基础上增加"
        )
    formatted_content = f"""
请根据最近故事内容分析物品清单、角色状态的变化。

这是一些可参考的规则：
- 物品就用名称表述，如果描述太长，就概述到名称不超过10个汉字，每件东西都单独列成一条，除非是套装，
- 基本上角色已经获得过的道具不会重复获得，除非特别说明又继续增加或更多了。
- 角色状态只会包含下面状态列表里的项目。
- 除了数据结构中的字段，其他都用中文回复。
{extra_state_hint}

当前状态
<物品清单>
{current_data.get('inventory', {})}
</物品清单>

<角色状态>
{current_data.get('character_state', {})}
</角色状态>

<最近故事内容>
{content}
</最近故事内容>
"""
    log_and_print("game_context_formatter:\n", formatted_content)
    return formatted_content


# 创建分析器实例
analyzer = ContentAnalyzer(
    llm_client=gemini_client,
    generate_func=gemini_generate_with_schema,
    response_schema=settings.response_schema,
    system_prompt=settings.state_system_prompt,
    context_formatter=game_context_formatter,
    response_formatter=game_response_formatter
)


def get_gradio_version() -> str:
    """获取Gradio版本号

    Returns:
        Gradio版本号字符串
    """
    return gr.__version__


def create_initial_state() -> Dict:
    """创建初始游戏状态

    Returns:
        包含初始状态的字典
    """
    initial_state = {
        "gr_version": get_gradio_version(),
        "story_chapter": "起因",
        "story_chapter_stage": 1,
        "inventory": {},
        "character_state": {}
    }
    # 深度拷贝确保完全隔离
    for attr in settings.config["state_attributes"]:
        initial_state["character_state"][attr] = {
            "state": "",  # 主状态
        }
        # 添加额外属性
        for extra_attr in settings.config.get("extra_state_attributes", []):
            initial_state["character_state"][attr][extra_attr] = 0
    return initial_state


game_state = create_initial_state()

state_manager = StateManager(create_initial_state(), settings.config)

# 区分用于处理的历史记录和用于显示的历史记录
# 使用字典来存储对话历史和当前ID
chat_data = {
    "current_id": 0,
    "history": []  # 列表中存储对话记录字典，每条记录包含role、content、only_for_display和id属性
}


def detect_state_changes(game_state_dict: dict, story_output: str) -> Dict:
    """检测游戏状态变化

    Args:
        game_state_dict: 当前游戏状态字典
        story_output: 故事输出文本

    Returns:
        状态更新信息
    """
    updates = analyzer.analyze(game_state_dict, story_output)
    return state_manager.apply_updates(updates)


def _should_append_state() -> bool:
    """判断是否需要附加状态信息"""
    return (
        game_state["story_chapter_stage"] > 1 or
        game_state["story_chapter"] != "起因" or
        chat_data["current_id"] > 1
    )


def _handle_special_messages(message: str, history: List[Tuple[str, str]], ignore_job: str) -> str:
    """处理特殊消息命令"""
    if message == "开始" and not history:
        begin_message = settings.begin
        if settings.config["initial_state"]:
            begin_message += (
                "\n请在应当确认随机内容的时机一并初始化状态和持有物品，状态属性清单如下：" +
                "\n".join(settings.config["state_attributes"])
            )
        return begin_message

    if message == "确认" and len(history) == 2:
        if '{explored_jobs}' in settings.confirm:
            explored_text = f"这些是已探索过，需要排除的职业：{ignore_job}，" if ignore_job else ""
            return settings.confirm.format(explored_jobs=explored_text)
        return settings.confirm

    return message


def _process_response(chunk: Any, response: str) -> Tuple[str, bool]:
    """处理响应块，返回更新的响应和是否需要中断"""
    if not chunk.text:
        return response, False

    if "状态变化：" in chunk.text:
        response += chunk.text.split("状态变化：")[0]
        return response, True

    if ("【情节完成】" in chunk.text) and (chat_data["current_id"] > 1):
        response += chunk.text.split("【情节完成】")[0] + "【情节完成】"
        return response, True

    return response + chunk.text, False


def build_contents(message=None, before_message=None):
    """构建内容列表"""
    contents = []
    for val in chat_data["history"]:
        if val.get("only_for_display", False):
            continue
        contents.append(format_content(
            val["role"],
            val["content"]
        ))

    if before_message:
        contents.append(format_content(
            "assistant",
            before_message
        ))

    if message:
        contents.append(format_content(
            "user",
            message
        ))
    return contents


def respond(
    message: str,
    history: List[Tuple[str, str]],
    use_system_message: bool,
    ignore_job: str
) -> str:
    """处理用户输入并生成响应

    Args:
        message: 用户输入消息
        history: 对话历史
        use_system_message: 是否使用系统消息
        ignore_job: 要忽略的职业

    Returns:
        生成的响应文本
    """
    # 构建对话历史
    message = _handle_special_messages(message, history, ignore_job)
    # 判断是否需要附加状态信息
    if _should_append_state():
        message += state_manager.get_state_str()

    if message:
        # 处理普通消息
        contents = build_contents(message)
        response = ""
        config = get_content_config(use_system_message, settings.system_role)
        chat_data["current_id"] += 1
        chat_data["history"].append({
            "role": "user",
            "content": message,
            "idx": chat_data["current_id"]
        })
        log_and_print("before main response:\n", message)
        for chunk in gemini_client.models.generate_content_stream(
            model=MODEL_NAME,
            contents=contents,
            config=config
        ):
            response, should_break = _process_response(chunk, response)
            yield response
            if should_break:
                break
        log_and_print("after main response:\n", response)
        chat_data["history"].append({
            "role": "assistant",
            "content": response,
            "idx": chat_data["current_id"]
        })
        updates_str, _ = detect_state_changes(state_manager.get_state(), response)
        if updates_str:
            chat_data["history"].append({
                "role": "assistant",
                "content": updates_str,
                "only_for_display": True,
                "idx": chat_data["current_id"]
            })
            if "状态变化：" in response:
                yield response + "\n" + updates_str
            else:
                yield response + "\n状态变化：\n" + updates_str


with gr.Blocks(theme="soft") as demo:
    chatbot = gr.ChatInterface(
        respond,
        # fill_height=True,
        title=settings.config["title"],
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
            gr.Textbox(value="咖啡师", label="ignore job"),
        ],

    )

    def update_game_state_output() -> List[Any]:
        """更新游戏状态输出

        Returns:
            状态和历史记录列表
        """
        if settings.config["show_chat_history"]:
            return [state_manager.get_state(), chat_data["history"]]
        return state_manager.get_state()

    with gr.Accordion("查看故事状态", open=False):
        game_state_output = gr.JSON(value=state_manager.get_state())

    if settings.config["show_chat_history"]:
        with gr.Accordion("查看历史对话", open=False):
            history_output = gr.JSON(value=chat_data["history"])
        outputs_list = [game_state_output, history_output]
    else:
        outputs_list = [game_state_output]

    # 直接监听状态变化
    chatbot.chatbot.change(
        update_game_state_output,
        outputs=outputs_list
    )

    def clear_chat() -> List[Any]:
        """清空聊天历史

        Returns:
            更新后的状态和历史记录
        """
        chat_data['current_id'] = 0
        chat_data['history'] = []
        state_manager.state = create_initial_state()
        state_manager.state_history = create_initial_state()
        return update_game_state_output()

    chatbot.chatbot.clear(
        clear_chat,
        outputs=outputs_list
    )

    def undo_chat() -> List[Any]:
        """撤销上一步聊天

        Returns:
            更新后的状态和历史记录
        """
        if chat_data["current_id"] > 0:
            # 找到并删除最后一组对话（可能包含多条记录）
            current_idx = chat_data["current_id"]
            chat_data["history"] = [
                msg for msg in chat_data["history"]
                if msg["idx"] != current_idx
            ]
            chat_data["current_id"] -= 1
        state_manager.reset_state()
        return update_game_state_output()

    chatbot.chatbot.undo(
        undo_chat,
        outputs=outputs_list
    )

    chatbot.chatbot.retry(
        undo_chat,
        outputs=outputs_list
    )

if __name__ == "__main__":
    ssl_cert = get_file_path("localhost+1.pem", current_dir=current_dir)
    ssl_key = get_file_path("localhost+1-key.pem", current_dir=current_dir)

    if os.path.exists(ssl_cert) and os.path.exists(ssl_key):
        demo.launch(server_name="0.0.0.0", ssl_certfile=ssl_cert, ssl_keyfile=ssl_key)
    else:
        demo.launch(server_name="0.0.0.0", share=True)
