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
    if settings.config.get("log_level", "") == "debug":
        log_and_print("game_response_formatter:\n", response.text)
    updates = json.loads(response.text)

    state_updates = updates.get('stateUpdates', [])

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
    # 获取配置
    state_config = settings.config["state_analysis"]
    extra_state_attribute = settings.config.get("extra_state_attributes", [])

    # 构建规则和提示
    rules = "\n".join(f"- {rule}" for rule in state_config["rules"])
    extra_state_hint = ""

    # 构建状态内容
    character_state = current_data.get('character_state', {})

    formatted_state = []
    for attr, state in character_state.items():
        if attr and state:
            formatted_state.append(f"{attr}: {state['state'] if state['state'] else '暂无'}")
            # formatted_state.append(f"{attr}: {state['state']}")
    final_state = "- " + '\n- '.join(formatted_state)

    state_content = f"""
故事发生前的状态
【物品清单】
{current_data.get('inventory', {})}

【角色状态】
{final_state}
"""
    # 构建完整内容
    formatted_content = state_config["template"].format(
        rules=rules,
        extra_hint=extra_state_hint
    ) + state_content

    # Debug日志
    if settings.config.get("log_level", "") == "debug":
        log_and_print("game_context_formatter:\n", formatted_content + "【最近故事内容】\n")

    # 返回带故事内容的完整格式
    return formatted_content + f"""
【最近故事内容】
{content}
【最近故事内容】
"""


def game_extra_response_formatter(response: Any) -> Dict[str, Any]:
    """格式化额外属性响应"""
    if settings.config.get("log_level", "") == "debug":
        log_and_print("game_extra_response_formatter:\n", response.text)
    updates = json.loads(response.text)
    state_updates = updates.get('stateUpdates', [])
    extra_state_attribute = settings.config.get("extra_state_attributes", [])[0]
    for update in state_updates:
        if 'new_value' in update:
            update[extra_state_attribute] = update.pop('new_value')
    return {'stateUpdates': state_updates}


def game_extra_context_formatter(base_result: Dict, current_data: Dict) -> str:
    """游戏特定的额外评估上下文格式化"""
    changes = []
    extra_attrs = settings.config.get('extra_state_attributes', [])
    attr_names = settings.config.get('state_attribute_names', {})
    guidance = settings.config.get('extra_value_evaluation', {}).get('guidance', [])

    for update in base_result.get('stateUpdates', []):
        attr = update['attribute']
        current_state = current_data.get('character_state', {}).get(attr, {})
        if update['to_state'] != current_state.get('state', ''):
            changes.extend([
                f"属性：{attr}",
                f"初始状态：{current_state.get('state', '')}",
                f"变化后的状态：{update['to_state']}"
            ])
            # 添加所有配置的extra属性
            for extra_attr in extra_attrs:
                attr_name = attr_names.get(extra_attr, extra_attr)
                changes.append(
                    f"初始{attr_name}值：{current_state.get(extra_attr, 0)}"
                )
            changes.append("")  # 添加空行分隔

    result = ""
    if changes:
        result = "\n".join(guidance) + "\n" + "\n".join(changes)

    if settings.config.get("log_level", "") == "debug":
        log_and_print("game_extra_context_formatter:\n", result, "\n id:", chat_data.get('current_id', 0))

    return result


def game_merge_updates(base_result: Dict, extra_result: Dict) -> Dict:
    """游戏特定的更新合并策略"""
    base_updates = base_result.get('character_state', [])
    extra_updates = extra_result.get('stateUpdates', [])

    # 记录每个属性首次出现的位置和更新
    seen_attrs = {}
    final_updates = []

    # 检查是否有extra属性用于比较
    extra_state_attributes = settings.config.get("extra_state_attributes", [])
    extra_state_attribute = extra_state_attributes[0] if extra_state_attributes else None

    # 处理基础更新
    for update in base_updates:
        attr = update['attribute']
        # 标准化更新数据格式
        normalized_update = {
            'attribute': attr,
            'from_state': update.get('from_state'),
            'state': update.get('to_state'),  # 统一使用 state 作为键名
            **{k: v for k, v in update.items() if k not in ['attribute', 'from_state', 'to_state', 'state']}
        }

        if attr in seen_attrs:
            prev_idx = seen_attrs[attr]
            prev_update = final_updates[prev_idx]

            if (extra_state_attribute in update and
                extra_state_attribute in prev_update):
                # 如果新的extra值更大，则替换旧的更新
                if update[extra_state_attribute] > prev_update[extra_state_attribute]:
                    final_updates[prev_idx] = normalized_update
            else:
                # 没有extra属性时保留后出现的更新
                final_updates[prev_idx] = normalized_update
        else:
            seen_attrs[attr] = len(final_updates)
            final_updates.append(normalized_update)

    # 处理额外属性更新
    for extra in extra_updates:
        attr = extra['attribute']
        if attr in seen_attrs:
            idx = seen_attrs[attr]
            # 将额外属性合并到对应的更新中
            final_updates[idx].update({
                k: v for k, v in extra.items()
                if k != 'attribute'
            })

    # 构建最终结果
    result = {k: v for k, v in base_result.items() if k != 'character_state'}
    result['character_state'] = final_updates
    result['extraUpdates'] = extra_result

    return result


# 创建分析器实例
analyzer = ContentAnalyzer(
    llm_client=gemini_client,
    generate_func=gemini_generate_with_schema,
    response_schema=settings.response_schema,
    system_prompt=settings.state_system_prompt,
    context_formatter=game_context_formatter,
    response_formatter=game_response_formatter,
    # 额外评估配置
    extra_schema=settings.config.get('extra_value_evaluation', {}).get('schema'),
    extra_prompt=settings.config.get('extra_value_evaluation', {}).get('system_role'),
    extra_context_formatter=game_extra_context_formatter,
    extra_response_formatter=game_extra_response_formatter,
    merge_updates=game_merge_updates
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
            initial_state["character_state"][attr][extra_attr] = -1
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


def _handle_special_messages(message: str, history: List[Tuple[str, str]], ignore_job: str) -> str:
    """处理特殊消息命令"""
    jobs_config = settings.config.get("explored_jobs", {})

    def apply_jobs_template(template_text: str) -> str:
        """应用职业排除模板"""
        if message == jobs_config.get("trigger") and '{explored_jobs}' in template_text:
            explored_text = jobs_config["template"].format(jobs=ignore_job) if ignore_job else ""
            return template_text.format(explored_jobs=explored_text)
        return template_text

    if message == "开始" and not history:
        begin_message = settings.begin
        if settings.config["initial_state"]:
            begin_message += (
                "\n请在应当确认随机内容的时机一并初始化状态和持有物品，状态属性清单如下：\n" +
                "\n".join(settings.config["state_attributes"])
            )
        return apply_jobs_template(begin_message)

    if message == "确认" and len(history) == 2:
        return apply_jobs_template(settings.confirm)

    return message


def _should_append_state(message: str) -> Tuple[bool, bool, str]:
    """判断是否需要附加状态信息

    Args:
        message: 输入的消息

    Returns:
        Tuple[bool, bool, str]:
        - 是否需要附加状态
        - 是否是控制命令
        - 处理后的消息
    """
    # 检查是否是控制命令
    is_control = message.startswith('ct')
    processed_message = message[2:].strip() if is_control else message

    # 判断是否需要附加状态（控制命令一定不附加，其他情况按原有逻辑判断）
    should_append = False if is_control else (
        game_state["story_chapter_stage"] > 1 or
        game_state["story_chapter"] != "起因" or
        chat_data["current_id"] > 1
    )

    return should_append, is_control, processed_message


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
    add_extra_message: bool,
    ignore_job: str,
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

    should_append, is_control, message = _should_append_state(message)

    if (should_append) and (add_extra_message):
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

        if settings.config.get("log_level", "") in ["debug", "info"]:
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

        if settings.config.get("log_level", "") in ["debug", "info"]:
            log_and_print("after main response:\n", response)

        chat_data["history"].append({
            "role": "assistant",
            "content": response,
            "idx": chat_data["current_id"]
        })

        if not is_control:
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
    # 1. 创建界面组件
    chatbot = gr.ChatInterface(
        respond,
        title=settings.config["title"],
        type="messages",
        # examples=[["开始",True,settings.config.get("explored_jobs", {}).get("default", "")]],
        chatbot=gr.Chatbot(
            placeholder="输入 【开始】 开始进行创作",
            height="80vh",
            show_share_button=True,
            editable="user",
            show_copy_all_button=True,
            type="messages",
        ),
        additional_inputs=[
            # gr.Row([
            gr.Checkbox(value=True, label="Use system message"),
            gr.Checkbox(value=True, label="Add Extra message"),
            # ]),
            gr.Textbox(
                value=settings.config.get("explored_jobs", {}).get("default", ""),
                label="ignore job"
            ),
        ],
    )

    # 2. 创建状态显示组件
    outputs = []
    with gr.Accordion("查看故事状态", open=False):
        state_output = gr.JSON(value=state_manager.get_state())
        outputs.append(state_output)

    if settings.config["show_chat_history"]:
        with gr.Accordion("查看历史对话", open=False):
            history_output = gr.JSON(value=chat_data["history"])
            outputs.append(history_output)

    # 3. 定义事件处理函数（保持全局状态访问）
    def update_state() -> List[Any]:
        return ([state_manager.get_state(), chat_data["history"]]
                if settings.config["show_chat_history"]
                else state_manager.get_state())

    def clear_state() -> List[Any]:
        chat_data['current_id'] = 0
        chat_data['history'] = []
        state_manager.state = create_initial_state()
        state_manager.state_history = create_initial_state()
        return update_state()

    def undo_state() -> List[Any]:
        if chat_data["current_id"] > 0:
            chat_data["history"] = [
                msg for msg in chat_data["history"]
                if msg["idx"] != chat_data["current_id"]
            ]
            chat_data["current_id"] -= 1
        state_manager.reset_state()
        return update_state()

    # 4. 绑定事件处理
    chatbot.chatbot.change(fn=update_state, outputs=outputs)
    chatbot.chatbot.clear(fn=clear_state, outputs=outputs)
    chatbot.chatbot.undo(fn=undo_state, outputs=outputs)
    chatbot.chatbot.retry(fn=undo_state, outputs=outputs)


if __name__ == "__main__":
    try:
        # 尝试获取命令行参数
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
        parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
        parser.add_argument("--share", action="store_true", help="Enable sharing")
        args = parser.parse_args()
        launch_kwargs = {
            "server_name": args.host,
            "server_port": args.port,
            "share": args.share
        }
    except:
        # 在notebook中运行时使用默认值
        launch_kwargs = {
            "server_name": "0.0.0.0",
            "share": False
        }

    ssl_cert = get_file_path("localhost+1.pem", current_dir=current_dir)
    ssl_key = get_file_path("localhost+1-key.pem", current_dir=current_dir)

    if os.path.exists(ssl_cert) and os.path.exists(ssl_key):
        launch_kwargs.update({
            "ssl_certfile": ssl_cert,
            "ssl_keyfile": ssl_key
        })

    # 启动应用
    demo.launch(**launch_kwargs)
