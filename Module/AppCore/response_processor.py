"""响应处理器模块,负责处理LLM响应的核心逻辑"""
import json
from typing import Dict, Any, List, Tuple, Generator
from Module.Components.state_manager import StateManager
from Module.Common.scripts.llm.gemini_sdk import (
    types,
    get_safety_settings,
    format_content,
    get_content_config
)
from Module.Common.scripts.common.debug_utils import log_and_print
from Module.Common.scripts.llm.utils.schema_response import ContentAnalyzer


class ResponseProcessor:
    """响应处理器类,处理LLM响应的核心逻辑"""

    def __init__(
        self,
        state_manager: StateManager,
        llm_client: Any,
        model_name: str,
        settings: Any,
    ):
        self.state_manager = state_manager
        self.llm_client = llm_client
        self.model_name = model_name
        self.settings = settings
        self.safety_settings = get_safety_settings()
        for detail_safety_settings in self.safety_settings:
            detail_safety_settings.threshold = self.settings.config.get("safety_settings", "OFF")

        self.extra_schema = self.settings.config.get('extra_value_evaluation', {}).get('schema')
        self.extra_prompt = self.settings.config.get('extra_value_evaluation', {}).get('system_role')
        enable_extra = self.extra_schema and self.extra_prompt
        # 初始化分析器
        self.analyzer = ContentAnalyzer(
            llm_client=self.llm_client,
            generate_func=self._gemini_generate_with_schema,
            response_schema=self.settings.response_schema,
            system_prompt=self.settings.state_system_prompt,
            context_formatter=self._game_context_formatter,
            response_formatter=self._game_response_formatter,
            post_process=self.extra_post_process,
            enable_extra=enable_extra,
        )

    def safe_check(self, message: str) -> bool:
        """安全检查"""
        format_msg = self.settings.config.get("prompt_safe_check", "检查{user_message}").format(user_message=message)
        response = self.llm_client.models.generate_content(
            model=self.model_name,
            contents=[format_msg],
        )
        cleaned_text = response.text.strip().replace('\n', '')
        if cleaned_text == "安全":
            return True
        return False

    def pre_process(
        self,
        message: str,
        history: List[Tuple[str, str]],
        add_extra_message: bool,
        ignore_job: str,
        state: Dict,
        chat_data: Dict
    ) -> Tuple[str, bool]:
        """预处理阶段,处理消息和状态"""
        # 处理特殊消息
        message = self._handle_special_messages(message, history, ignore_job)

        # 判断是否需要附加状态
        should_append, is_control, message = self._should_append_state(message, state, chat_data)

        # 附加状态信息
        if should_append and add_extra_message:
            message += self.state_manager.get_state_str()

        return message, is_control

    def main_process(
        self,
        message: str,
        use_system_message: bool,
        state: Dict,
        chat_data: Dict
    ) -> Generator[str, None, None]:
        """主处理阶段,调用LLM并处理响应"""
        if not message:
            print("DEBUG-RP-102: Empty message, returning")
            return

        # 构建内容和配置
        contents = self._build_contents(message, chat_data=chat_data)
        config = get_content_config(use_system_message, self.settings.system_role)
        config.temperature=self.settings.config.get("generate_config_temperature", 1)
        config.frequency_penalty=self.settings.config.get("generate_config_frequency_penalty", 1)
        config.presence_penalty=self.settings.config.get("generate_config_presence_penalty", 1)

        # 调试日志
        if self.settings.config.get("log_level", "") in ["debug", "info"]:
            separator = "\n" + "★"*30 + "《CONTENT START》" + "★"*30 + "\n"
            separator_end = "\n" + "☆"*30 + "《CONTENT END》" + "☆"*30 + "\n"
            log_and_print("content before main response:\n", separator, message, separator_end)

        # 生成响应
        response = ""
        for chunk in self.llm_client.models.generate_content_stream(
            model=self.model_name,
            contents=contents,
            config=config
        ):
            last_chunk = chunk  # 保存最后一个 chunk
            response, should_break = self._process_response(chunk, response, chat_data)
            yield response
            if should_break:
                break

        # 更新对话历史
        if response:
            # 更新对话历史
            chat_data["current_id"] += 1
            chat_data["history"].append({
                "role": "user",
                "content": message,
                "idx": chat_data["current_id"]
            })
            chat_data["history"].append({
                "role": "assistant",
                "content": response,
                "idx": chat_data["current_id"]
            })

        # 记录最终响应日志（新增部分）
        if self.settings.config.get("log_level", "") in ["debug", "info"]:
            separator = "\n" + "="*30 + "RESPONSE START" + "="*30 + "\n"
            separator_end = "\n" + "-"*30 + "RESPONSE END" + "-"*30 + "\n"

            error_info = ""
            if isinstance(last_chunk, str):
                error_info = last_chunk
            else:
                candidates = getattr(last_chunk, 'candidates', None)
                if candidates:
                    finish_reason = getattr(candidates[0], 'finish_reason', None)
                    if finish_reason and finish_reason != "STOP":
                        error_info = f"Finish reason: {finish_reason}"
                else:
                    prompt_feedback = getattr(last_chunk, 'prompt_feedback', None)
                    if prompt_feedback:
                        block_reason = getattr(prompt_feedback, 'block_reason', None)
                        if block_reason:
                            error_info = f"Block reason: {block_reason}"

            if error_info:
                response += "\n结果异常中断：" + error_info

            # 记录处理后的响应
            log_and_print("Processed response:", separator, response, separator_end)

    def post_process(
        self,
        response: str,
        is_control: bool,
        auto_analysis_state: bool,
        state: Dict,
        chat_data: Dict
    ) -> str:
        """后处理阶段,处理状态变化"""
        if not is_control and auto_analysis_state:
            updates = self.analyzer.analyze(state, response)
            updates_str, _ = self.state_manager.apply_updates(updates)
            if updates_str:
                chat_data["history"].append({
                    "role": "assistant",
                    "content": updates_str,
                    "only_for_display": True,
                    "idx": chat_data["current_id"]
                })
                # 根据响应内容格式化状态更新
                if "状态变化：" in response:
                    return response + "\n" + updates_str
                else:
                    return response + "\n状态变化：\n" + updates_str
            return response
        return response

    def _handle_special_messages(
        self,
        message: str,
        history: List[Tuple[str, str]],
        ignore_job: str
    ) -> str:
        """处理特殊消息命令"""
        jobs_config = self.settings.config.get("explored_jobs", {})

        def apply_jobs_template(template_text: str) -> str:
            if message == jobs_config.get("trigger") and '{explored_jobs}' in template_text:
                explored_text = jobs_config["template"].format(jobs=ignore_job) if ignore_job else ""
                return template_text.format(explored_jobs=explored_text)
            return template_text

        if message == "开始" and not history:
            begin_message = self.settings.begin
            if self.settings.config["initial_state"]:
                begin_message += (
                    "\n请在应当确认随机内容的时机一并初始化状态和持有物品，状态属性清单如下：\n" +
                    "\n".join(self.settings.config["state_attributes"])
                )
            return apply_jobs_template(begin_message)

        if message == "确认" and len(history) == 2:
            return apply_jobs_template(self.settings.confirm)

        return message

    def _should_append_state(
        self,
        message: str,
        state: Dict,
        chat_data: Dict
    ) -> Tuple[bool, bool, str]:
        """判断是否需要附加状态信息"""
        is_control = message.startswith('ct')
        processed_message = message[2:].strip() if is_control else message

        should_append = False if is_control else (
            state["story_chapter_stage"] > 1 or
            state["story_chapter"] != "起因" or
            chat_data["current_id"] > 1
        )

        return should_append, is_control, processed_message

    def _process_response(
        self,
        chunk: Any,
        response: str,
        chat_data: Dict
    ) -> Tuple[str, bool]:
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

    def _build_contents(
        self,
        message: str = None,
        before_message: str = None,
        chat_data: Dict = None
    ) -> List[Dict]:
        """构建内容列表"""
        contents = []
        if chat_data:
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

    def _gemini_generate_with_schema(
        self,
        client: Any,
        input_text: str,
        response_schema: Dict,
        system_prompt: str = ""
    ) -> Any:
        """使用Gemini生成带schema的响应"""
        return client.models.generate_content(
            model=self.model_name,
            contents=[format_content("user", input_text)],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=response_schema,
                safety_settings=self.safety_settings,
            ),
        )

    def _game_response_formatter(self, response: Any) -> Dict[str, Any]:
        """格式化游戏响应"""
        if self.settings.config.get("log_level", "") == "debug":
            log_and_print("game_response_formatter:\n", response.text)
        updates = json.loads(response.text)

        state_updates = updates.get('stateUpdates', [])
        updates['stateUpdates'] = [u for u in state_updates if u is not None]

        return {
            'inventory': updates.get('itemUpdates', []),
            'character_state': updates.get('stateUpdates', [])
        }

    def _game_context_formatter(self, current_data: Dict[str, Any], content: str) -> str:
        """格式化游戏上下文"""
        # 获取配置
        state_config = self.settings.config["state_analysis"]
        extra_state_attribute = self.settings.config.get("extra_state_attributes", [])

        # 构建规则和提示
        rules = "\n".join(f"- {rule}" for rule in state_config["rules"])
        extra_state_hint = ""

        # 构建状态内容
        character_state = current_data.get('character_state', {})

        formatted_state = []
        for attr, state in character_state.items():
            if attr and state:
                formatted_state.append(f"{attr}: {state['state'] if state['state'] else '暂无'}")
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
        if self.settings.config.get("log_level", "") == "debug":
            log_and_print("game_context_formatter:\n", formatted_content + "【最近故事内容】\n")
        else:
            log_and_print("game_context_formatter: log_level is not debug")

        # 返回带故事内容的完整格式
        return formatted_content + f"""
【最近故事内容】
{content}
【最近故事内容】
"""

    def extra_post_process(self, result: Dict, current_data: Dict) -> Dict:
        """额外属性后处理"""
        filtered_result = {
            **result,
            'character_state': [update for update in result.get('character_state', [])
                            if update.get('to_state') != '暂无']
        }

        state_updates = filtered_result['character_state']
        update_groups = self._group_updates_by_order(state_updates)

        extra_results = []
        # 对每个分组分别进行额外属性评估
        for group_idx, group in enumerate(update_groups):
        # for group in update_groups:
            extra_context = self._game_extra_context_formatter(group, current_data)
            if extra_context:
                extra_response = self._gemini_generate_with_schema(
                    self.llm_client,
                    extra_context,
                    self.extra_schema,
                    self.extra_prompt,
                )
                extra_result = self._game_extra_response_formatter(extra_response)
                for item in extra_result.get('stateUpdates', []):
                    item['_group_index'] = group_idx  # 添加分组标记
                extra_results.extend(extra_result.get('stateUpdates', []))

        # 合并所有额外属性更新, 主要实现根据extra的值和位置去重
        result = self._game_merge_updates(
            filtered_result,
            {'stateUpdates': extra_results},
            current_data
        )
        return result

    def _group_updates_by_order(self, updates: List[Dict]) -> List[Dict]:
        """按出现顺序将更新分组

        返回：[
            {  # 第一组：所有属性的第一次出现
                'stateUpdates': [
                    {'attribute': 'A', ...},
                    {'attribute': 'B', ...},
                    {'attribute': 'C', ...}
                ]
            },
            {  # 第二组：第二次出现的属性
                'stateUpdates': [
                    {'attribute': 'B', ...}
                ]
            },
            # ... 可能有更多组
        ]
        """
        seen_attrs = {}  # 记录每个属性出现的次数
        groups = []  # 动态增长的分组列表

        for update in updates:
            attr = update['attribute']
            count = seen_attrs.get(attr, 0)

            # 确保有足够的分组
            while len(groups) <= count:
                groups.append({'stateUpdates': []})

            # 将更新添加到对应的分组
            groups[count]['stateUpdates'].append(update)
            seen_attrs[attr] = count + 1

        # 移除空组（如果有的话）
        return [group for group in groups if group['stateUpdates']]

    def _game_extra_response_formatter(self, response: Any) -> Dict[str, Any]:
        """格式化额外属性响应"""
        if self.settings.config.get("log_level", "") == "debug":
            log_and_print("game_extra_response_formatter:\n", response.text)
        updates = json.loads(response.text)
        state_updates = updates.get('stateUpdates', [])
        extra_state_attribute = self.settings.config.get("extra_state_attributes", [])[0]
        for update in state_updates:
            if 'new_value' in update:
                update[extra_state_attribute] = update.pop('new_value')
        return {'stateUpdates': state_updates}

    def _game_extra_context_formatter(self, base_result: Dict, current_data: Dict) -> str:
        """游戏特定的额外评估上下文格式化"""
        changes = []
        extra_attrs = self.settings.config.get('extra_state_attributes', [])
        attr_names = self.settings.config.get('state_attribute_names', {})
        guidance = self.settings.config.get('extra_value_evaluation', {}).get('guidance', [])

        for update in base_result.get('stateUpdates', []):
            attr = update['attribute']
            current_state = current_data.get('character_state', {}).get(attr, {})
            if (update['to_state'] != current_state.get('state', '')) and (update['to_state'] != update['from_state']):
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

        if self.settings.config.get("log_level", "") == "debug":
            log_and_print("game_extra_context_formatter:\n", result)

        return result

    def _game_merge_updates(self, base_result: Dict, extra_result: Dict, current_data: Dict) -> Dict:
        """增强版游戏属性合并策略"""
        base_updates = base_result.get('character_state', [])
        extra_updates = extra_result.get('stateUpdates', [])
        extra_attr = self.settings.config.get("extra_state_attributes", [])[0]
        current_states = current_data.get('character_state', {})
        used_extra_updates = []  # 记录实际使用的额外更新

        # 构建属性更新链（记录所有历史版本）
        update_chains = {}
        for idx, update in enumerate(base_updates):
            attr = update['attribute']
            if attr not in update_chains:
                update_chains[attr] = []
            update_chains[attr].append({
                'base_index': idx,
                'from_state': update.get('from_state'),
                'to_state': update.get('to_state'),
                'group_index': len(update_chains[attr]),  # 当前属性出现的次数作为分组索引
            })

        # 处理带分组标记的额外属性更新
        for extra in extra_updates:
            attr = extra['attribute']
            group_idx = extra.get('_group_index', 0)

            # 找到对应分组的base更新
            if attr in update_chains and group_idx < len(update_chains[attr]):
                target_group = update_chains[attr][group_idx]
                base_update = base_updates[target_group['base_index']]

                # 合并额外属性（保留原始值优先级）
                current_value = current_states.get(attr, {}).get(extra_attr, 0)
                new_value = extra.get(extra_attr, 0)
                if new_value > current_value:
                    base_update[extra_attr] = new_value
                    base_update['group_index'] = group_idx

        # 生成最终更新列表（应用业务规则）
        final_updates = []
        for attr, groups in update_chains.items():
            # 按业务规则选择最终值：最高extra值，相同则选最后出现的
            selected_group = max(
                groups,
                key=lambda x: (
                    x.get(extra_attr, 0),
                    x['group_index']  # 相同值时选择最后出现的
                )
            )
            final_update = base_updates[selected_group['base_index']].copy()

            for extra in extra_updates:
                if (attr == extra['attribute']) and (extra['_group_index'] == selected_group['group_index']):
                    used_extra_updates.append(extra)  # 记录实际使用的额外更新
            final_updates.append(final_update)

        deduped_updates_dict: Dict[str, Dict[str, Any]] = {}
        for update in final_updates:
            deduped_updates_dict[update['attribute']] = update
        deduped_updates: List[Dict[str, Any]] = list(deduped_updates_dict.values())

        final_result = {
            **base_result,
            'character_state': deduped_updates,
            'extraUpdates': {
                **extra_result,
                'stateUpdates': used_extra_updates  # 使用过滤后的有效额外更新
            }
        }

        return final_result

    def _should_break_response(self, response: str) -> bool:
        """检查是否需要中断响应生成"""
        # 检查状态变化标记
        if "状态变化：" in response:
            return True

        # 检查情节完成标记
        if ("【情节完成】" in response) and (self.chat_data["current_id"] > 1):
            return True

        return False