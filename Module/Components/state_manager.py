"管理状态变化"
from typing import Dict, List, Optional
from copy import deepcopy
import random
# pylint: disable=too-few-public-methods


class StateManager:
    """状态管理器"""
    def __init__(self, initial_state: Dict, config: Dict):
        self.state = deepcopy(initial_state)
        self.state_history = deepcopy(initial_state)
        self.config = config
        self._extra_state_changes = {}

    def get_state(self) -> Dict:
        """获取状态"""
        return self.state

    def get_state_history(self) -> Dict:
        """获取历史状态"""
        return self.state_history

    def reset_state(self) -> Dict:
        """重置状态"""
        self.state = deepcopy(self.state_history)
        return self.state

    def apply_updates(self, updates: Dict) -> Dict:
        """应用状态更新"""
        self.state_history = deepcopy(self.state)

        inventory = self.state.get('inventory', {})
        character_state = self.state.get('character_state', {})

        # 处理物品更新
        item_updates_dict = self._process_item_updates(updates.get('inventory', []))
        self._update_inventory(inventory, item_updates_dict)

        # 处理状态更新
        self._update_character_state(character_state, updates.get('character_state', []))

        self.state['inventory'] = inventory
        self.state['character_state'] = character_state
        clean_updates = {
            'itemUpdates': [
                {'name': name, 'change_amount': change}
                for name, change in item_updates_dict.items()
            ],
            'stateUpdates': updates.get('character_state', [])
        }
        final_state_str = self.format_state_changes(clean_updates)

        return final_state_str, clean_updates

    def _format_item_changes(self, updates: Dict) -> List[str]:
        """格式化物品变化信息"""
        messages = []
        for item in updates.get('itemUpdates', []):
            name = item['name']
            change = item['change_amount']
            if change > 0:
                messages.append(f"获得了 {name} x{change}")
            elif change < 0:
                messages.append(f"失去了 {name} x{abs(change)}")
        return messages

    def _format_extra_state_changes(self, attr: str) -> List[str]:
        """格式化额外状态属性变化信息"""
        extra_changes = []
        if attr not in self._extra_state_changes:
            print(f"状态变化日志：{attr}应该有变化，但实际没有变化")
            return extra_changes

        for extra_attr, change in self._extra_state_changes[attr].items():
            from_value = change['from_value']
            to_value = change['to_value']
            if from_value == to_value:
                print(f"状态变化日志：{attr}的{extra_attr}从{int(from_value)}变为{int(to_value)}")
                continue

            extra_attr_name = self.config.get(
                'state_attribute_names',
                {}
            ).get(extra_attr, extra_attr)

            if from_value > 0:
                extra_changes.append(
                    f"{extra_attr_name}从{int(from_value)}变为{int(to_value)}"
                )
            else:
                extra_changes.append(
                    f"{extra_attr_name}变为{int(to_value)}"
                )

        return extra_changes

    def _format_main_state_change(self, state: Dict) -> Optional[str]:
        """格式化主状态变化信息"""
        from_state = state.get('from_state')
        to_state = state['to_state']

        if from_state == to_state:
            return None

        if from_state and from_state != "无":
            return f"从「{from_state}」变成了「{to_state}」"
        if to_state:
            return f"变成了「{to_state}」"
        print(f"状态变化日志：{state['attribute']}从「{from_state}」变成了「{to_state}」")
        return None

    def format_state_changes(self, updates: Dict) -> str:
        """格式化输出的更新"""
        messages = self._format_item_changes(updates)

        # Format state changes
        extra_state_attributes = self.config.get("extra_state_attributes", [])
        for state in updates.get('stateUpdates', []):
            attr = state['attribute']
            state_msg_parts = []

            # 添加主状态变化
            main_state_change = self._format_main_state_change(state)
            if main_state_change:
                state_msg_parts.append(main_state_change)

            # 添加额外属性变化
            if extra_state_attributes:
                extra_changes = self._format_extra_state_changes(attr)
                if extra_changes:
                    state_msg_parts.append(f"({', '.join(extra_changes)})")

            # 添加完整的状态变化消息
            if state_msg_parts:
                messages.append(f"{attr}" + "".join(state_msg_parts))

        return "\n".join(messages) if messages else ""

    def _process_extra_attributes(self, character_state: Dict, extra_attrs: List[str]) -> str:
        """处理额外属性并生成指导字符串"""
        if not extra_attrs:
            return ""

        # 收集所有状态的extra_attr值
        attr_values = [
            (attr, state.get(extra_attrs[0], 0))
            for attr, state in character_state.items()
        ]
        # 先取一个值也行吧
        extra_attr_name = self.config.get(
            'state_attribute_names',
            {}
        ).get(extra_attrs[0], extra_attrs[0])

        if not attr_values:
            return ""

        # 按值排序
        attr_values.sort(key=lambda x: x[1])

        # 获取最低和最高值
        # Filter out zero values for highest calculation
        non_zero_values = [x for x in attr_values if x[1] > 0]

        # 获取最低值的属性
        min_value = attr_values[0][1]
        min_value_attrs = [x for x in attr_values if x[1] == min_value]

        # 根据最低值属性数量选择不同的随机策略
        if len(min_value_attrs) >= 4:
            # 如果最低值属性超过4个，从中随机选3个
            lowest_three = random.sample(min_value_attrs, 3)
        else:
            # 否则从排序最低的5个中随机选3个
            lowest_five = attr_values[:min(5, len(attr_values))]
            lowest_three = random.sample(lowest_five, min(3, len(lowest_five)))

        highest_three = non_zero_values[-min(3, len(non_zero_values)):] if non_zero_values else []

        guidance_parts = []
        # 添加最低值信息
        if lowest_three:
            lowest_str = ", ".join(
                f"{attr}({extra_attr_name}值:{val})"
                for attr, val in lowest_three
            )
            guidance_parts.append(
                f"选中的需要提高{extra_attr_name}的最低的三个属性是: {lowest_str}"
            )

        # 添加随机高值信息
        if highest_three:
            random_high = random.choice(highest_three)
            guidance_parts.append(
                f"选中的需要继续加强提高{extra_attr_name}的属性是: "
                f"{random_high[0]}({extra_attr_name}值:{random_high[1]})"
            )

        return "\n" + "\n".join(guidance_parts) if guidance_parts else ""

    def get_state_str(self, ignore_blank: bool = False) -> str:
        """获取状态字符串"""
        inventory = self.state.get('inventory', {})
        character_state = self.state.get('character_state', {})

        # 如果ignore_blank为True，过滤掉空值
        if ignore_blank:
            character_state = {k: v for k, v in character_state.items() if v}

        # 从配置中获取模板和指导说明
        state_display = self.config.get('state_display', {})
        template = state_display.get('template', '')
        guidance = state_display.get('guidance', [])

        # 构建指导说明字符串
        guidance_str = "\n".join(guidance)

        # 处理额外属性
        extra_attrs = self.config.get('extra_state_attributes', [])
        if extra_attrs:
            guidance_str += self._process_extra_attributes(character_state, extra_attrs)

        character_state = {k: v['state'] for k, v in character_state.items()}

        state_str = template.format(
            state_guidance=guidance_str,
            inventory=inventory,
            character_state=character_state
        )

        return state_str

    def _process_item_updates(self, item_updates: List[Dict]) -> Dict:
        """处理物品更新，合并重复项"""
        item_updates_dict = {}
        for item in item_updates:
            if 'name' in item:
                name = item['name']
                change = item.get('change_amount', 1)
                if name in item_updates_dict:
                    if abs(change) > abs(item_updates_dict[name]):
                        item_updates_dict[name] = change
                else:
                    item_updates_dict[name] = change
        return item_updates_dict

    def _update_inventory(self, inventory: Dict, updates: Dict):
        """更新物品栏"""
        for name, change in updates.items():
            if name not in inventory:
                inventory[name] = 0
            inventory[name] += change
            if inventory[name] <= 0:
                del inventory[name]

    def _update_character_state(self, character_state: Dict, updates: List[Dict]):
        """更新角色状态"""
        self._extra_state_changes = {}
        for state in updates:
            attr = state['attribute']
            to_state = state['to_state']

            # 确保属性存在
            attr_is_exist = attr in character_state
            if not attr_is_exist:
                character_state[attr] = {"state": ""}

            # 更新主状态
            character_state[attr]["state"] = to_state

            # 更新额外属性
            for extra_attr in self.config.get('extra_state_attributes', []):
                if extra_attr in state:
                    old_extra_value = character_state[attr].get(extra_attr, 0)
                    # 确保新值至少比旧值大1，但不超过100
                    if (
                        attr_is_exist and
                        state['from_state'] and
                        state['from_state'] != state['to_state']
                    ):
                        # 从配置中获取diff值，如果没有配置则默认为1
                        diff = self.config.get('state_attribute_diffs', {}).get(extra_attr, 1)
                        new_value = max(min(state[extra_attr], 100), old_extra_value + diff)
                        if new_value != state[extra_attr]:
                            print(
                                f"状态变化特别处理日志："
                                f"{attr}的{extra_attr}"
                                f"本来是{state[extra_attr]}，"
                                f"实际为{new_value}"
                            )
                    else:
                        new_value = state[extra_attr]
                    character_state[attr][extra_attr] = new_value
                    self._extra_state_changes[attr] = {}
                    self._extra_state_changes[attr][extra_attr] = {
                        'from_value': old_extra_value,
                        'to_value': new_value
                    }
