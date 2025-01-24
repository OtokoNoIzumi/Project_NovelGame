"管理状态变化"
from typing import Dict, List, Optional, Tuple
from copy import deepcopy
import random
# pylint: disable=too-few-public-methods


class ExtraAttributeManager:
    """管理额外属性的处理逻辑"""
    def __init__(self, config: Dict):
        self.config = config
        self.extra_attrs = config.get('extra_state_attributes', [])
        self.attr_names = config.get('state_attribute_names', {})
        self.dependencies = config.get('state_dependencies', {})
        self.attr_diffs = config.get('state_attribute_diffs', {})
        self._changes = {}

    def has_extra_attrs(self) -> bool:
        """检查是否有额外属性需要处理"""
        return bool(self.extra_attrs)

    def check_state_dependencies(
        self,
        selected_attrs: List[Tuple[str, int]],
        attr_values: List[Tuple[str, int]],
    ) -> List[Tuple[str, int]]:
        """检查并处理状态依赖关系"""
        attr_dict = dict(attr_values)
        result = list(selected_attrs)

        for i, (attr, value) in enumerate(selected_attrs):
            if attr in self.dependencies:
                for prereq in self.dependencies[attr]:
                    prereq_value = attr_dict.get(prereq, 0)
                    if prereq_value <= value and prereq not in [a[0] for a in result]:
                        print(f"状态依赖关系：{attr}依赖于{prereq}，{prereq}的值为{prereq_value}，{attr}当前值为{value}")
                        result[i] = (prereq, prereq_value)
                        break

        return result

    def process_extra_attributes(self, character_state: Dict) -> str:
        """处理额外属性并生成指导字符串"""
        if not self.extra_attrs:
            return ""

        # 收集所有状态的extra_attr值
        attr_values = [
            (attr, state.get(self.extra_attrs[0], 0))
            for attr, state in character_state.items()
        ]
        extra_attr_name = self.attr_names.get(self.extra_attrs[0], self.extra_attrs[0])

        if not attr_values:
            return ""

        # 按值排序
        attr_values.sort(key=lambda x: x[1])
        non_zero_values = [x for x in attr_values if x[1] > 0]

        # 获取最低值的属性
        min_value = attr_values[0][1]
        min_value_attrs = [x for x in attr_values if x[1] == min_value]

        # 选择最低值属性
        if len(min_value_attrs) >= 4:
            lowest_three = random.sample(min_value_attrs, 3)
        else:
            lowest_five = attr_values[:min(5, len(attr_values))]
            lowest_three = random.sample(lowest_five, min(3, len(lowest_five)))

        # 检查依赖关系
        lowest_three = self.check_state_dependencies(lowest_three, attr_values)

        # 过滤掉已达到100的值，并获取最高的三个
        non_max_values = [v for v in non_zero_values if v[1] < 100]
        highest_three = non_max_values[-min(3, len(non_max_values)):] if non_max_values else []

        # 构建指导信息
        guidance_parts = []
        if lowest_three:
            lowest_str = ", ".join(
                f"{attr}({extra_attr_name}值:{val})"
                for attr, val in lowest_three
            )
            guidance_parts.append(
                f"选出的需要提高{extra_attr_name}的最低的三个属性是: {lowest_str}"
            )

        if highest_three:
            random_high = random.choice(highest_three)
            guidance_parts.append(
                f"选出的需要继续加强提高{extra_attr_name}的属性是: "
                f"{random_high[0]}({extra_attr_name}值:{random_high[1]})"
            )

        return "\n" + "\n".join(guidance_parts) if guidance_parts else ""

    def format_state_guidance(self, character_state: Dict) -> str:
        """格式化状态指导信息"""
        guidance = self.config.get('state_display', {}).get('guidance', [])
        guidance_str = "\n".join(guidance)

        if self.extra_attrs:
            guidance_str += self.process_extra_attributes(character_state)

        return guidance_str

    def format_extra_changes(self, attr: str) -> List[str]:
        """格式化指定属性的额外属性变化信息"""
        extra_changes = []
        if not self._changes or attr not in self._changes:
            print(f"状态变化日志：{attr}应该有变化，但实际没有变化")
            return extra_changes

        attr_changes = self._changes[attr]
        for extra_attr, values in attr_changes.items():
            from_value = values['from_value']
            to_value = values['to_value']

            if from_value == to_value:
                print(f"状态变化日志：{attr}的{extra_attr}从{int(from_value)}变为{int(to_value)}")
                continue

            display_name = self.attr_names.get(extra_attr, extra_attr)

            if from_value > 0:
                extra_changes.append(
                    f"{display_name}从{int(from_value)}变为{int(to_value)}"
                )
            elif to_value > 0:
                extra_changes.append(
                    f"{display_name}变为{int(to_value)}"
                )

        return extra_changes

    def process_state_update(self, attr: str, state: Dict, current_value: float = 0) -> Dict:
        """处理单个状态的额外属性更新"""
        if not self.has_extra_attrs():
            return state

        result = state.copy()
        for extra_attr in self.extra_attrs:
            if extra_attr in state:
                old_value = current_value
                new_value = self.calculate_new_value(
                    attr, extra_attr, old_value,
                    state[extra_attr],
                    state.get('from_state'),
                    state.get('to_state')
                )
                result[extra_attr] = new_value
                self._record_change(attr, extra_attr, old_value, new_value)

        return result

    def calculate_new_value(
        self,
        attr: str,
        extra_attr: str,
        old_value: float,
        target_value: float,
        from_state: str,
        to_state: str
    ) -> float:
        """计算新的属性值"""
        if not from_state or from_state == to_state:
            return max(min(target_value, 100), old_value)

        base_diff = self.attr_diffs.get(extra_attr, 1)
        diff = random.randint(base_diff, base_diff + 2)
        new_value = max(min(target_value, 100), old_value + diff)

        if new_value != target_value:
            print(
                f"状态变化特别处理日志："
                f"{attr}的{extra_attr}"
                f"本来要从{old_value}变成{target_value}，"
                f"实际变为{new_value}"
            )
        return new_value

    def _record_change(self, attr: str, extra_attr: str, from_value: float, to_value: float):
        """记录属性变化"""
        if attr not in self._changes:
            self._changes[attr] = {}
        self._changes[attr][extra_attr] = {
            'from_value': from_value,
            'to_value': to_value
        }

    def get_changes(self) -> Dict:
        """获取记录的变化"""
        return self._changes

    def clear_changes(self):
        """清除记录的变化"""
        self._changes = {}


class StateManager:
    """状态管理器"""
    def __init__(self, initial_state: Dict, config: Dict):
        self.state = deepcopy(initial_state)
        self.state_history = deepcopy(initial_state)
        self.config = config
        self.extra_attr_manager = ExtraAttributeManager(config)
        # self._extra_state_changes = {}

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

        # 处理状态更新，现在分为基础更新和额外更新
        self._update_character_state(
            character_state,
            updates.get('character_state', []),
            updates.get('extraUpdates', [])  # 新增参数
        )

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

    def _format_main_state_change(self, state: Dict) -> Optional[str]:
        """格式化主状态变化信息"""
        from_state = state.get('from_state')
        to_state = state['state']

        if from_state == to_state:
            return None

        if from_state and from_state != "无" and from_state != "暂无":
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
                extra_changes = self.extra_attr_manager.format_extra_changes(attr)
                if extra_changes:
                    state_msg_parts.append(f"({', '.join(extra_changes)})")

            # 添加完整的状态变化消息
            if state_msg_parts:
                messages.append(f"{attr}" + "".join(state_msg_parts))

        return "\n".join(messages) if messages else ""

    def get_state_str(self, ignore_blank: bool = False) -> str:
        """获取状态字符串"""
        inventory = self.state.get('inventory', {})
        character_state = self.state.get('character_state', {})

        # 如果ignore_blank为True，过滤掉空值
        if ignore_blank:
            character_state = {k: v for k, v in character_state.items() if v}

        # 从配置中获取模板和指导说明
        template = self.config.get('state_display', {}).get('template', '')
        guidance_str = self.extra_attr_manager.format_state_guidance(character_state)
        # 只保留基础状态
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

    def _update_character_state(self, character_state: Dict, updates: List[Dict], extra_updates: List[Dict] = None):
        """更新角色状态"""
        if updates:
            self.extra_attr_manager.clear_changes()

            # 检查是否有extra属性用于比较
            if extra_updates:
                extra_state_attributes = self.config.get("extra_state_attributes", [])
                extra_state_attribute = extra_state_attributes[0] if extra_state_attributes else None

            for state in updates:
                attr = state['attribute']
                to_state = state['state']

                # 确保属性存在
                if attr not in character_state:
                    character_state[attr] = {"state": ""}

                # 更新主状态
                character_state[attr]["state"] = to_state

                # 如果有额外属性更新，单独处理
                if attr in character_state:
                    if extra_updates.get('stateUpdates', []):
                        # 初始化update变量
                        update = None

                        # 查找匹配的更新
                        for extra_info in extra_updates.get('stateUpdates', []):
                            if extra_info.get('attribute') == attr:
                                update = extra_info
                                break

                        if update is not None:
                            # 获取当前值
                            current_value = character_state[attr].get(extra_state_attribute, 0)

                            # 更新新值
                            new_value = update.get(extra_state_attribute, current_value)

                            modified_new_value = self.extra_attr_manager.calculate_new_value(
                                attr,
                                extra_state_attribute,
                                current_value,
                                new_value,
                                character_state[attr].get('from_state'),
                                character_state[attr].get('state')
                            )

                            character_state[attr][extra_state_attribute] = modified_new_value
                            # 记录变化
                            self.extra_attr_manager._record_change(
                                attr,
                                extra_state_attribute,
                                current_value,
                                modified_new_value
                            )
