"管理状态变化"
from typing import Dict, List
from copy import deepcopy
# pylint: disable=too-few-public-methods


class StateManager:
    """状态管理器"""
    def __init__(self, initial_state: Dict, config: Dict):
        self.state = deepcopy(initial_state)
        self.state_history = deepcopy(initial_state)
        self.config = config

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
        print('test_begin', self.state_history)

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
        print('test_end', self.state_history)

        return final_state_str, clean_updates

    def format_state_changes(self, updates: Dict) -> str:
        """格式化输出的更新"""
        messages = []

        # Format item changes
        for item in updates.get('itemUpdates', []):
            name = item['name']
            change = item['change_amount']
            if change > 0:
                messages.append(f"获得了 {name} x{change}")
            elif change < 0:
                messages.append(f"失去了 {name} x{abs(change)}")
        # Format state changes
        for state in updates.get('stateUpdates', []):
            attr = state['attribute']
            from_state = state.get('from_state')
            to_state = state['to_state']
            if from_state and from_state != "无" and from_state != to_state:
                messages.append(f"{attr}从「{from_state}」变成了「{to_state}」")
            elif to_state:
                messages.append(f"{attr}变成了「{to_state}」")
            else:
                print(f"状态变化日志：{attr}从「{from_state}」变成了「{to_state}」")

        if not messages:
            return ""

        return "\n".join(messages)

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
        guidance_str = " ".join(guidance)

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
        for state in updates:
            attr = state['attribute']
            character_state[attr] = state['to_state']
