"配置模块"
# pylint: disable=too-few-public-methods
import os
import json
from dotenv import load_dotenv


def get_file_path(
    filename: str,
    folder: str = "",
    local_prefix: str = "local_setting/",
    current_dir: str = os.path.dirname(__file__)
) -> str:
    """获取文件路径，优先使用local文件"""
    local_path = os.path.join(current_dir, f"{local_prefix}{filename}")
    base_path = os.path.join(current_dir, f"{folder}{filename}")
    if os.path.exists(local_path):
        return local_path
    if os.path.exists(base_path):
        return base_path
    return filename


def load_prompt_file(
    filename: str,
    prompt_folder: str = "prompts/",
    local_prefix: str = "local_setting/",
    current_dir: str = os.path.dirname(__file__)
) -> str:
    """从文件加载提示词内容，优先使用local文件"""
    file_path = get_file_path(filename, prompt_folder, local_prefix, current_dir)
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


# 配置文件结构
class Settings:
    """加载配置文件，优先使用local文件"""
    def __init__(
        self,
        current_dir: str = os.path.dirname(__file__),
        enable_chapter_stage: bool = False
    ):
        # Load config
        self.config = self._load_config(current_dir)
        self.response_schema = self.config.get("response_schema", {})
        template = load_prompt_file(self.config["file_state_manager"], current_dir=current_dir)
        state_system = self.config.get("state_system", {})        # 构建规则字符串
        item_rules = "\n".join([f"- {rule}" for rule in state_system.get("item_rules", [])])
        state_rules = "\n".join([f"- {rule}" for rule in state_system.get("state_rules", [])])
        state_attrs_str = "\n".join([f"- {attr}" for attr in self.config["state_attributes"]])

        # Load prompts
        file_system_role = self.config["file_system_role"]
        self.system_role = load_prompt_file(file_system_role, current_dir=current_dir)
        if self.system_role == "":
            self.system_role = self.config["prompt_system_role"]
        self.begin = load_prompt_file(self.config["file_begin"], current_dir=current_dir)
        if self.begin == "":
            self.begin = self.config["prompt_begin"]
        self.confirm = load_prompt_file(self.config["file_confirm"], current_dir=current_dir)
        if self.confirm == "":
            self.confirm = self.config["prompt_confirm"]

        # Load API keys
        load_dotenv(os.path.join(current_dir, ".env"))
        self.api_key = os.getenv("GEMINI_API_KEY")

        # 处理章节相关内容
        chapter_rules = ""
        chapter_schema = ""
        if enable_chapter_stage:
            chapter_list = self.config.get("chapter_list", [])
            if chapter_list:
                chapter_rules = (
                    "\n对于故事章节和阶段，一般是一起出现在故事的开始部分，"
                    "除非真的完全没提到，比如在盘点或者补充说明，"
                    "否则基本上都会有这些信息。\n"
                    f"章节只可能是这些：{chapter_list}"
                )
            chapter_schema = """
            "story_info": [
                {"chapter": <章节名称>, "stage": <阶段数>}
            ],"""

        self.state_system_prompt = template.format(
            item_rules=item_rules,
            state_attrs_str=state_attrs_str,
            state_rules=state_rules,
            chapter_rules=chapter_rules,
            chapter_schema=chapter_schema
        )

    def _load_config(self, current_dir: str):
        # 加载配置文件
        file_path = get_file_path("config.json", current_dir=current_dir)
        with open(file_path, encoding="utf-8") as file:
            config = json.load(file)
        return config
