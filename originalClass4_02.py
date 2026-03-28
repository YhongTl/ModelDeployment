"""
***大模型私有化部署服务---未使用vllm的四种部署模式运行代码，采用类方法封装，结构清晰，易于维护和扩展***
支持四种运行模式：cli / ui / api / ui+api
"""
import torch # PyTorch 深度学习框架
from transformers import AutoTokenizer, AutoModelForCausalLM # Hugging Face Transformers 库，用于加载预训练模型和 tokenizer
import secrets# 生成安全随机字符串
import json# 处理 JSON 文件
import os# 文件和路径操作
import re# 正则表达式
import argparse# 解析命令行参数
import threading# 线程锁
import time# 时间相关操作
import logging# 日志记录
from logging.handlers import RotatingFileHandler# 日志文件自动切割
from collections import defaultdict# 字典子类，提供默认值
from dataclasses import dataclass, field# 数据类
from pathlib import Path# 文件路径操作

# ================================================================
#                        配置类
# ================================================================
@dataclass
class Config:
    """全局配置类，集中管理所有可配置项"""
    # ---------- 模型相关 ----------
    model_path = Path(r"XXXXX")# 模型路径，支持绝对路径或相对路径，权重等文件所在地
    max_context_length: int = 10000# 模型最大上下文长度（包含输入和输出）
    max_new_tokens: int = 1024
    torch_dtype: str = "float16"  # float16 / bfloat16 / float32

    # ---------- 生成参数 ----------
    temperature: float = 0  # 设为 0 触发 do_sample=False，关闭采样，使用贪心解码，速度更快（但多样性下降）
    top_p: float = 0.9# nucleus 采样的累积概率阈值，配合 temperature > 0 使用
    top_k: int = 50# top-k 采样的 k 值，配合 temperature > 0 使用
    repetition_penalty: float = 1.0# 重复惩罚，默认 1.0 表示不使用，>1.0 会惩罚重复生成的 token

    # ---------- 服务相关 ----------
    host: str = "0.0.0.0"# 监听地址，
    api_port: int = 8000# API 服务端口
    ui_port: int = 7860# Gradio UI 服务端口
    uiapi_path: str = "/uiapi"  # Gradio 挂载路径（ui+api 模式）
    uvicorn_log_level: str = "warning"# Uvicorn 日志级别，生产环境建议 warning 或更高，开发调试可设为 info

    # ---------- API Key 相关 ----------
    api_key_config_path: str = "api_config.json"# API Key 配置文件路径
    default_user_count: int = 10# 默认生成用户数量（当配置文件不存在或加载失败时使用）
    default_rate_limit: int = 50  # 每分钟请求上限
    rate_limit_window: int = 60  # 速率限制窗口（秒）

    # ---------- 日志相关 ----------
    log_file: str = "chatbot.log"# 日志文件路径
    log_max_bytes: int = 10 * 1024 * 1024  # 日志文件最大大小（10 MB）
    log_backup_count: int = 5# 日志文件备份数量
    log_level: str = "INFO"# 日志级别，默认 INFO，生产环境建议 WARNING 或更高

    def get_torch_dtype(self):
        """将字符串转为 torch.dtype"""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.torch_dtype, torch.float16)

# ================================================================
#                        日志配置
# ================================================================
def setup_logger(config: Config) -> logging.Logger:
    """配置并返回全局 logger（防止重复添加 handler）"""
    logger = logging.getLogger("chatbot")
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, config.log_level.upper(), logging.INFO))
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    )
    # 控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    # 文件（自动切割）
    file_handler = RotatingFileHandler(
        config.log_file,
        maxBytes=config.log_max_bytes,
        backupCount=config.log_backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

# ================================================================
#                        速率限制器
# ================================================================
class RateLimiter:
    """基于滑动窗口的请求速率限制器"""
    def __init__(self, window: int = 60):
        self._window = window
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def check(self, user_id: str, limit: int) -> bool:
        """
        检查用户是否超限
        返回 True 表示允许，False 表示超限
        """
        now = time.time()
        with self._lock:
            # 清除过期记录
            self._requests[user_id] = [
                t for t in self._requests[user_id] if now - t < self._window
            ]
            if len(self._requests[user_id]) >= limit:
                return False
            self._requests[user_id].append(now)
            return True

# ================================================================
#                        ChatBot 核心类
# ================================================================
class ChatBot:
    """基于 DeepSeek-R1-Distill-Qwen-1.5B 的多轮对话机器人"""
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.model_path = config.model_path
        self._lock = threading.Lock()  # 推理锁，保证线程安全
        self.logger.info("正在从本地加载模型...")
        self.logger.info(f"模型路径: {self.model_path}")
        self.logger.info("正在加载 tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.logger.info("正在加载 model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=config.get_torch_dtype(),
            device_map="cuda:0" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        self.model.eval()
        # ★ 模型编译，只在这里执行一次
        if torch.cuda.is_available():
            try:
                self.model = torch.compile(self.model)
                self.logger.info("torch.compile 加速已启用 ✅")
            except Exception as e:
                self.logger.warning(f"torch.compile 不可用，跳过: {e}")
        self.logger.info("模型加载完毕 ✅")

    def _trim_context(
        self, history: list[dict]
    ) -> tuple[dict, list[dict]]:
        """
        根据 max_context_length 裁剪对话历史
        返回 (token_index, 裁剪后的 history)
        """
        trimmed = history.copy()
        format_chat = self.tokenizer.apply_chat_template(
            trimmed, tokenize=False, add_generation_prompt=True
        )
        token_index = self.tokenizer(format_chat, return_tensors="pt").to(
            self.model.device
        )
        while (
            token_index["input_ids"].shape[1] + self.config.max_new_tokens
            > self.config.max_context_length
            and len(trimmed) > 1
        ):
            trimmed.pop(0)
            # 如果裁剪后第一条是 assistant，也丢弃（保证 user 开头）
            if trimmed and trimmed[0]["role"] == "assistant":
                trimmed.pop(0)
            format_chat = self.tokenizer.apply_chat_template(
                trimmed, tokenize=False, add_generation_prompt=True
            )
            token_index = self.tokenizer(format_chat, return_tensors="pt").to(
                self.model.device
            )
            self.logger.warning(
                f"上下文过长，已丢弃最早的对话"
                f"（当前: {token_index['input_ids'].shape[1]} tokens）"
            )
        return token_index, trimmed

    def chat(
        self,
        user_input: str,
        history: list[dict] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> tuple[str, list[dict]]:
        """
        发送消息并获取回复
        返回 (response_text, updated_history)
        """
        if history is not None and not isinstance(history, list):
            raise ValueError("history must be a list or None")
        # 使用传入值或配置默认值
        temperature = temperature if temperature is not None else self.config.temperature
        top_p = top_p if top_p is not None else self.config.top_p
        history = history.copy() if history is not None else []
        history.append({"role": "user", "content": user_input})
        # 裁剪上下文
        token_index, trimmed_history = self._trim_context(history)
        # 构建生成参数
        generation_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": self.config.repetition_penalty,
        }
        if temperature > 0:
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": self.config.top_k,
                }
            )
        else:
            generation_kwargs["do_sample"] = False
        # 线程安全推理
        with self._lock:
            try:
                with torch.no_grad():
                    #  使用 autocast 混合精度推理，在 float16 基础上进一步提速
                    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                        outputs = self.model.generate(
                            **token_index,
                            **generation_kwargs,
                            # 以下两个参数可以提升生成速度
                            use_cache=True,          # 确保 KV Cache 开启（默认开启，显式声明）
                            num_beams=1,             # 关闭 beam search，使用贪心/采样（更快）
                        )
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        new_tokens = outputs[0][token_index["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        trimmed_history.append({"role": "assistant", "content": response})
        return response, trimmed_history

# ================================================================
#              工具函数：API Key 配置加载
# ================================================================
def generate_default_users(count: int = 10) -> dict:
    """生成指定数量的默认用户"""
    users = {}
    for i in range(1, count + 1):
        username = f"user{i}"
        users[username] = {
            "api_key": f"sk-default-{secrets.token_urlsafe(16)}",
            "name": f"默认用户{i}",
            "email": f"user{i}@example.com",
            "rate_limit": 50,
        }
    return users

def load_api_keys(
    config: Config, logger: logging.Logger
) -> tuple[dict, dict, dict]:
    """
    加载 API Key 配置
    返回 (users_config, api_keys, api_key_to_user 反向索引)
    """
    config_path = config.api_key_config_path
    users_config = {}
    api_keys = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                file_config = json.load(f)
            users_raw = file_config.get("users", {})
            # 兼容列表格式：[{"id": "xx", "api_key": "sk-xxx", ...}, ...]
            if isinstance(users_raw, list):
                users = {}
                for item in users_raw:
                    if not isinstance(item, dict):
                        logger.warning(f"用户项不是字典，已跳过: {item}")
                        continue
                    # 支持 "id" 或 "username" 作为用户标识键
                    uid = item.get("id") or item.get("username")
                    if not uid:
                        logger.warning(f"用户项缺少 'id' 或 'username' 字段，已跳过: {item}")
                        continue
                    if "api_key" not in item:
                        logger.warning(f"用户 '{uid}' 缺少 'api_key' 字段，已跳过")
                        continue
                    # 去掉 id/username 字段，其余全部保留
                    users[uid] = {k: v for k, v in item.items() if k not in ("id", "username")}
            elif isinstance(users_raw, dict):
                users = users_raw
            else:
                raise ValueError(f"users 字段类型不支持: {type(users_raw)}")    
                
                
            for user_id, user_data in users.items():
                if isinstance(user_data, dict) and "api_key" in user_data:
                    users_config[user_id] = user_data
                    api_keys[user_id] = user_data["api_key"]
                elif isinstance(user_data, str):
                    users_config[user_id] = {
                        "api_key": user_data,
                        "rate_limit": config.default_rate_limit,
                    }
                    api_keys[user_id] = user_data
            logger.info(f"从配置文件加载了 {len(api_keys)} 个用户 API Key")
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            users_config = generate_default_users(config.default_user_count)
            api_keys = {uid: ud["api_key"] for uid, ud in users_config.items()}
    else:
        users_config = generate_default_users(config.default_user_count)
        api_keys = {uid: ud["api_key"] for uid, ud in users_config.items()}
        logger.warning("未找到配置文件，已自动生成默认用户")
    # 建立反向索引：api_key -> user_id
    api_key_to_user = {v: k for k, v in api_keys.items()}
    return users_config, api_keys, api_key_to_user

# ================================================================
#              工具函数：响应解析（思考过程折叠）
# ================================================================
def parse_response(response: str) -> str:
    """将 DeepSeek-R1 的思考过程折叠显示"""
    think_pattern = re.compile(r"^(.*?)</think>(.*)$", re.DOTALL)
    def replace_think(match):
        think_content = match.group(1).strip()
        answer_content = match.group(2)
        if not think_content:
            return answer_content.strip()
        think_html = think_content.replace("\n", "<br>")
        return (
            f'<details style="background:#f0f0f0; padding:8px; '
            f'border-radius:6px; margin:6px 0;">'
            f'<summary style="cursor:pointer; color:#666;">'
            f"💭 <b>思考过程</b>（点击展开）</summary>"
            f'<div style="margin-top:6px; color:#444; font-size:0.9em;">'
            f"{think_html}"
            f"</div></details>"
            f"{answer_content}"
        )
    result = think_pattern.sub(replace_think, response)
    return result.strip()

# ================================================================
#                     工具函数：标准化聊天历史
# ================================================================
def normalize_history(chat_history: list) -> list[dict]:
    """将 Gradio 传来的 chat_history 标准化为 [{role, content}, ...] 格式"""
    if not isinstance(chat_history, list):
        return []
    normalized = []
    for item in chat_history:
        if isinstance(item, dict) and "role" in item and "content" in item:
            content = item["content"]
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            elif not isinstance(content, str):
                content = str(content)
            normalized.append({"role": item["role"], "content": content})
    return normalized

# ================================================================
#                 Gradio 界面构建(前后端一体或分离)
# ================================================================
def build_gradio_app(
    botagent: ChatBot,
    logger: logging.Logger,
    api_base_url: str | None = None,
):
    """
    构建 Gradio Blocks 实例
    参数：
        botagent:       ChatBot 实例
        logger:         日志实例
        api_base_url:   若为 None，直接调用模型（纯 UI 模式）
                        若有值，通过 HTTP 调用 FastAPI（UI+API 模式）
    """
    import gradio as gr
    import httpx

    def _call_via_api(
        base_url: str,
        message: str,
        history: list[dict],
        api_key: str,
    ) -> list[dict]:
        """通过 HTTP API 调用聊天接口"""
        url = f"{base_url}/api/chat"
        with httpx.Client(timeout=120) as client:
            resp = client.post(
                url,
                json={"message": message, "history": history},
                headers={"Authorization": f"Bearer {api_key}"},
            )
        if resp.status_code == 403:
            raise Exception("API Key 无效，请检查后重试")
        if resp.status_code == 429:
            raise Exception("请求过于频繁，请稍后再试")
        if resp.status_code != 200:
            raise Exception(f"服务端错误 (HTTP {resp.status_code})")
        data = resp.json()
        return data["history"]

    def _call_direct(
        message: str,
        history: list[dict],
    ) -> list[dict]:
        """直接调用模型（纯 UI 模式）"""
        _, updated_history = botagent.chat(message, history)
        return updated_history
    
    def save_api_key(api_key):
        api_key = api_key.strip()
        if not api_key:
            raise gr.Error("API Key不能为空")
        return gr.update(visible=False), api_key

    def gradio_chat(user_message: str, chat_history: list, api_key_input: str, api_key_saved: str):
        if not user_message or not user_message.strip():
            return "", chat_history or []
        normalized = normalize_history(chat_history)
        # 优先使用已保存的 api_key，否则使用当前输入框的值
        key_to_use = api_key_saved if api_key_saved else api_key_input
        try:
            if api_base_url:
                if not key_to_use:
                    raise Exception("请先输入 API Key")

                updated_history = _call_via_api(
                    api_base_url,
                    user_message,
                    normalized,
                    key_to_use.strip(),
                )
            else:
                updated_history = _call_direct(user_message, normalized)

        except Exception as e:
            logger.error(f"对话异常: {e}", exc_info=True)
            updated_history = normalized + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": f"⚠️ 错误: {str(e)}"},
            ]
        # 渲染（折叠思考过程）
        display_history = []
        for item in updated_history:
            content = (
                parse_response(item["content"])
                if item["role"] == "assistant"
                else item["content"]
            )
            display_history.append({"role": item["role"], "content": content})
        return "", display_history

    def clear_chat():
        logger.info("对话历史已清空 🔄")
        return "", []

    # ---------- 构建界面 ----------
    need_api_key = api_base_url is not None
    with gr.Blocks(title="DeepSeek ChatBot") as demo:
        # 定义一个状态，用于保存用户确认的API Key
        api_key_state = gr.State(value="")

        # ========= Sidebar =========
        with gr.Sidebar(position="left", open=True):
            gr.Markdown("## 🔑 鉴权认证")

            if need_api_key:
                api_key_input = gr.Textbox(
                    label="API Key",
                    placeholder="请输入 API Key（sk-xxx）",
                    type="password",
                )

                save_key_btn = gr.Button("确认 Key ✅")

            else:
                api_key_input = gr.Textbox(value="no-key-needed", visible=False)
                
        # ========= 主区域 =========
        with gr.Column():
            gr.HTML(
            """
<div style="text-align: center; margin: 0px 0;">
    <div style="font-weight: bold; font-size: 26px; color: #3b82f6;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.1);">
        基于 XXX 模型的私有化部署服务 
    </div>
    <div style="font-size: 14px; color: #999; margin-top: 10px;">
        🔧 由选手"躲后面抠脚"实力开发，后续功能敬请期待......
    </div>
</div>
"""
            )

            chatbot = gr.Chatbot(
                label="对话窗口",
                height=470,
                value=[],
                type="messages",
                allow_tags=False,     #  修复 allow_tags 警告
            )

            with gr.Row():
                user_input = gr.Textbox(
                    placeholder="请输入你的问题...",
                    scale=8,
                    show_label=False,
                )
                send_btn = gr.Button("发送 📤", scale=1, variant="primary")
            clear_btn = gr.Button("清空对话 🗑️")

        # =========================
        # 事件绑定
        # =========================

        # API Key 保存
        if need_api_key:
            save_key_btn.click(
                fn=save_api_key,
                inputs=[api_key_input],
                outputs=[api_key_input, api_key_state], 
            )

        # 聊天
        send_btn.click(
            fn=gradio_chat,
            inputs=[user_input, chatbot, api_key_input, api_key_state],
            outputs=[user_input, chatbot],
        )

        user_input.submit(
            fn=gradio_chat,
            inputs=[user_input, chatbot, api_key_input, api_key_state],
            outputs=[user_input, chatbot],
        )

        # 清空
        clear_btn.click(
            fn=clear_chat,
            outputs=[user_input, chatbot],
        )

    return demo

# ================================================================
#                    Cli命令行终端交互（纯后端）
# ================================================================

def run_cli(botagent: ChatBot, logger: logging.Logger):
    logger.info("=" * 50)
    logger.info("  命令行聊天模式（输入 quit 退出，输入 clear 清空历史）")
    logger.info("=" * 50)
    history = []
    while True:
        try:
            user_input = input("\n用户: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        if user_input.lower() == "clear":
            history.clear()
            print("对话历史已清空 🔄")
            continue
        if not user_input:
            print("请输入内容...")
            continue
        try:
            response, history = botagent.chat(user_input, history)
            print(f"助手: {response}")
        except Exception as e:
            logger.error(f"对话异常: {e}", exc_info=True)
            print(f"发生错误: {e}")
# ================================================================
#                     纯 Gradio UI（前后端全栈）
# ================================================================
def run_ui(botagent: ChatBot, config: Config, logger: logging.Logger):
    # 纯 UI 模式：api_base_url=None，直接调用模型
    demo = build_gradio_app(botagent, logger, api_base_url=None)
    logger.info(f"🌐 Web UI 启动中: http://{config.host}:{config.ui_port}")
    demo.launch(
        server_name=config.host,
        server_port=config.ui_port,
        share=False,
    )
# ================================================================
#              API 服务（后端，带 API Key 认证 + 速率限制）
# ================================================================
def run_api(botagent: ChatBot, config: Config, logger: logging.Logger):
    from fastapi import FastAPI, Depends, HTTPException, Security
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from pydantic import BaseModel
    from fastapi.concurrency import run_in_threadpool
    import uvicorn
    app = FastAPI(title="DeepSeek ChatBot API", version="1.0.0")

    users_config, api_keys, api_key_to_user = load_api_keys(config, logger)
    rate_limiter = RateLimiter(window=config.rate_limit_window)
    security_scheme = HTTPBearer(
        scheme_name="API Key",
        description="在此输入你的 API Key",
    )

    def verify_api_key(
        credentials: HTTPAuthorizationCredentials = Security(security_scheme),
    ) -> str:
        token = credentials.credentials
        candidate_user = api_key_to_user.get(token)
        if candidate_user is not None:
            stored_key = api_keys[candidate_user]
            if secrets.compare_digest(token, stored_key):
                # 速率限制检查
                user_info = users_config.get(candidate_user, {})
                limit = user_info.get("rate_limit", config.default_rate_limit)
                if not rate_limiter.check(candidate_user, limit):
                    raise HTTPException(
                        status_code=429,
                        detail=f"请求过于频繁，限制 {limit} 次/分钟",
                    )
                return candidate_user
        raise HTTPException(
            status_code=403,
            detail="无效的 API Key ❌",
            headers={"WWW-Authenticate": "Bearer"},
        )

    class ChatRequest(BaseModel):
        message: str
        history: list[dict] | None = None

    class ChatResponse(BaseModel):
        response: str
        history: list[dict]

    class HistoryResponse(BaseModel):
        status: str
        message: str

    @app.get("/")
    async def root():
        return {
            "service": "DeepSeek ChatBot API",
            "endpoints": {
                "POST /api/chat": "发送消息（需认证）",
                "POST /api/clear": "清空历史（需认证）",
                "GET /api/health": "健康检查",
            },
        }

    @app.get("/api/health")
    async def health_check():
        return {"status": "ok", "model": botagent.model_path}

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat_endpoint(
        req: ChatRequest, username: str = Depends(verify_api_key)
    ):
        user_info = users_config.get(username, {})
        user_name = user_info.get("name", username)
        logger.info(f"👤 用户 {user_name} 发送消息: {req.message[:50]}...")
        try:
            response, new_history = await run_in_threadpool(
                botagent.chat, req.message, req.history
            )
            return ChatResponse(response=response, history=new_history)
        except torch.cuda.OutOfMemoryError:
            logger.error("GPU 显存不足", exc_info=True)
            raise HTTPException(status_code=503, detail="服务器资源不足，请稍后重试")
        except Exception as e:
            logger.error(f"推理异常: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="内部服务错误")

    @app.post("/api/clear", response_model=HistoryResponse)
    async def clear_endpoint(username: str = Depends(verify_api_key)):
        user_info = users_config.get(username, {})
        user_name = user_info.get("name", username)
        logger.info(f"🗑️ 用户 {user_name} 清空了对话历史")
        return HistoryResponse(
            status="ok", message="请客户端丢弃本地 history"
        )

    logger.info(f"🚀 API 服务启动中: http://{config.host}:{config.api_port}")
    logger.info(f"📖 API 文档地址: http://{config.host}:{config.api_port}/docs")
    uvicorn.run(
        app,
        host=config.host,
        port=config.api_port,
        log_level=config.uvicorn_log_level,
    )
# ================================================================
#               UI + API 组合服务（前后端分离，单端口访问）
# ================================================================
def run_ui_api(botagent: ChatBot, config: Config, logger: logging.Logger):
    """
    Gradio UI + FastAPI API 同时运行在同一个端口
    访问方式：
    - 网页界面：http://host:port/ui
    - API 文档：http://host:port/docs
    - API 接口：POST http://host:port/api/chat（需 Bearer Token）
    """
    from fastapi import FastAPI, Depends, HTTPException, Security
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from pydantic import BaseModel
    from fastapi.concurrency import run_in_threadpool
    import uvicorn
    import gradio as gr
    # ========== 1. 创建 FastAPI 应用 ==========
    app = FastAPI(
        title="DeepSeek ChatBot - UI + API 组合服务",
        version="2.0.0",
        description="Gradio 网页界面 + RESTful API（带 API Key 认证 + 速率限制）",
    )
    # ========== 2. 加载 API Key ==========
    users_config, api_keys, api_key_to_user = load_api_keys(config, logger)
    rate_limiter = RateLimiter(window=config.rate_limit_window)
    logger.info(f"🔑 已加载 {len(api_keys)} 个用户 API Key")
    for username, user_info in users_config.items():
        name = user_info.get("name", username)
        key = user_info.get("api_key", "???")
        logger.info(f"  👤 {name} ({username}): {key}")
    # ========== 3. API Key 认证（带速率限制）==========
    security_scheme = HTTPBearer(
        scheme_name="API Key",
        description="在此输入你的 API Key",
    )
    def verify_api_key(
        credentials: HTTPAuthorizationCredentials = Security(security_scheme),
    ) -> str:
        token = credentials.credentials
        candidate_user = api_key_to_user.get(token)
        if candidate_user is not None:
            stored_key = api_keys[candidate_user]
            if secrets.compare_digest(token, stored_key):
                user_info = users_config.get(candidate_user, {})
                limit = user_info.get("rate_limit", config.default_rate_limit)
                if not rate_limiter.check(candidate_user, limit):
                    raise HTTPException(
                        status_code=429,
                        detail=f"请求过于频繁，限制 {limit} 次/分钟",
                    )
                return candidate_user
        raise HTTPException(
            status_code=403,
            detail="无效的 API Key ❌",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # ========== 4. 数据模型 ==========
    class ChatRequest(BaseModel):
        message: str
        history: list[dict] | None = None
    class ChatResponse(BaseModel):
        response: str
        history: list[dict]
    class HistoryResponse(BaseModel):
        status: str
        message: str
    # ========== 5. FastAPI 路由 ==========
    @app.get("/")
    async def root():
        return {
            "service": "DeepSeek ChatBot - UI + API 组合服务",
            "access": {
                "网页界面": f"http://{config.host}:{config.api_port}{config.uiapi_path}",
                "API 文档": f"http://{config.host}:{config.api_port}/docs",
                "聊天接口": f"POST http://{config.host}:{config.api_port}/api/chat",
                "清空历史": f"POST http://{config.host}:{config.api_port}/api/clear",
                "健康检查": f"GET http://{config.host}:{config.api_port}/api/health",
            },
            "users_count": len(api_keys),
        }
    @app.get("/api/health")
    async def health_check():
        return {
            "status": "ok",
            "model": botagent.model_path,
            "users_count": len(api_keys),
            "mode": "ui+api",
        }
    @app.post("/api/chat", response_model=ChatResponse)
    async def chat_endpoint(
        req: ChatRequest, username: str = Depends(verify_api_key)
    ):
        user_info = users_config.get(username, {})
        user_name = user_info.get("name", username)
        rate_limit = user_info.get("rate_limit", config.default_rate_limit)
        logger.info(
            f"🔌 [API] 用户 {user_name} ({username}) "
            f"发送消息: {req.message[:50]}... "
            f"[速率限制: {rate_limit}/分钟]"
        )
        try:
            response, new_history = await run_in_threadpool(
                botagent.chat, req.message, req.history
            )
            return ChatResponse(response=response, history=new_history)
        except torch.cuda.OutOfMemoryError:
            logger.error("GPU 显存不足", exc_info=True)
            raise HTTPException(
                status_code=503, detail="服务器资源不足，请稍后重试"
            )
        except Exception as e:
            logger.error(f"推理异常: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="内部服务错误")
    @app.post("/api/clear", response_model=HistoryResponse)
    async def clear_endpoint(username: str = Depends(verify_api_key)):
        user_info = users_config.get(username, {})
        user_name = user_info.get("name", username)
        logger.info(f"🗑️ [API] 用户 {user_name} 清空了对话历史")
        return HistoryResponse(
            status="ok", message="对话历史已清空，请客户端丢弃本地 history"
        )
    # ========== 6. 挂载 Gradio ==========
    # UI+API 模式：Gradio 通过 HTTP 调用本机 FastAPI 接口
    api_base_url = f"http://127.0.0.1:{config.api_port}"
    gradio_demo = build_gradio_app(botagent, logger, api_base_url=api_base_url)
    app = gr.mount_gradio_app(app, gradio_demo, path=config.uiapi_path)
    # ========== 7. 启动 ==========
    logger.info("=" * 60)
    logger.info("  🚀 UI + API 组合服务启动")
    logger.info("=" * 60)
    logger.info(f"  🌐 网页界面:   http://{config.host}:{config.api_port}{config.uiapi_path}")
    logger.info(f"  📖 API 文档:   http://{config.host}:{config.api_port}/docs")
    logger.info(f"  🔌 聊天接口:   POST http://{config.host}:{config.api_port}/api/chat")
    logger.info(f"  🗑️ 清空历史:   POST http://{config.host}:{config.api_port}/api/clear")
    logger.info(f"  💊 健康检查:   GET  http://{config.host}:{config.api_port}/api/health")
    logger.info(f"  🔑 已注册用户: {len(api_keys)} 个")
    logger.info("=" * 60)
    uvicorn.run(
        app,
        host=config.host,
        port=config.api_port,
        log_level=config.uvicorn_log_level,
    )

# ================================================================
#                           主入口
# ================================================================
def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="DeepSeek ChatBot 服务启动器")
    parser.add_argument(
        "--mode",
        type=str,
        default="ui+api",
        choices=["cli", "ui", "api", "ui+api"],
        help="运行模式: cli / ui / api / ui+api（默认: ui+api）",
    )
    parser.add_argument(
        "--host", type=str, default=None, help="服务监听地址（默认: 0.0.0.0）"
    )
    parser.add_argument(
        "--port", type=int, default=None, help="服务端口"
    )
    parser.add_argument(
        "--model-path", type=str, default=None, help="本地模型路径"
    )
    parser.add_argument(
        "--api-key-config", type=str, default=None, help="API Key 配置文件路径"
    )
    parser.add_argument(
        "--uiapi-path", type=str, default=None, help="Gradio UI 挂载路径（仅 ui+api 模式）"
    )
    parser.add_argument(
        "--temperature", type=float, default=None, help="生成温度（0 为贪心解码）"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=None, help="最大生成 token 数"
    )
    return parser.parse_args()

def build_config_from_args(args: argparse.Namespace) -> Config:
    """根据命令行参数覆盖默认配置"""
    config = Config()
    # 命令行参数优先级高于默认值（仅覆盖非 None 值）
    if args.model_path is not None:
        config.model_path = args.model_path
    if args.host is not None:
        config.host = args.host
    if args.api_key_config is not None:
        config.api_key_config_path = args.api_key_config
    if args.uiapi_path is not None:
        config.uiapi_path = args.uiapi_path
    if args.temperature is not None:
        config.temperature = args.temperature
    if args.max_new_tokens is not None:
        config.max_new_tokens = args.max_new_tokens
    # 端口处理：根据模式设置默认端口
    if args.port is not None:
        config.api_port = args.port
        config.ui_port = args.port
    else:
        if args.mode == "ui":
            config.ui_port = 7860
        elif args.mode in ("api", "ui+api"):
            config.api_port = 8000
    return config

if __name__ == "__main__":
    args = parse_args()
    config = build_config_from_args(args)
    logger = setup_logger(config)
    logger.info(f"运行模式: {args.mode}")
    logger.info(f"模型路径: {config.model_path}")
    logger.info(f"生成温度: {config.temperature}")
    logger.info(f"最大生成 tokens: {config.max_new_tokens}")

    # 实例化模型
    botagent = ChatBot(config=config, logger=logger)

    # 根据模式启动
    if args.mode == "cli":
        run_cli(botagent, logger)
    elif args.mode == "ui":
        run_ui(botagent, config, logger)
    elif args.mode == "api":
        run_api(botagent, config, logger)
    elif args.mode == "ui+api":
        run_ui_api(botagent, config, logger)
 