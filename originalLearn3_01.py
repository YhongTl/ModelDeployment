"""
***大模型私有化部署服务---原始代码，函数式编程版本，三种原始模式是学习范本***
支持三种运行模式：cli / ui / api 
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import secrets  # 用于安全比较 API Key，安全随机数生成
class ChatBot:
    """基于 XXX 模型的多轮对话机器人"""
    # -------- 1: 定义属性 → 参数属性与状态属性 --------
    def __init__(
        self,
        cache_dir: str = r"XXXXX",  # 模型路径，支持绝对路径或相对路径，权重等文件所在地
        max_context_length: int = 32768,
        max_new_tokens: int = 1024,
    ):
        # -------- 1.2: 全局变量 → 实例属性 --------
        self.max_context_length = max_context_length
        self.max_new_tokens = max_new_tokens
        

        # -------- 1.3: 加载全部封装到构造函数 --------
        print("正在从本地加载模型...")
        self.model_path = cache_dir  # 直接使用本地路径，无需下载
        print(f"模型路径: {self.model_path}")

        print("正在加载 tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )

        print("正在加载 model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        self.model.eval()  # 关闭 dropout 等训练专用层
        print("模型加载完毕 ✅")

    # -------- 2: 上下文裁剪逻辑抽离为私有方法 --------
    def _trim_context(self, token_index, history):
        """
        裁剪上下文，避免窗口超限，操作传入的 history 而非实例历史
        """
        input_length = token_index["input_ids"].shape[1]
        # 临时复制 history 来修改
        temp_history = history.copy()
        while (
            input_length + self.max_new_tokens > self.max_context_length
            and len(temp_history) > 1
            ): 
            # 删除最早的一轮对话（一问一答 = 2条记录）
            temp_history.pop(0)  # 删最早的 user
            if temp_history and temp_history[0]["role"] == "assistant":
                temp_history.pop(0)  # 删对应的 assistant

            # 重新格式化和编码
            format_chat = self.tokenizer.apply_chat_template(
                temp_history, tokenize=False, add_generation_prompt=True
            )
            token_index = self.tokenizer(format_chat, return_tensors="pt").to(
                self.model.device
            )
            input_length = token_index["input_ids"].shape[1]
            print(f"⚠️ 上下文过长，已丢弃最早的对话（当前: {input_length} tokens）")

        return token_index, temp_history

    # -------- 3: 全局函数chatbot→ 实例方法 chat --------
    def chat(self, user_input: str, history: list[dict] | None = None) -> tuple[str, list[dict]]:
        """
        单轮交互：接收用户输入和历史，返回模型回复与新历史。
        history参数为对话上下文历史（无状态传递）
        """
        # 类型断言，防止异常输入（改进1）
        if history is not None and not isinstance(history, list):
            raise ValueError("history must be a list or None")
        # 使用传入的历史或空列表
        history = history.copy() if history is not None else []
        # 1. 预处理
        history.append({"role": "user", "content": user_input}) # 添加用户输入
        # 格式化对话
        format_chat = self.tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
        )
        token_index = self.tokenizer(format_chat, return_tensors="pt").to(self.model.device)

        # 上下文长度检查（调用私有方法）
        token_index, trimmed_history = self._trim_context(token_index, history) # 用trimmed_history替换history向后传递

        # 2. 推理
        with torch.no_grad():
            outputs = self.model.generate(
                **token_index,
                max_new_tokens=self.max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # 3. 后处理
        new_tokens = outputs[0][token_index["input_ids"].shape[1] :] # 只取新生成的部分
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True) # 解码生成内容
         # 添加模型回复到历史
        trimmed_history.append({"role": "assistant", "content": response})

        return response, trimmed_history

    # -------- 4: 新增清空历史的便捷方法 --------
    #def clear_history(self):
        # 修改点：此处移除对实例历史的清空操作，保持无状态设计
        #print("对话历史已清空 ")
        # 注意真正清空历史的操作由调用代码负责


# ================================================================
#                服务模式 1：命令行终端交互（后端应用）
# ================================================================
def run_cli(botagent: ChatBot):
    """命令行交互循环"""
    print("=" * 50)
    print("  命令行聊天模式（输入 quit 退出，输入 clear 清空历史）")
    print("=" * 50)
    history = []  # 修正：本地维护对话历史，传入chat方法，避免无上下文问题
    while True:
        user_input = input("\n用户: ").strip()

        if user_input.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        if user_input.lower() == "clear":
            history.clear()  # 同时清空本地历史
            print("对话历史已清空 🔄")  # 提示用户清空成功
            continue

        if not user_input:
            print("请输入内容...")
            continue
        try:
            response, history = botagent.chat(user_input, history)
            print(f"助手: {response}")
        except Exception as e:
            print(f"发生错误: {e}")

# ================================================================
#            服务模式 2：Gradio Web UI（前后端一体全栈应用）
# ================================================================
def run_ui(botagent: ChatBot, server_name: str = "0.0.0.0", server_port: int = 7860):
    import gradio as gr
    import re
    def parse_response(response: str) -> str:
        """
        将 DeepSeek-R1 的思考过程折叠显示
        把<details type="reasoning" done="true" duration="0">
        <summary>Thought for 0 seconds</summary>&gt; ...</details>转换为 HTML 折叠标签
        """
        # 匹配 ...</think> 标签（支持多行）
        think_pattern = re.compile(r"^(.*?)</think>(.*)$", re.DOTALL)
        
        def replace_think(match):
            think_content = match.group(1).strip()
            answer_content = match.group(2)
            if not think_content:
                return ""
            # 将换行转为 <br> 以便 HTML 正确显示
            think_html = think_content.replace("\n", "<br>")
            return (
                f'<details style="background:#f0f0f0; padding:8px; '
                f'border-radius:6px; margin:6px 0;">'
                f'<summary style="cursor:pointer; color:#666;">💭 <b>思考过程</b>（点击展开）</summary>'
                f'<div style="margin-top:6px; color:#444; font-size:0.9em;">'
                f'{think_html}'
                f'</div></details>'
                f'{answer_content}'
            )
        
        result = think_pattern.sub(replace_think, response)
        return result.strip()
    
    def gradio_chat(user_message, chat_history):
        """
        Gradio Chatbot 组件的回调函数
        chat_history 格式: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
        """
        # 防御性处理：确保 chat_history 是列表
        if not isinstance(chat_history, list): # 类型检查
            chat_history = []
        # 同步 Gradio 历史 → bot（确保 content 是字符串）
        # 将历史内容标准化，不再使用实例变量，符合无状态设计
        normalized_history = []
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
                normalized_history.append({"role": item["role"], "content": content})
        # 调用模型，传入历史
        try:
            response, updated_history = botagent.chat(user_message, normalized_history)
        except Exception as e:
            error_msg = f"发生错误: {e}"
            updated_history = normalized_history + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": error_msg},
    ]
        response, updated_history = botagent.chat(user_message, normalized_history)  # 修正：传入历史，接收更新的历史
        chat_history = updated_history  # 关键改进：使用 updated_history 替换 chat_history，保持上下文最新
        # 将显示的对话内容加工处理后，作为 Gradio 输出
        chat_history_display = []
        for item in chat_history:
            # 仅格式化assistant内容，避免对用户输入格式化
            if item["role"] == "assistant":
                content = parse_response(item["content"])
            else:
                content = item["content"]
            chat_history_display.append({"role": item["role"], "content": content})

        return "", chat_history_display

    def clear_chat():
        # 修改点：不调用 botagent.clear_history()
        # 直接在 UI 层清空聊天记录
        print("对话历史已清空 🔄")
        return "", []

    # 构建界面
    with gr.Blocks(title="DeepSeek ChatBot") as demo:
        gr.HTML('''
<div style="text-align: center; margin: 0px 0;">
    <div style="font-weight: bold; font-size: 26px; color: #3b82f6; 
                text-shadow: 2px 2px 4px rgba(0,0,0,0.1);">
        基于 XXX 模型的私有化部署服务
    </div>
    <div style="font-size: 14px; color: #999; margin-top: 10px;">
        🔧 由选手“躲后面抠脚”实力开发，后续功能敬请期待......
    </div>
</div>
''')
        chatbot = gr.Chatbot(
            label="对话窗口",
            height=470,
            value=[],
            sanitize_html=False,  # ← 允许 HTML 渲染（折叠标签需要）
        )

        with gr.Row():
            user_input = gr.Textbox(
                label="输入消息",
                placeholder="请输入你的疑问",
                scale=8,
                show_label=False,
            )
            send_btn = gr.Button("发送 📤", scale=1, variant="primary")

        clear_btn = gr.Button("清空对话 🗑️")

        # 绑定事件
        send_btn.click(
            fn=gradio_chat,
            inputs=[user_input, chatbot],
            outputs=[user_input, chatbot],
        )
        user_input.submit(
            fn=gradio_chat,
            inputs=[user_input, chatbot],
            outputs=[user_input, chatbot],
        )
        clear_btn.click(
            fn=clear_chat,
            inputs=[],
            outputs=[user_input, chatbot],
        )

    print(f"🌐 Web UI 启动中: http://{server_name}:{server_port}")
        # 添加用户名密码认证
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        auth=("admin", "123456"),  # 添加这行
        auth_message="请输入用户名和密码"  # 可选：提示信息
    )
# ================================================================
#               服务模式 3：FastAPI Web API（后端应用）
# ================================================================
def run_api(botagent: ChatBot, host: str = "0.0.0.0", port: int = 8000, api_key_config: str | None = None): # 属于：手动依赖注入函数，botagent 是通过参数注入的依赖
    """
    基于 FastAPI + Uvicorn 的 RESTful API 服务
    提供以下接口：
    - POST /chat: 接收用户消息和历史，返回模型回复和新历史
    - POST /clear: 清空对话历史（无状态设计下提示客户端丢弃历史）
    - GET /health: 健康检查接口，返回服务状态
    """
    from fastapi import FastAPI, Depends, HTTPException, Security  # 导入 Depends, HTTPException, Security
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials  # 导入 Bearer Token 安全方案
    from pydantic import BaseModel
    from fastapi.concurrency import run_in_threadpool
    import uvicorn
    import json
    import os
    # 属于：应用结构定义部分，包含格式定义（请求/响应模型）和结构定义（路由）
    app = FastAPI(title="DeepSeek ChatBot API", version="1.0.0")# 属于：基础配置（硬编码）
    
    # ==========================================
    # 多用户 API Key 配置文件加载 - 开始
    # ==========================================

    # 确定配置文件路径
    config_path = api_key_config or "api_config.json"

    print(f"🔍 调试信息:")
    print(f"   配置文件路径: {config_path}")
    print(f"   当前工作目录: {os.getcwd()}")
    print(f"   文件是否存在: {os.path.exists(config_path)}")

    users_config = {}

    # 加载 API Key 配置
    api_keys = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                users = config.get('users', {})
                
                # 保存完整的用户配置信息
                for user_id, user_data in users.items():
                    if isinstance(user_data, dict) and 'api_key' in user_data:
                        users_config[user_id] = user_data
                        api_keys[user_id] = user_data['api_key']
                    elif isinstance(user_data, str):
                        # 兼容简化格式
                        users_config[user_id] = {"api_key": user_data}
                        api_keys[user_id] = user_data
            
            print(f"✅ 自动加载配置文件: {config_path}")    
            print(f"📁 从配置文件加载了 {len(api_keys)} 个用户 API Key")
            
            # 打印用户详情（包括速率限制）
            print("🔑 用户配置详情:")
            for username, user_info in users_config.items():
                rate_limit = user_info.get('rate_limit', '默认')
                name = user_info.get('name', username)
                print(f"  👤 {name} ({username}): 速率限制={rate_limit}/分钟")
        except Exception as e:
            print(f"❌ 配置文件加载失败: {e}")
            # 同时设置 users_config 和 api_keys
            users_config = generate_default_users(10)
            api_keys = {user_id: user_data['api_key'] for user_id, user_data in users_config.items()}
    else:
        # 【添加这行调试信息】
        print(f"❌ 文件不存在，完整路径: {os.path.abspath(config_path)}")
        # 同时设置 users_config 和 api_keys
        users_config = generate_default_users(10)
        api_keys = {user_id: user_data['api_key'] for user_id, user_data in users_config.items()}
        print("⚠️  未找到配置文件，已自动生成10个默认用户")
    
    # 打印所有用户和对应的 API Key
    print("🔑 可用用户数量:", len(api_keys))
    print("📋 请求时在 Header 中添加: Authorization: Bearer <your-key>")

    # 定义 Bearer Token 安全方案
    security_scheme = HTTPBearer(
        scheme_name="API Key",
        description="在此输入你的 API Key",
    )

    def verify_api_key(
        credentials: HTTPAuthorizationCredentials = Security(security_scheme),
    ) -> str:
        """
        验证多个 API Key 中的任意一个
        """
        # 安全地比较输入的 Key 是否在允许的列表中
        for user_id, api_key in api_keys.items():
            if secrets.compare_digest(credentials.credentials, api_key):
                return user_id  # 返回用户名，用于后续处理
        
        raise HTTPException(
            status_code=403,
            detail="无效的 API Key ❌",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # ==========================================
    # 多用户 API Key 配置文件加载 - 结束
    # ==========================================

    # ---------- 格式定义部分 (Format)，请求/响应模型 ----------
    class ChatRequest(BaseModel): # 请求模型，包含用户消息和可选的历史记录
        message: str
        history: list[dict] | None = None  # 可选：客户端传入历史

    class ChatResponse(BaseModel): # 响应模型，包含模型回复和当前历史记录
        response: str
        history: list[dict]

    class HistoryResponse(BaseModel): # 历史清空响应模型，包含状态和消息
        status: str
        message: str

    # ---------- 结构定义部分 (Structure)，路由 ----------
    @app.get("/")
    async def root():
        return {
            "service": "DeepSeek ChatBot API",
            "auth": "请在请求头中添加 Authorization: Bearer <your-api-key>",
            "users_count": len(api_keys),  # 显示用户数量
            "endpoints": {
                "POST /chat": "发送消息并获取回复（需认证）",
                "POST /clear": "清空对话历史（需认证）",
                "GET /health": "健康检查（公开）",
            },
        }
    @app.get("/health")
    async def health_check():
        return {"status": "ok", "model": botagent.model_path,"users_count": len(api_keys)} # 在路由中使用注入的对象
    # 修改 /chat 路由，去除修改实例属性，改用无状态方式 ---添加 dependencies 参数要求 API Key 认证
    @app.post("/chat", response_model=ChatResponse)
    async def chat_endpoint(req: ChatRequest, username: str = Depends(verify_api_key)):
        try:
            # 获取用户配置信息
            user_info = users_config.get(username, {})
            user_name = user_info.get('name', username)
            rate_limit = user_info.get('rate_limit', 50)  # 默认50
            print(f"👤 用户 {user_name} ({username}) 发送消息: {req.message[:50]}...")
            print(f"📊 用户速率限制: {rate_limit}/分钟")
            # 传入请求的历史，返回 response 和新历史
            response, new_history = await run_in_threadpool(botagent.chat, req.message, req.history)
            return ChatResponse(response=response, history=new_history)
        except Exception as e:
        # 简单异常处理，实际可更详细
            return ChatResponse(response=f"内部错误: {str(e)}", history=req.history or [])

    #  clear_history 路由依然调用实例方法清空实例历史，或改为空操作 ---
    @app.post("/clear", response_model=HistoryResponse)
    async def clear_endpoint(username: str = Depends(verify_api_key)):
    # 无状态设计下，服务端不持有历史
    # 客户端只需自行丢弃本地 history 即可
        user_info = users_config.get(username, {})
        user_name = user_info.get('name', username)
        print(f"🗑️ 用户 {user_name} 清空了对话历史")
        return HistoryResponse(status="ok", message="对话历史已清空，请客户端丢弃本地 history")
    # 端口接收部分 (Port Binding)属于：服务启动和端口绑定
    print(f"🚀 API 服务启动中: http://{host}:{port}") # CND Content Delivery Network（内容分发网络），属于内容资源缓存储存服务器，只上传一次所有用户都能从距自己最近点的cdn服务器得到源站的资源。
    print(f"📖 API 文档地址: http://{host}:{port}/docs") # pip install fastapi-cdn-host，文档配置文件请求网站在国外故打开慢，采用此方式建立
    uvicorn.run(app, host=host, port=port)

# 生成带完整信息的默认用户
def generate_default_users(count: int = 10) -> dict:
    """生成指定数量的默认用户（包含完整信息）"""
    users = {}
    for i in range(1, count + 1):
        username = f"user{i}"
        users[username] = {
            "api_key": f"sk-default-{secrets.token_urlsafe(16)}",
            "name": f"默认用户{i}",
            "email": f"user{i}@example.com",
            "rate_limit": 50  # 默认速率限制
        }
    return users
   
# ================================================================
#                           主入口
# ================================================================
import argparse

if __name__ == "__main__": # 主函数入口，当前文件作为主程序允许
    parsertool = argparse.ArgumentParser(description="DeepSeek ChatBot 服务启动器") # 创建命令行参数解析器
    parsertool.add_argument(
        "--mode",
        type=str,
        default="ui",
        choices=["cli", "ui", "api"],
        help="运行模式: cli(命令行), WebUI(Gradio界面), API(FastAPI接口)  默认: api",
    )
    parsertool.add_argument("--host", type=str, default="0.0.0.0", help="服务监听地址 (ui/api模式)")
    # 定义了一个命令行选项--host，在命令行参数中可以指定服务监听的地址ip，默认为0000
    parsertool.add_argument("--port", type=int, default=None, help="服务端口 (ui默认7860, api默认8000)")
    parsertool.add_argument(
        "--model-path",
        type=str,
        default=r"D:\VScode\neural network\Deployment\model\modelscope\DeepSeek-R1-Distill-Qwen-1.5B",
        help="本地模型路径",
    )
    # 参数名和帮助信息
    parsertool.add_argument(
        "--api-key-config", type=str, default=None,  # 参数名改为 --api-key-config
        help="API Key 配置文件路径，支持多用户配置",
    )

    args = parsertool.parse_args() # 命令行参数解析函数

    # 实例化模型（三种模式共用同一个 bot）
    botagent = ChatBot(cache_dir=args.model_path)

    # 根据 --mode 参数选择运行模式，只会启动其中一种
    if args.mode == "cli":
        run_cli(botagent)

    elif args.mode == "ui":
        port = args.port if args.port else 7860
        run_ui(botagent, server_name=args.host, server_port=port)

    elif args.mode == "api":
        port = args.port if args.port else 8000
        run_api(botagent, host=args.host, port=port, api_key_config=args.api_key_config)