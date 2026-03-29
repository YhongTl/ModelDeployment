# 大模型私有化部署服务
🎯 面向个人开发者和中小团队的大模型私有化部署解决方案。  
一个支持 CLI / Web UI / API / UI+API 的大模型私有化部署框架（支持 vLLM 高性能推理）。  
## 📖 项目简介
本项目是一个基于私有大模型的多轮对话机器人服务部署框架，旨在为个人提供快速、稳定、可扩展的大模型私有化部署方案。项目从基础的函数式实现逐步演进，支持从最简原型 → 工程化重构 → 高性能推理（vLLM）的完整演进路径，最终提供完整的工业级部署方案。  
## ✨主要特性
### 🧠 部署能力
✅ 本地离线部署（无需外网）  
✅ 支持 CLI / UI / API / UI+API 四种运行模式  
✅ Docker 部署，提供完整的 Docker 部署方案，支持容器化运行  
### ⚡ 性能优化
✅ 集成 vLLM 推理引擎（吞吐提升 10~20x）  
✅ 支持 FP16 / 批处理 / 量化  
### 🔐 工程能力
✅ 多用户 API Key 认证 + 速率限制  
✅ 无状态设计（支持分布式扩展）  
✅ 日志系统，完善的日志记录和轮转功能  
✅ 配置灵活，支持命令行参数和配置文件双重配置   
### 🧩 体验增强
✅ 支持流式输出  
✅ 自动上下文裁剪  
## 📄版本说明
01 版	函数式编程范本，三种原始模式（cli/ui/api）  
02 版	类方法封装，四种模式（cli/ui/api/ui+api）  
03 版	vLLM 加速版本（需 Linux 环境）  
04 版	vLLM Docker API 版本（最终推荐）  
## 🛠️环境要求
Python 3.9+  
CUDA 11.8+（GPU 推理）  
Docker（可选，用于 vLLM 部署）  
至少 8GB 显存（推荐 16GB+）  
## 🔥基座模型
本项目仅提供原始代码，不包含具体模型，大家可根据需要自行选择模型，前往 Hugging Face 或 ModelScope 下载相应的开源模型即可。 
## 📌安装依赖
根据代码需要安装相应的依赖即可，建议提前做环境隔离。 
## 🚀基础使用
0. 启动参数说明：python main.py --mode ui+api --model-path /your model
1. 命令行模式：python vLLMDockerAPI_04.py --mode cli
2. Web UI 模式：python vLLMDockerAPI_04.py --mode ui   
3. API 服务模式：python vLLMDockerAPI_04.py --mode api   
4. UI + API 组合模式（推荐）：python vLLMDockerAPI_04.py --mode ui+api   
## 🐳 Docker 部署（vLLM）
### 1️⃣ 拉取镜像
docker pull vllm/vllm-openai:latest
### 2️⃣ 启动服务
docker run --gpus all -it --rm \  
  -v /your/model/path:/models \  
  -p 8001:8000 \  
  vllm/vllm-openai:latest \  
  --model /models/DeepSeek-R1 # 依据下方网址验证提供的模型id  
### 3️⃣ 验证
curl http://127.0.0.1:8001/v1/models  
## 📊性能优化
使用 vLLM 引擎：吞吐量提升 10-20 倍  
调整批处理大小：根据显存调整 --max-num-batched-tokens  
启用 FP16 混合精度：减少显存占用  
使用量化模型：如 GPTQ、AWQ 量化版本  
## 🔧故障排查
1.CUDA out of memory  
减小 max_context_length 和 max_new_tokens  
调整 vLLM 的 gpu_memory_utilization 参数  
2.vLLM 连接失败  
检查 vLLM 容器是否正常运行  
确认端口映射是否正确  
验证模型路径是否正确  
3.API Key 认证失败  
检查 api_config.json 格式是否正确  
确认请求头包含 Authorization: Bearer <key>  
## 📝日志查看
1.查看服务日志  
tail -f chatbot.log  
2.设置日志级别  
export LOG_LEVEL=DEBUG  
## 🤝 贡献指南
感谢你对本项目的关注！本项目致力于构建一个工程化、可扩展的大模型私有化部署框架，欢迎任何形式的贡献 ！  
🌐 贡献方式：  
🐛 提交 Bug（Issue）  
💡 提出新功能 / 改进建议  
🛠️ 提交代码（Pull Request）  
📚 完善文档（README / 注释 / 示例）  
⚡ 性能优化（推理 / 并发 / 显存）  
🔌 新功能扩展（如 RAG / 多模型 / 插件系统）  
## 📄 许可证
本项目采用 MIT 许可证，详见 LICENSE 文件。
## 📧 联系方式
项目主页：[GitHub (https://github.com/YhongTl)]  
作者邮箱：[175186639@qq.com]  
## 🙏 致谢
DeepSeek - 提供强大的基础模型  
vLLM - 高性能推理引擎  
Gradio - Web UI 框架  
FastAPI - API 框架  
## 由选手"躲后面抠脚"实力开发，持续迭代中, 后续功能敬请期待...
