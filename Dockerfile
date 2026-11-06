# 使用官方 Python 3.11 精简镜像作为运行环境。
FROM python:3.11-slim

# 禁止 Python 生成 .pyc 文件，减少镜像和运行目录中的无用文件。
# 让 Python 日志直接输出到容器标准输出，便于 Docker 查看日志。
# 禁用 pip 缓存，减少镜像体积。
# 设置 Streamlit 为无头模式，不依赖浏览器或桌面环境。
# 监听所有网卡，使宿主机能够访问容器内的 Streamlit。
# 统一将 Streamlit 服务端口设置为 8888。
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8888

# 设置容器内的项目工作目录；后续相对路径都以此目录为基准。
WORKDIR /app

# 先只复制依赖清单，使源码变化时 Docker 可以复用依赖安装缓存。
COPY requirements.txt ./

# 升级 pip，并安装项目运行所需的 Python 依赖。
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

# 创建不能登录的普通用户，避免应用进程以 root 权限运行。
# 创建运行时数据目录和 Streamlit 配置目录。
# 将整个应用目录交给普通用户，确保应用可以写入运行所需目录。
RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/data /app/.streamlit \
    && chown -R appuser:appuser /app

# 将项目源码复制到镜像，并将文件所有者设置为 appuser。
COPY --chown=appuser:appuser . .

# 后续命令及容器主进程均使用普通用户执行。
USER appuser

# 声明容器对外提供的服务端口；实际映射由 docker run -p 决定。
EXPOSE 8888

# 定期访问 Streamlit 健康检查接口。
# 如果请求失败，Docker 会将容器标记为 unhealthy。
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8888/_stcore/health', timeout=3)"

# 以模块方式启动 Streamlit，运行项目入口 app.py。
# 显式指定监听地址和端口，确保容器外部可以访问应用。
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8888"]
