# 基础镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    coturn \
    redis-server \
    && rm -rf /var/lib/apt/lists/*

# 设置默认源（国内源）
ARG USE_CN_SOURCE=true

# 根据环境变量选择是否使用国内源
RUN if [ "$USE_CN_SOURCE" = "true" ]; then \
        pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple; \
    else \
        pip config set global.index-url https://pypi.org/simple; \
    fi

# 复制 requirements.txt 并安装依赖（利用 Docker 缓存）
COPY requirements.txt /app/
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install fastapi uvicorn langchain_core langchain_openai langchain langchain_community openai redis google-search-results nltk

# 如果有 nltk 数据，可提前复制
COPY nltk_data /app/nltk_data
ENV NLTK_DATA=/app/nltk_data

# 复制代码
COPY . /app

# 复制配置文件
COPY turnserver.conf /etc/turnserver.conf
COPY redis.conf /etc/redis/redis.conf

# 设置数据卷
VOLUME /data

# 暴露端口
EXPOSE 10082 3478 6379

# 启动服务
CMD ["sh", "-c", "turnserver -c /etc/turnserver.conf --listening-ip=0.0.0.0 --listening-port=3478 & redis-server /etc/redis/redis.conf --protected-mode no & sleep 3 && uvicorn server:app --host 0.0.0.0 --port 10082"]
