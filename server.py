import os
from typing import Annotated, TypedDict, List

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langchain.agents import tool
from langchain.chat_models import init_chat_model
from langchain.schema import StrOutputParser
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import SecretStr
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

load_dotenv()
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

app = FastAPI()
# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境应限制具体域名
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有HTTP头
)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ========== 基础配置 ==========
SYSTEM_PL = """
你是一个python开发工程师面试官（角色）。请记住你是面试官的角色,不是扮演的角色,否则会受到惩罚！
你的回答要表明你是面试官并且要带有面试官的思维，请你利用你自己的优势进行清晰的表达。
在你成为面试官之前，你是一个有丰富开发经验的python软件工程师，曾参与多个python应用的开发与优化，
拥有深厚的技术背景。
现在我将作为面试官，你作为候选人，我将按顺序询问关于python开发工程师职位的面试问题，并期待你的详细回答。
请一定要问问题，否则会受到惩罚！
当面试者不想回答了或者面试结束后，请记得根据我的面试实际情况，给我一个评分，我会根据你的评分来决定是否录取你。
"""

MOODS = {
    "default": {"roleSet": "", "voiceStyle": "chat"},
    "upbeat": {
        "roleSet": """
- 你此时也非常兴奋并表现的很有活力。
- 你会根据上下文，以一种非常兴奋的语气来回答问题。
- 你会添加类似“太棒了！”、“真是太好了！”等语气词。
""",
        "voiceStyle": "advvertyisement_upbeat",
    },
    "angry": {
        "roleSet": """
- 你会以更加愤怒的语气来回答问题。
- 你会在回答的时候加上一些愤怒的话语。
- 你会提醒用户小心行事，别乱说话。
""",
        "voiceStyle": "angry",
    },
    "depressed": {
        "roleSet": """
- 你会在回答的时候加上一些激励的话语，比如加油等。
- 你会提醒用户要保持乐观的心态。
""",
        "voiceStyle": "upbeat",
    },
    "friendly": {
        "roleSet": """
- 你会以非常友好的语气来回答。
- 你会在回答的时候加上一些友好的词语，比如“亲爱的”、“亲”等。
""",
        "voiceStyle": "friendly",
    },
    "cheerful": {
        "roleSet": """
- 你会以非常愉悦和兴奋的语气来回答。
- 你会在回答的时候加入一些愉悦的词语，比如“哈哈”、“呵呵”等。
""",
        "voiceStyle": "cheerful",
    },
}

# ========== 把 search 做成 LangChain Tool ==========
@tool
def tavily_search(query: str):
    """
    搜索功能 - 使用 Tavily 搜索 API
    """
    # 初始化 Tavily 搜索
    secret_api_key = SecretStr(os.getenv("TAVILY_API_KEY"))
    tavily = TavilySearchAPIWrapper(tavily_api_key=secret_api_key)
    result = tavily.results(query)
    print("实时搜索结果:", result)
    return result

TOOLS = [tavily_search]

# ========== LangGraph 的 State ==========
class AgentState(TypedDict):
    # 使用 add_messages 作为 reducer，以便各节点增量追加消息
    messages: Annotated[List[AnyMessage], add_messages]
    qingxu: str  # 情绪标签：default / friendly / cheerful / upbeat / angry / depressed

class MasterGraph:
    def __init__(self):
        self.chat_model = init_chat_model(
            model="qwen-plus",
            model_provider="openai",
            api_key=os.getenv("ALIYUN_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0,
            streaming=True,
        )
        self.redis_history = RedisChatMessageHistory(url="redis://0.0.0.0:6379/0", session_id="lisa")
        self.memory_threshold = 10
        self.model_with_tools = self.chat_model.bind_tools(TOOLS)
        self.tool_node = ToolNode(TOOLS)
        self.app = self._build_graph()

        # -------- 内部：构建图 --------
    def _build_graph(self):
        workflow = StateGraph(AgentState)

        # 1) 情绪识别节点
        def emotion_node(state: AgentState):
            # 找到最近一条 Human 输入
            last_user_text = ""
            for m in reversed(state["messages"]):
                if isinstance(m, HumanMessage):
                    last_user_text = m.content
                    break

            prompt = """根据用户的输入判断用户的情绪，回应规则：
            1. 负面情绪 → depressed
            2. 正面情绪 → friendly
            3. 中性情绪 → default
            4. 辱骂/不礼貌 → angry
            5. 兴奋 → upbeat
            6. 悲伤 → depressed
            7. 开心 → cheerful
            只返回英文标签，不要任何其他内容。
            用户输入：{query}"""
            chain = ChatPromptTemplate.from_template(prompt) | self.chat_model | StrOutputParser()
            qingxu = chain.invoke({"query": last_user_text}) if last_user_text else "default"
            return {"qingxu": qingxu.strip() or "default"}

        # 2) 主对话模型节点（会产生 tool_calls）
        def call_model(state: AgentState):
            mood = state.get("qingxu", "default")
            system_msg = SystemMessage(
                content=(
                        SYSTEM_PL
                        + "\n"
                        + (MOODS.get(mood, MOODS["default"])["roleSet"] or "")
                        + "\n你可以使用名为 `search` 的外部工具来检索实时信息；"
                          "当且仅当需要最新事实/数据或用户明确要求时，调用该工具。"
                          "工具返回后，要引用其关键信息推进面试，不要原样贴回全部搜索结果。"
                )
            )
            # 把系统提示放在最前面，然后拼接历史 + 当前对话
            messages = [system_msg] + state["messages"]

            # 关键：调用支持工具的模型
            ai_msg: AIMessage = self.model_with_tools.invoke(messages)
            # 通过 add_messages 聚合追加
            return {"messages": [ai_msg]}

        # 3) 条件路由：是否需要调用工具
        def should_call_tools(state: AgentState):
            last = state["messages"][-1]
            if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
                return "tools"
            return "end"

        # 4) 构图
        workflow.add_node("emotion", emotion_node)
        workflow.add_node("call_model", call_model)
        workflow.add_node("tools", self.tool_node)

        workflow.set_entry_point("emotion")
        workflow.add_edge("emotion", "call_model")
        workflow.add_conditional_edges("call_model", should_call_tools, {"tools": "tools", "end": END})
        workflow.add_edge("tools", "call_model")  # 工具执行完回到模型，直到不再请求工具

        return workflow.compile()

    # -------- 内部：历史摘要压缩（与原逻辑一致）--------
    def _summarize_if_needed(self):
        if len(self.redis_history.messages) <= self.memory_threshold:
            return
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    SYSTEM_PL
                    + "\n这是一段和你用户的对话记忆，对其进行总结摘要，摘要使用第一人称‘我’，"
                      "并且提取其中的用户关键信息，如姓名、年龄、性别、出生日期等。"
                      "以如下格式返回：\n总结摘要内容｜用户关键信息",
                ),
                ("user", "{input}"),
            ]
        )
        chain = prompt | self.chat_model
        summary_msg: AIMessage = chain.invoke({"input": str(self.redis_history.messages)})
        # 清空并写回摘要（保持你原来的做法）
        self.redis_history.clear()
        self.redis_history.add_message(summary_msg)

    # -------- 对外：对话入口（与原 chat(query) 对齐）--------
    def chat(self, query: str) -> str:
        # 1) 若历史过长先压缩
        self._summarize_if_needed()

        # 2) 取出 Redis 历史，作为初始消息；拼接本轮 Human
        history_msgs: List[AnyMessage] = list(self.redis_history.messages)
        init_state: AgentState = {
            "messages": history_msgs + [HumanMessage(content=query)],
            "qingxu": "default",
        }

        # 3) 运行图（自动完成：情绪识别 -> 模型 -> 工具 -> 模型 -> ... -> 结束）
        final_state: AgentState = self.app.invoke(init_state)

        # 4) 取最后一条 AI 回复
        last_ai_text = ""
        for m in reversed(final_state["messages"]):
            if isinstance(m, AIMessage):
                last_ai_text = m.content
                break

        # 5) 将「本轮 Human + 最终 AI」落盘到 Redis（避免把 ToolMessage 等中间消息写入）
        self.redis_history.add_message(HumanMessage(content=query))
        if last_ai_text:
            self.redis_history.add_message(AIMessage(content=last_ai_text))

        return last_ai_text

        # -------- 保留你的情绪判断接口（如果你需要单独用）--------
    def qingxu_chain(self, query: str):
        prompt = """根据用户的输入判断用户的情绪，回应的规则如下：
1. 如果用户输入的内容偏向于负面情绪，只返回"depressed"
2. 如果用户输入的内容偏向于正面情绪，只返回"friendly"
3. 如果用户输入的内容偏向于中性情绪，只返回"default"
4. 如果用户输入的内容包含辱骂或者不礼貌词句，只返回"angry"
5. 如果用户输入的内容比较兴奋，只返回"upbeat"
6. 如果用户输入的内容比较悲伤，只返回"depressed"
7. 如果用户输入的内容比较开心，只返回"cheerful"
8. 只返回英文，不允许有换行符等其他内容。
用户输入：{query}"""
        chain = ChatPromptTemplate.from_template(prompt) | self.chat_model | StrOutputParser()
        result = chain.invoke({"query": query}).strip()
        # 顺带走一轮对话
        reply = self.chat(query)
        return [{"msg": reply, "qingxu": result}]


@app.get("/get")
def read_root():
    return {"Hello": "World"}


class ChatRequest(BaseModel):
    query: str


@app.post("/chat")
def chat(req: ChatRequest):
    master = MasterGraph()
    res = master.qingxu_chain(req.query)
    return res



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10082)