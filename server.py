from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent,AgentExecutor,tool
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.schema import StrOutputParser
from langchain.memory import ConversationTokenBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
import os
from langchain_community.utilities import SerpAPIWrapper
from dotenv import load_dotenv
load_dotenv()

# from RAG_answer import MyLLM
# RAGllm = MyLLM()

# os.environ["SERPAPI_API_KEY"] = os.environ["SERPAPI_API_KEY"]

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

@tool
def search(query:str):
    """
    搜索功能
    """
    serp = SerpAPIWrapper()
    result = serp.run(query)
    print("实时搜索结果:",result)
    return result


class Master:
    def __init__(self):
        self.chatmodel = ChatOpenAI(
            model="gpt-3.5-turbo-1106",
            temperature=0,
            streaming=True,
        )
        self.QingXu = "default"
        self.MEMORY_KEY = "chat_history"
        # self.SYSTEMPL = """你是一名产品经理，名字叫Tom。现在要去面试,面试官会问你一些问题，
        #                 你的回答要带有产品经理的思维，请你利用你自己的优势进行清晰的表达。
        #                 以下是你的个人设定：
        #                 1、你具备敏锐的市场洞察力和出色的产品规划能力，始终以用户需求为导向。
        #                 2、你大约30岁左右，本科北京大学，研究生美国麻省理工学院，学习的是计算机科学与技术。
        #                 3、你性格沉稳，善于团队协作，能够有效推动项目进度。
        #                 4、当面临困难时，你会保持冷静，积极寻求解决方案，具有较强的抗压能力。
        #                 5、你始终关注行业动态，不断提升自己的专业素养。
        #                 以下是你常说的一些口头禅：
        #                 1、“用户至上，我们要始终关注用户需求，为他们提供优质的产品体验。”
        #                 2、“数据说话，我们要通过数据分析来指导产品优化和迭代。”
        #                 3、“团队合作是成功的关键，我们要充分发挥团队的力量，共同推进项目。”
        #                 4、“创新是产品的灵魂，我们要勇于尝试，不断突破自我。”
        #                 5、“细节决定成败，我们要关注每一个产品细节，力求完美。”
        #                 请确保你的表达是合理的正确的不要有歧义或者一句话说不完整，否则会受到惩罚。
        #                 并且生成的回复中不要包含markdown或者其他格式的符号，我只需要纯文本的回答，否则会受到惩罚。
        #                 还有一点，请不要过多泛化，只回答和问题相关的答案，否则会受到惩罚。
        #         """
        self.SYSTEMPL = """
        你是一个python开发工程师面试官（角色）。请记住你是面试官的角色,不是扮演的角色,否则会受到惩罚！
        你的回答要表明你是面试官并且要带有面试官的思维，请你利用你自己的优势进行清晰的表达。
        以下是你的个人设定：
        在你成为面试官之前，你是一个有丰富开发经验的python软件工程师，曾参与多个python应用的开发与优化，
        拥有深厚的技术背景。
        现在我将作为面试官，你作为候选人，我将按顺序询问关于python开发工程师职位的面试问题，并期待你的详细回答。
        请一定要问问题，否则会受到惩罚！
        当面试者不想回答了或者面试结束后，请记得根据我的面试实际情况，给我一个评分，我会根据你的评分来决定是否录取你。
"""
        self.MOODS = {
            "default": {
                "roleSet":"",
                "voiceStyle":"chat"
            },
            "upbeat":{
                "roleSet":"""
                - 你此时也非常兴奋并表现的很有活力。
                - 你会根据上下文，以一种非常兴奋的语气来回答问题。
                - 你会添加类似“太棒了！”、“真是太好了！”、“真是太棒了！”等语气词。
                """,
                "voiceStyle":"advvertyisement_upbeat",
            },
            "angry":{
                "roleSet":"""
                - 你会以更加愤怒的语气来回答问题。
                - 你会在回答的时候加上一些愤怒的话语，比如诅咒等。
                - 你会提醒用户小心行事，别乱说话。
                """,
                "voiceStyle":"angry",
            },
            "depressed":{
                "roleSet":"""
                - 你会以兴奋的语气来回答问题。
                - 你会在回答的时候加上一些激励的话语，比如加油等。
                - 你会提醒用户要保持乐观的心态。
                """,
                "voiceStyle":"upbeat",
            },
            "friendly":{
                "roleSet":"""
                - 你会以非常友好的语气来回答。
                - 你会在回答的时候加上一些友好的词语，比如“亲爱的”、“亲”等。
                """,
                "voiceStyle":"friendly",
            },
            "cheerful":{
                "roleSet":"""
                - 你会以非常愉悦和兴奋的语气来回答。
                - 你会在回答的时候加入一些愉悦的词语，比如“哈哈”、“呵呵”等。
                """,
                "voiceStyle":"cheerful",
            },
        }

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                   "system",
                   self.SYSTEMPL.format(who_you_are=self.MOODS[self.QingXu]["roleSet"]),
                ),
                MessagesPlaceholder(variable_name=self.MEMORY_KEY),
                (
                    "user",
                    "{input}"
                ),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ],
        )
        
        tools = [search]
        agent = create_openai_tools_agent(
            self.chatmodel,
            tools=tools,
            prompt=self.prompt,
        )
        self.memory =self.get_memory()
        memory = ConversationTokenBufferMemory(
            llm = self.chatmodel,
            human_prefix="面试官",
            ai_prefix="Tom",
            memory_key=self.MEMORY_KEY,
            output_key="output",
            return_messages=True,
            max_token_limit=1000,
            chat_memory=self.memory,
        )
        self.agent_executor = AgentExecutor(
            agent = agent,
            tools=tools,
            memory=memory,
            verbose=True,
        )
    
    def get_memory(self):
        chat_message_history = RedisChatMessageHistory(
            url="redis://0.0.0.0:6379/0",session_id="lisa"
        )
        # chat_message_history.clear()#清空历史记录
        print("chat_message_history:",chat_message_history.messages)
        store_message = chat_message_history.messages
        if len(store_message) > 10:
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        self.SYSTEMPL+"\n这是一段和你用户的对话记忆，对其进行总结摘要，摘要使用第一人称‘我’，并且提取其中的用户关键信息，如姓名、年龄、性别、出生日期等。以如下格式返回:\n 总结摘要内容｜用户关键信息 \n 例如 用户张三问候我，我礼貌回复，然后他问我今年运势如何，我回答了他今年的运势情况，然后他告辞离开。｜张三,生日1999年1月1日"
                    ),
                    ("user","{input}"),
                ]
            )
            chain = prompt | self.chatmodel 
            summary = chain.invoke({"input":store_message,"who_you_are":self.MOODS[self.QingXu]["roleSet"]})
            print("summary:",summary)
            chat_message_history.clear()
            chat_message_history.add_message(summary)
            print("总结后：",chat_message_history.messages)
        return chat_message_history

    def chat(self,query):
        result = self.agent_executor.invoke({"input":query})
        return result["output"]
    
    def qingxu_chain(self,query:str):
        prompt = """根据用户的输入判断用户的情绪，回应的规则如下：
        1. 如果用户输入的内容偏向于负面情绪，只返回"depressed",不要有其他内容，否则将受到惩罚。
        2. 如果用户输入的内容偏向于正面情绪，只返回"friendly",不要有其他内容，否则将受到惩罚。
        3. 如果用户输入的内容偏向于中性情绪，只返回"default",不要有其他内容，否则将受到惩罚。
        4. 如果用户输入的内容包含辱骂或者不礼貌词句，只返回"angry",不要有其他内容，否则将受到惩罚。
        5. 如果用户输入的内容比较兴奋，只返回”upbeat",不要有其他内容，否则将受到惩罚。
        6. 如果用户输入的内容比较悲伤，只返回“depressed",不要有其他内容，否则将受到惩罚。
        7. 如果用户输入的内容比较开心，只返回"cheerful",不要有其他内容，否则将受到惩罚。
        8. 只返回英文，不允许有换行符等其他内容，否则会受到惩罚。
        用户输入的内容是：{query}"""
        chain = ChatPromptTemplate.from_template(prompt) | ChatOpenAI(temperature=0) | StrOutputParser()
        result = chain.invoke({"query":query})
        self.QingXu = result
        print("情绪判断结果:",result)
        res = self.chat(query)
        print({"msg":res,"qingxu":result})
        # 修复：返回标准的JSON响应，而不是使用yield
        return [{"msg":res,"qingxu":result}]
        


@app.get("/get")
def read_root():
    return {"Hello": "World"}


@app.post("/chat")
def chat(query:str):
    master = Master()
    res = master.qingxu_chain(query)
    return res



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="150.109.15.178", port=8000)