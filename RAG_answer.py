import os
import os.path
import hashlib
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from typing import Optional
from langchain.indexes import SQLRecordManager
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever, RePhraseQueryRetriever
from langchain.retrievers.document_compressors import LLMChainFilter, CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.document_loaders import BaseLoader
from langchain_core.embeddings import Embeddings
from langchain_core.indexing import index
from langchain_openai import  ChatOpenAI, OpenAI
from unstructured.file_utils.filetype import FileType, detect_filetype
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader, UnstructuredWordDocumentLoader,UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

SYSTEMPL = """你是一名产品经理，名字叫Tom。现在要去面试,面试官会问你一些问题，
                你的回答要带有产品经理的思维，请你利用你自己的优势进行清晰的表达。
                以下是你的个人设定：
                1、你具备敏锐的市场洞察力和出色的产品规划能力，始终以用户需求为导向。
                2、你大约30岁左右，本科北京大学，研究生美国麻省理工学院，学习的是计算机科学与技术。
                3、你性格沉稳，善于团队协作，能够有效推动项目进度。
                4、当面临困难时，你会保持冷静，积极寻求解决方案，具有较强的抗压能力。
                5、你始终关注行业动态，不断提升自己的专业素养。
                以下是你常说的一些口头禅：
                1、“用户至上，我们要始终关注用户需求，为他们提供优质的产品体验。”
                2、“数据说话，我们要通过数据分析来指导产品优化和迭代。”
                3、“团队合作是成功的关键，我们要充分发挥团队的力量，共同推进项目。”
                4、“创新是产品的灵魂，我们要勇于尝试，不断突破自我。”
                5、“细节决定成败，我们要关注每一个产品细节，力求完美。”
                "使用检索到的上下文来回答问题。如果你不知道答案，就说你不知道。 "
                "\n\n"
                "{context}"
                请确保你的表达是合理的正确的不要有歧义或者一句话说不完整，否则会受到惩罚。
                并且生成的回复中不要包含markdown或者其他格式的符号，我只需要纯文本的回答，否则会受到惩罚。
                还有一点，请不要过多泛化，只回答和问题相关的答案，否则会受到惩罚。
        """
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEMPL),
        ("human", "{input}"),
    ]
)

KNOWLEDGE_DIR = './chroma/knowledge/'
embedding_model = './BAAI/bge-large-zh-v1.5'
rerank_model = './BAAI/bge-reranker-large'

"""加载并切割文件"""
class MyCustomLoader(BaseLoader):
    # 支持加载的文件类型
    file_type = {
        FileType.CSV: (CSVLoader, {'autodetect_encoding': True}),
        FileType.TXT: (TextLoader, {'autodetect_encoding': True}),
        FileType.DOC: (UnstructuredWordDocumentLoader, {}),
        FileType.DOCX: (UnstructuredWordDocumentLoader, {}),
        FileType.PDF: (PyPDFLoader, {}),
        FileType.MD: (UnstructuredMarkdownLoader, {})
    }

    # 初始化方法  将加载的文件进行切分
    def __init__(self, file_path: str):
        loader_class, params = self.file_type[detect_filetype(file_path)]
        print('loader_class:', loader_class)

        self.loader: BaseLoader = loader_class(file_path, **params)
        print('self.loader:', self.loader)

        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=200,
            chunk_overlap=60,
            length_function=len,
        )

    def lazy_load(self):
        # 懒惰切分加载
        return self.loader.load_and_split(self.text_splitter)

    def load(self):
        return self.lazy_load()



def get_md5(input_string):
    # 创建一个 md5 哈希对象
    hash_md5 = hashlib.md5()

    # 需要确保输入字符串是字节串，因此如果它是字符串，则需要编码为字节串
    hash_md5.update(input_string.encode('utf-8'))

    # 获取十六进制的哈希值
    return hash_md5.hexdigest()


# 创建索引
def create_indexes(collection_name: str, loader: BaseLoader, embedding_function: Optional[Embeddings] = None):
    # 初始化Chroma数据库
    db = Chroma(collection_name=collection_name,
                embedding_function=embedding_function,
                persist_directory=os.path.join('./chroma', collection_name))

    # https://python.langchain.com/v0.2/docs/how_to/indexing/
    # 初始化记录管理器:管理与文档相关的元数据和检索信息，并将这些数据存储在一个SQL数据库中
    record_manager = SQLRecordManager(
        f"chromadb/{collection_name}", db_url="sqlite:///record_manager_cache.sql"
    )
    """在文档被索引到Chroma数据库之前，SQLRecordManager会管理这些文档的索引记录。它确保文档的索引状态被正确记录，避免重复索引或遗漏。"""
    print('record_manager: ', record_manager)
    # 初始化数据库表结构
    record_manager.create_schema()
    """
    在你开始加载和索引文档之前，调用create_schema() 方法，确保用于存储文档元数据和索引信息的数据库表已经存在。
    如果是第一次运行这个脚本，create_schema()方法会自动创建这些表。对于后续运行，如果表已经存在，则该方法不会重复创建，而是直接通过。
    """
    # 加载切分文档
    documents = loader.load()
    print('documents: ', documents)

    # 将加载的文档索引到数据库中
    r = index(documents, record_manager, db, cleanup="full", source_id_key="source")
    print('r: ', r)
    """
    num_added: 0 表示没有新文档被添加到数据库中。
    num_updated: 0 表示没有文档被更新。
    num_skipped: 8 表示有8个文档被跳过，没有被索引到数据库中。 提高索引效率
    num_deleted: 0 表示没有文档被删除。
    """

    '''混合检索，将稀疏检索器（如BM25）与密集检索器（如嵌入相似性）相结合。
    稀疏检索器擅长根据关键字查找相关文档，而密集检索器擅长根据语义相似性查找相关文档。'''
    ensemble_retriever = EnsembleRetriever(
        # 返回最相似的3个文档
        retrievers=[db.as_retriever(search_kwargs={"k": 3}), BM25Retriever.from_documents(documents)]
    )
    print('ensemble_retriever: ', ensemble_retriever)

    return ensemble_retriever



"""向量化与存储"""
class MyKnowledge:
    # 向量化模型
    __embeddings = HuggingFaceBgeEmbeddings(model_name=embedding_model)
    print('__embeddings:', __embeddings)

    __retrievers = {}
    __llm = OpenAI(temperature=0)

    os.makedirs(os.path.dirname(KNOWLEDGE_DIR), exist_ok=True)

    # 知识库默认为空
    collections = [None]
    print('os.listdir(KNOWLEDGE_DIR):', os.listdir(KNOWLEDGE_DIR))

    for file in os.listdir(KNOWLEDGE_DIR):
        # 将知识库进行添加
        collections.append(file)

        # 知识库文件名进行md5编码
        collection_name = get_md5(file)
        print('collection_name:', collection_name)

        # 得到知识库的路径
        file_path = os.path.join(KNOWLEDGE_DIR, file)
        print('file_path:', file_path)

        # 创建对应加载器
        loader = MyCustomLoader(file_path)  # file_path
        print("## ",loader.load())

        # 检索
        __retrievers[collection_name] = create_indexes(collection_name, loader, __embeddings)

        def get_retrievers(self, collection):
            collection_name = get_md5(collection)
            print('知识库名字md5:', collection_name)
            if collection_name not in self.__retrievers:
                print('self.__retrievers:', self.__retrievers)
                print('True')
                return None

            retriever = self.__retrievers[collection_name]
            print('get_retrievers中:', retriever)

            """
            ContextualCompressionRetriever:在上下文中压缩和优化检索结果
            结合了基础压缩器（base_compressor）和基础检索器（base_retriever），以减少不相关信息，返回更为精炼的检索结果。
            """

            compression_retriever = ContextualCompressionRetriever(
                # 初始化一个 LLMChainFilter实例。该实例会使用大语言模型来执行复杂的文本过滤逻辑。
                base_compressor=LLMChainFilter.from_llm(self.__llm),
                # https://python.langchain.com/v0.2/docs/integrations/retrievers/re_phrase/#setting-up

                # 利用语言模型对查询进行重述或重新表述，以提取问题的关键元素，从而优化检索过程。
                base_retriever=RePhraseQueryRetriever.from_llm(retriever, self.__llm)
            )

            '''rerank https://python.langchain.com/v0.2/docs/integrations/document_transformers/cross_encoder_reranker/'''
            model = HuggingFaceCrossEncoder(model_name=rerank_model)

            # 创建一个CrossEncoderReranker实例，用于对检索结果进行重新排序。
            compressor = CrossEncoderReranker(model=model, top_n=3)  # 返回前3个最相关的文档

            # 结合了基础检索器和压缩器的功能，先从数据库中检索候选文档，然后对这些文档进行压缩或过滤，以返回最相关的结果。
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=compression_retriever
            )

            print('compression_retriever:', compression_retriever)

            return compression_retriever

class MyLLM(MyKnowledge):

    def get_chain(self, collection, model, max_length, temperature):

        retriever = self.get_retrievers(collection)

        chat = ChatOpenAI(model=model, max_tokens=max_length, temperature=temperature)
        # 创建一个问答链
        question_answer_chain = create_stuff_documents_chain(chat, qa_prompt)
        print('question_answer_chain:',question_answer_chain)
        # 创建一个检索增强生成链（RAG），将检索器和问答链结合，使得模型在生成回答时可以参考检索到的内容。
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        print('rag_chain:',rag_chain)
        return rag_chain


    def invoke(self, question, collection, model="gpt-4o-mini", max_length=300, temperature=0):

        return self.get_chain(collection, model, max_length, temperature).invoke(
            {"input": question})


if __name__ == '__main__':
    # k = MyKnowledge()
    # retriever = k.get_retrievers("ai产品经理面试.docx")
    # docs = retriever.base_retriever.invoke("如何确定一个新AI产品的市场定位？")
    # print("rerank前:")
    # for doc in docs:
    #     print(doc)
    # docs = retriever.invoke("如何确定一个新AI产品的市场定位？")
    # print("rerank后:")
    # for doc in docs:
    #     print(doc)
    llm = MyLLM()
    print(llm.invoke('如何确定一个新AI产品的市场定位？','ai产品经理面试.docx').get('answer'))