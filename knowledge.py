"""
知识库
"""

import datetime
import hashlib
import os

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config_data as config


def check_md5(md5_str: str):
    """检查传入的md5字符串是否已经被处理过了"""
    if not os.path.exists(config.md5_path):
        open(config.md5_path, "w").close()  # 如果文件不存在，则创建一个空文件
        return False
    else:
        for line in open(config.md5_path, "r", encoding="utf-8").readlines():
            line = line.strip()  # 去掉每行的换行符和空格
            if line == md5_str:
                return True

        return False


def save_md5(md5_str: str):
    """将传入的md5字符串，记录到文件内保存"""
    with open(config.md5_path, "a", encoding="utf-8") as f:
        f.write(md5_str + "\n")


def get_string_md5(input_str: str,encoding='utf-8'):
    """将传入的字符串转换为md5字符串"""
    str_bytes = input_str.encode(encoding = encoding)
    md5_obj = hashlib.md5() # 得到md5对象
    md5_obj.update(str_bytes)   # 更新内容
    md5_hex = md5_obj.hexdigest()  # 获取md5字符串
    return md5_hex 


class KnowledgeBaseService(object):
    def __init__(self):
        os.makedirs(config.persist_directory, exist_ok=True)  # 创建Chroma数据库存储目录
        self.chroma = Chroma(
            collection_name=(config.collection_name),
            embedding_function=DashScopeEmbeddings(model="text-embedding-v4"),
            persist_directory=config.persist_directory
        )      # 向量存储的实例 Chroma向量库对象
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,   #自然短路划分符号
            length_function=len,
        )     # 文本分割器的对象

    def upload_by_str(self, data, filename):
        """将传入的字符串，进行向量化，存入向量数据库中"""
        md5_str = get_string_md5(data)
        if check_md5(md5_str):
            return f"文件 {filename} 已经被处理过了，跳过处理"
        if len(data) > config.max_splkit_char_number:
            knowledge_chunks = self.spliter.split_text(data)  # 分割文本
        else:
            knowledge_chunks = [data]

        metadata = {
            "source": filename,
            "create_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator": "admin",
        }



        self.chroma.add_texts(
            knowledge_chunks,
            metadatas=[metadata for _ in range(len(knowledge_chunks))]
        )

        save_md5(md5_str)

        return f"文件 {filename} 已经被处理完成，向量化存储成功"

if __name__ == "__main__":
    service = KnowledgeBaseService()
    service.upload_by_str("这是一个测试文本", "test.txt")

