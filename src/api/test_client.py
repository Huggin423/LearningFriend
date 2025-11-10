"""
API服务测试客户端
演示如何调用各个API接口
"""

import requests
import base64
import time
from pathlib import Path


class APIClient:
    """API客户端封装"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def _post(self, endpoint: str, json_data: dict = None):
        """POST请求"""
        url = f"{self.base_url}{endpoint}"
        response = self.session.post(url, json=json_data)
        return response.json()
    
    def _get(self, endpoint: str):
        """GET请求"""
        url = f"{self.base_url}{endpoint}"
        response = self.session.get(url)
        return response.json()
    
    # ==================== 服务管理 ====================
    
    def health_check(self):
        """健康检查"""
        return self._get("/health")
    
    def load_all(self):
        """加载所有服务"""
        return self._post("/load/all")
    
    def load_asr(self):
        """加载ASR模型"""
        return self._post("/load/asr")
    
    def load_llm(self):
        """初始化LLM客户端"""
        return self._post("/load/llm")
    
    # ==================== ASR接口 ====================
    
    def transcribe_file(self, audio_path: str, language: str = "zh", use_itn: bool = True):
        """识别音频文件"""
        return self._post("/asr/transcribe", {
            "audio_path": audio_path,
            "language": language,
            "use_itn": use_itn
        })
    
    def transcribe_base64(self, audio_base64: str, language: str = "zh", use_itn: bool = True):
        """识别Base64音频"""
        return self._post("/asr/transcribe", {
            "audio_base64": audio_base64,
            "language": language,
            "use_itn": use_itn
        })
    
    # ==================== LLM接口 ====================
    
    def chat(self, message: str, use_history: bool = True, system_prompt: str = None):
        """LLM对话"""
        data = {
            "message": message,
            "use_history": use_history
        }
        if system_prompt:
            data["system_prompt"] = system_prompt
        
        return self._post("/llm/chat", data)
    
    def clear_history(self):
        """清空对话历史"""
        return self._post("/llm/history/clear")
    
    # ==================== 综合接口 ====================
    
    def conversation_file(self, audio_path: str, language: str = "zh"):
        """对话流程：音频文件 -> 识别 -> LLM -> 文字回复"""
        return self._post("/conversation", {
            "audio_path": audio_path,
            "language": language,
            "use_itn": True
        })
    
    def conversation_base64(self, audio_base64: str, language: str = "zh"):
        """对话流程：Base64音频 -> 识别 -> LLM -> 文字回复"""
        return self._post("/conversation", {
            "audio_base64": audio_base64,
            "language": language,
            "use_itn": True
        })


def demo_basic_usage():
    """基础使用示例"""
    print("=" * 60)
    print("基础使用示例")
    print("=" * 60)
    
    client = APIClient("http://localhost:8000")
    
    # 1. 健康检查
    print("\n1. 健康检查...")
    result = client.health_check()
    print(f"   状态: {result}")
    
    # 2. 加载服务
    print("\n2. 加载所有服务...")
    result = client.load_all()
    print(f"   结果: {result['message']}")
    
    # 3. LLM对话测试
    print("\n3. LLM对话测试...")
    questions = [
        "你好，请用一句话介绍你自己",
        "你能做什么？",
        "谢谢你的介绍"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n   [{i}] 用户: {question}")
        result = client.chat(question, use_history=True)
        if result['success']:
            print(f"   [{i}] 助手: {result['reply']}")
            print(f"   耗时: {result['duration']:.2f}秒")
        else:
            print(f"   错误: {result['message']}")
        time.sleep(0.5)
    
    # 4. 清空历史
    print("\n4. 清空对话历史...")
    result = client.clear_history()
    print(f"   {result['message']}")


def demo_asr_usage(audio_path: str):
    """ASR使用示例"""
    print("\n" + "=" * 60)
    print("ASR语音识别示例")
    print("=" * 60)
    
    client = APIClient("http://localhost:8000")
    
    # 检查音频文件
    if not Path(audio_path).exists():
        print(f"\n错误: 音频文件不存在: {audio_path}")
        print("请提供一个有效的音频文件路径")
        return
    
    # 加载ASR模型
    print("\n1. 加载ASR模型...")
    result = client.load_asr()
    print(f"   {result['message']}")
    
    # 方式1: 使用文件路径
    print(f"\n2. 识别音频文件: {audio_path}")
    result = client.transcribe_file(audio_path)
    if result['success']:
        print(f"   识别结果: {result['text']}")
        print(f"   耗时: {result['duration']:.2f}秒")
    else:
        print(f"   错误: {result['message']}")
    
    # 方式2: 使用Base64编码
    print(f"\n3. 使用Base64编码识别...")
    with open(audio_path, "rb") as f:
        audio_base64 = base64.b64encode(f.read()).decode()
    
    result = client.transcribe_base64(audio_base64)
    if result['success']:
        print(f"   识别结果: {result['text']}")
        print(f"   耗时: {result['duration']:.2f}秒")
    else:
        print(f"   错误: {result['message']}")


def demo_conversation_flow(audio_path: str):
    """综合对话流程示例"""
    print("\n" + "=" * 60)
    print("综合对话流程示例（语音输入 -> 文字回复）")
    print("=" * 60)
    
    client = APIClient("http://localhost:8000")
    
    # 检查音频文件
    if not Path(audio_path).exists():
        print(f"\n错误: 音频文件不存在: {audio_path}")
        return
    
    # 加载所有服务
    print("\n1. 加载所有服务...")
    result = client.load_all()
    print(f"   {result['message']}")
    
    # 执行对话流程
    print(f"\n2. 执行对话流程...")
    print(f"   音频文件: {audio_path}")
    
    result = client.conversation_file(audio_path)
    
    if result['success']:
        print(f"\n   ✓ 对话流程成功!")
        print(f"   用户说: {result['message'].replace('用户说: ', '')}")
        print(f"   助手回复: {result['reply']}")
        print(f"   总耗时: {result['duration']:.2f}秒")
    else:
        print(f"   ✗ 失败: {result['message']}")


def demo_custom_system_prompt():
    """自定义系统提示词示例"""
    print("\n" + "=" * 60)
    print("自定义系统提示词示例")
    print("=" * 60)
    
    client = APIClient("http://localhost:8000")
    
    # 加载LLM
    print("\n1. 初始化LLM客户端...")
    result = client.load_llm()
    print(f"   {result['message']}")
    
    # 使用不同的系统提示词
    system_prompts = [
        ("默认", None),
        ("诗人", "你是一位才华横溢的诗人,用优美的诗句回答问题"),
        ("程序员", "你是一位资深程序员,用技术术语解释问题"),
    ]
    
    question = "什么是人工智能？"
    
    for name, prompt in system_prompts:
        print(f"\n2. 使用 [{name}] 身份回答问题:")
        print(f"   问题: {question}")
        
        result = client.chat(question, use_history=False, system_prompt=prompt)
        
        if result['success']:
            print(f"   回答: {result['reply'][:100]}...")
        else:
            print(f"   错误: {result['message']}")
        
        time.sleep(0.5)


def main():
    """主函数"""
    import sys
    
    print("\n" + "=" * 60)
    print("LearningFriend API 客户端测试")
    print("=" * 60)
    
    # 检查服务是否运行
    try:
        client = APIClient("http://localhost:8000")
        health = client.health_check()
        print(f"\n✓ API服务正在运行")
        print(f"  版本: {health.get('version', 'unknown')}")
    except Exception as e:
        print(f"\n✗ 无法连接到API服务")
        print(f"  错误: {str(e)}")
        print(f"\n请确保API服务已启动:")
        print(f"  python -m src.api.main")
        return
    
    # 运行示例
    print("\n选择要运行的示例:")
    print("  1. 基础使用示例（LLM对话）")
    print("  2. ASR语音识别示例")
    print("  3. 综合对话流程示例")
    print("  4. 自定义系统提示词示例")
    print("  5. 运行所有示例")
    
    choice = input("\n请输入选项 (1-5): ").strip()
    
    if choice == "1":
        demo_basic_usage()
    
    elif choice == "2":
        audio_path = input("请输入音频文件路径: ").strip()
        if audio_path:
            demo_asr_usage(audio_path)
        else:
            print("未提供音频文件路径")
    
    elif choice == "3":
        audio_path = input("请输入音频文件路径: ").strip()
        if audio_path:
            demo_conversation_flow(audio_path)
        else:
            print("未提供音频文件路径")
    
    elif choice == "4":
        demo_custom_system_prompt()
    
    elif choice == "5":
        demo_basic_usage()
        demo_custom_system_prompt()
        
        # 如果提供了音频文件，也运行ASR示例
        if len(sys.argv) > 1:
            audio_path = sys.argv[1]
            demo_asr_usage(audio_path)
            demo_conversation_flow(audio_path)
    
    else:
        print("无效选项")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
