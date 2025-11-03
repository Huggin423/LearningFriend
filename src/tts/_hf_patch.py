"""
HuggingFace Hub 补丁：禁用重试机制
在离线模式下，避免长时间的网络重试和警告
"""
import os
import sys
import warnings
import logging

# 禁用所有警告（减少日志噪音）
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 禁用 urllib3 和 requests 的警告
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# 在导入 huggingface_hub 之前设置环境变量
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['HF_HUB_DISABLE_EXPERIMENTAL_WARNING'] = '1'

# Patch http_backoff 函数以减少重试次数和禁用警告
def patch_hf_hub():
    """修补 HuggingFace Hub 以减少重试和警告"""
    try:
        from huggingface_hub.utils import _http
        
        original_http_backoff = _http.http_backoff
        
        def patched_http_backoff(*args, max_retries=0, **kwargs):
            """修补后的 http_backoff，禁用重试并抑制警告"""
            # 临时禁用 warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return original_http_backoff(*args, max_retries=max_retries, **kwargs)
        
        _http.http_backoff = patched_http_backoff
        
        # 同时修补日志记录器
        _http.logger.setLevel(logging.ERROR)
    except Exception:
        pass  # 如果修补失败，忽略

# 自动执行修补
patch_hf_hub()

