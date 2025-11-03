"""
IndexTTS2 ModelScope 官方模型包装器
直接使用 ModelScope 上的预训练模型
"""

import os
import sys
import logging
import warnings

# 在导入其他模块之前，禁用 HuggingFace 重试
# 导入补丁模块（会自动设置环境变量和修补函数）
try:
    from . import _hf_patch
except ImportError:
    # 如果补丁不存在，直接设置环境变量
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    os.environ['HF_HUB_DISABLE_EXPERIMENTAL_WARNING'] = '1'

import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# 禁用所有警告（减少日志噪音）
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class IndexTTS2ModelScope:
    """IndexTTS2 ModelScope 官方模型包装器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化官方IndexTTS2模型
        
        Args:
            config: 配置字典
                - model_path: 本地模型路径（如果使用本地模型）
                - model: 模型 ID（如 "IndexTeam/IndexTTS-2"），如果使用 hub 模型
                - use_local: 是否强制使用本地模式（默认根据 model_path 自动判断）
        """
        # 最重要：在导入任何模块之前，立即设置离线模式
        # 这样可以避免 HuggingFace 库在初始化时尝试连接网络
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        
        # 禁用 HuggingFace 的重试和警告（减少日志噪音）
        os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
        os.environ['HF_HUB_DISABLE_EXPERIMENTAL_WARNING'] = '1'
        
        # 设置重试次数为 0（直接跳过重试）
        # 通过设置很短的超时和禁用重试异常来快速失败
        os.environ['HF_HUB_MAX_RETRIES'] = '0'
        
        # 确保使用正确的缓存目录
        if 'HF_HOME' not in os.environ:
            hf_home = os.path.expanduser('~/.cache/huggingface')
            os.environ['HF_HOME'] = hf_home
        
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.sample_rate = config.get('sample_rate', 22050)
        self.speed = config.get('speed', 1.0)
        
        # 判断使用本地模型还是 hub 模型
        self.use_local = config.get('use_local', None)  # None 表示自动判断
        model_path = config.get('model_path', 'models/indextts2')
        model_id = config.get('model', 'IndexTeam/IndexTTS-2')
        
        # 详细日志
        logger.info("=" * 60)
        logger.info("IndexTTS2 初始化配置")
        logger.info(f"  model_path: {model_path}")
        logger.info(f"  model (ID): {model_id}")
        logger.info(f"  use_local: {self.use_local}")
        logger.info(f"  device: {self.device}")
        logger.info("=" * 60)
        
        # 如果未指定 use_local，根据 model_path 是否存在自动判断
        if self.use_local is None:
            if not os.path.isabs(model_path):
                project_root = Path(__file__).parent.parent.parent
                model_path_abs = (project_root / model_path).resolve()
            else:
                model_path_abs = Path(model_path).resolve()
            
            # 如果路径存在且有文件，使用本地模式
            self.use_local = model_path_abs.exists() and any(model_path_abs.iterdir())
            logger.info(f"自动判断模式: {'本地模式' if self.use_local else 'Hub 模式'}")
        
        # 设置模型目录
        if self.use_local:
            # 本地模式：使用 model_path
            if not os.path.isabs(model_path):
                project_root = Path(__file__).parent.parent.parent
                self.model_dir = (project_root / model_path).resolve()
            else:
                self.model_dir = Path(model_path).resolve()
            logger.info(f"本地模式 - 模型目录: {self.model_dir}")
        else:
            # Hub 模式：先设置临时目录，后续会下载
            if not os.path.isabs(model_path):
                project_root = Path(__file__).parent.parent.parent
                self.model_dir = (project_root / model_path).resolve()
            else:
                self.model_dir = Path(model_path).resolve()
            self.model_id = model_id
            logger.info(f"Hub 模式 - 模型 ID: {self.model_id}, 缓存目录: {self.model_dir}")
        
        # 初始化模型
        self._load_model()
        
        logger.info("✓ IndexTTS2 ModelScope模型初始化成功")
        logger.info(f"  设备: {self.device}")
        logger.info(f"  模型目录: {self.model_dir}")
    
    def _load_model(self):
        """加载 IndexTTS2 模型（支持本地模式和 Hub 模式）"""
        try:
            # 根据模式加载模型
            if self.use_local:
                self._load_local_model()
            else:
                self._load_from_hub()
            
            # 确保模型目录存在且有文件
            if not self.model_dir.exists() or not any(self.model_dir.iterdir()):
                raise FileNotFoundError(
                    f"模型目录为空或不存在: {self.model_dir}\n"
                    f"请检查模型路径或重新下载模型"
                )
            
            # 查找并准备配置文件
            config_path = self._find_and_prepare_config()
            
            # 添加 index-tts 到 Python 路径并加载模型
            self._setup_indextts_and_load(config_path)
            
        except ImportError as e:
            logger.error(f"导入失败: {str(e)}")
            logger.error("请确保已安装所需依赖:")
            logger.error("  pip install modelscope")
            logger.error("  pip install transformers==4.52.1")
            raise
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            raise
    
    def _load_from_hub(self):
        """从 ModelScope Hub 下载模型"""
        from modelscope.hub.snapshot_download import snapshot_download
        
        logger.info(f"Hub 模式：从 ModelScope 下载模型 {self.model_id}...")
        
        # 检查模型是否已下载
        if not self.model_dir.exists() or not any(self.model_dir.iterdir()):
            logger.info("模型未找到，正在从 ModelScope 下载...")
            try:
                snapshot_download(
                    self.model_id, 
                    cache_dir=str(self.model_dir),
                    local_files_only=False
                )
                logger.info("✓ 模型下载完成")
            except Exception as e:
                logger.warning(f"下载到指定目录失败: {e}")
                # 检查是否在默认缓存位置
                default_cache = os.path.expanduser("~/.cache/modelscope/hub/" + self.model_id.replace("/", "--"))
                if os.path.exists(default_cache):
                    logger.info(f"使用 ModelScope 默认缓存位置: {default_cache}")
                    self.model_dir = Path(default_cache)
        else:
            logger.info(f"使用已存在的本地模型: {self.model_dir}")
    
    def _load_local_model(self):
        """加载本地已下载的模型"""
        logger.info(f"本地模式：使用本地模型路径 {self.model_dir}")
        
        # 验证模型目录存在
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"本地模型目录不存在: {self.model_dir}\n"
                f"请检查 model_path 配置或先下载模型"
            )
        
        if not any(self.model_dir.iterdir()):
            raise FileNotFoundError(
                f"本地模型目录为空: {self.model_dir}\n"
                f"请确保模型文件已正确下载到此目录"
            )
        
        logger.info(f"✓ 本地模型目录验证通过: {self.model_dir}")
    
    def _find_and_prepare_config(self):
        """查找并准备配置文件"""
        # 查找配置文件
        config_path = None
        possible_config_paths = [
            self.model_dir / "IndexTeam" / "IndexTTS-2" / "config.yaml",  # ModelScope 下载的标准路径
            self.model_dir / "config.yaml",  # 直接路径
        ]
        
        for path in possible_config_paths:
            if path.exists():
                config_path = path
                # 如果找到了 IndexTeam/IndexTTS-2 下的配置文件，更新 model_dir
                if "IndexTeam" in str(path) and "IndexTTS-2" in str(path):
                    # path 是 models/indextts2/IndexTeam/IndexTTS-2/config.yaml
                    # 我们需要 model_dir 指向 IndexTTS-2 目录
                    actual_model_dir = path.parent  # IndexTTS-2 目录
                    self.model_dir = actual_model_dir.resolve()
                    logger.info(f"找到配置文件: {config_path}")
                    logger.info(f"更新模型目录为: {self.model_dir}")
                    # 验证 Qwen 模型路径
                    qwen_check = self.model_dir / "qwen0.6bemo4-merge"
                    if qwen_check.exists():
                        logger.info(f"✓ 验证通过：Qwen 模型路径存在: {qwen_check}")
                    else:
                        logger.warning(f"⚠ 警告：Qwen 模型路径不存在: {qwen_check}")
                else:
                    logger.info(f"找到配置文件: {config_path}")
                break
        
        if config_path is None:
            # 如果找不到配置文件，尝试在模型目录的子目录中查找
            logger.info(f"在标准位置未找到配置文件，搜索子目录...")
            for subdir in self.model_dir.iterdir():
                if subdir.is_dir():
                    # 检查 IndexTeam/IndexTTS-2 结构
                    if subdir.name == "IndexTeam":
                        index_tts_dir = subdir / "IndexTTS-2"
                        if index_tts_dir.exists():
                            sub_config = index_tts_dir / "config.yaml"
                            if sub_config.exists():
                                config_path = sub_config
                                self.model_dir = index_tts_dir.resolve()  # 更新模型目录
                                logger.info(f"在 IndexTeam/IndexTTS-2 目录中找到配置文件: {config_path}")
                                logger.info(f"更新模型目录为: {self.model_dir}")
                                break
                    else:
                        # 检查其他子目录
                        sub_config = subdir / "config.yaml"
                        if sub_config.exists():
                            config_path = sub_config
                            self.model_dir = subdir.resolve()  # 更新模型目录
                            logger.info(f"在子目录中找到配置文件: {config_path}")
                            logger.info(f"更新模型目录为: {self.model_dir}")
                            break
        
        if config_path is None:
            raise FileNotFoundError(
                f"找不到配置文件 config.yaml\n"
                f"请检查模型目录: {self.model_dir}\n"
                f"预期路径: {self.model_dir / 'IndexTeam' / 'IndexTTS-2' / 'config.yaml'}"
            )
        
        # 修复配置文件中的路径问题（如果 qwen_emo_path 末尾有斜杠）
        # 读取并修复配置文件，确保路径格式正确
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config_content = yaml.safe_load(f)
            
            # 检查并修复 qwen_emo_path
            if config_content and 'qwen_emo_path' in config_content:
                qwen_path = config_content['qwen_emo_path']
                if isinstance(qwen_path, str) and qwen_path.endswith('/'):
                    # 移除末尾斜杠
                    config_content['qwen_emo_path'] = qwen_path.rstrip('/')
                    # 创建临时配置文件
                    import tempfile
                    temp_config = tempfile.NamedTemporaryFile(
                        mode='w', 
                        suffix='.yaml', 
                        delete=False,
                        dir=str(Path(config_path).parent)
                    )
                    yaml.dump(config_content, temp_config, default_flow_style=False, allow_unicode=True)
                    temp_config.close()
                    config_path = Path(temp_config.name)
                    logger.info(f"已修复配置文件中的路径（移除 qwen_emo_path 末尾斜杠），使用临时配置文件: {config_path}")
        except Exception as e:
            logger.warning(f"无法修复配置文件路径: {e}，使用原始配置文件")
        
        return config_path
    
    def _setup_indextts_and_load(self, config_path: Path):
        """设置 index-tts 路径并加载模型"""
        # 设置离线模式环境变量，避免从 HuggingFace 下载模型
        # IndexTTS2 会尝试下载 facebook/w2v-bert-2.0 等模型，如果网络不可达，需要离线模式
        import os
        
        # 检查 HuggingFace 缓存目录
        hf_cache_dir = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
        logger.info(f"HuggingFace 缓存目录: {hf_cache_dir}")
        
        # 检查 ModelScope 缓存目录
        ms_cache_dir = os.environ.get('MODELSCOPE_CACHE', os.path.expanduser('~/.cache/modelscope'))
        
        # 需要同步的模型映射：ModelScope路径 -> HuggingFace路径
        models_to_sync = {
            "facebook/w2v-bert-2.0": ("facebook/w2v-bert-2.0", "models--facebook--w2v-bert-2.0"),
            "amphion/MaskGCT": ("amphion/MaskGCT", "models--amphion--MaskGCT"),
            "funasr/campplus": ("funasr/campplus", "models--funasr--campplus"),
        }
        
        # 检查并同步模型
        for model_name, (ms_path, hf_path) in models_to_sync.items():
            ms_model_path = Path(ms_cache_dir) / "hub" / ms_path
            hf_model_path = Path(hf_cache_dir) / "hub" / hf_path
            
            if hf_model_path.exists():
                logger.debug(f"✓ {model_name} 已在 HuggingFace 缓存中: {hf_model_path}")
            elif ms_model_path.exists():
                # ModelScope 有但 HuggingFace 没有，尝试同步
                logger.info(f"发现 {model_name} 在 ModelScope 缓存中，正在同步到 HuggingFace 缓存...")
                try:
                    # 确保目标目录存在
                    hf_model_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # 创建符合 HuggingFace 标准的缓存结构
                    import hashlib
                    snapshot_hash = hashlib.md5(str(ms_model_path.resolve()).encode()).hexdigest()[:8]
                    snapshots_dir = hf_model_path / "snapshots"
                    snapshots_dir.mkdir(parents=True, exist_ok=True)
                    snapshot_path = snapshots_dir / snapshot_hash
                    
                    if not snapshot_path.exists():
                        ms_model_path_abs = ms_model_path.resolve()
                        try:
                            snapshot_path.symlink_to(ms_model_path_abs)
                            logger.info(f"✓ 已创建 snapshot 符号链接: {snapshot_path}")
                        except (OSError, PermissionError):
                            import shutil
                            shutil.copytree(ms_model_path, snapshot_path, dirs_exist_ok=True)
                            logger.info(f"✓ 已复制到 snapshot 目录: {snapshot_path}")
                    
                    # 创建 refs/main 文件
                    refs_dir = hf_model_path / "refs"
                    refs_dir.mkdir(exist_ok=True)
                    main_ref = refs_dir / "main"
                    main_ref.write_text(snapshot_hash)
                    logger.info(f"✓ 已创建 HuggingFace 标准缓存结构")
                except Exception as e:
                    logger.error(f"✗ 同步 {model_name} 失败: {e}")
                    logger.warning(f"  请运行同步脚本: python3 scripts/sync_modelscope_to_huggingface.py")
            else:
                if model_name == "facebook/w2v-bert-2.0":
                    logger.warning(f"⚠ {model_name} 未在本地缓存中找到")
                    logger.warning(f"  如果网络不可达，模型初始化可能会失败")
                    logger.warning(f"  建议在有网络时预先下载: huggingface-cli download {model_name}")
                elif model_name == "funasr/campplus":
                    logger.warning(f"⚠ {model_name} 未在本地缓存中找到")
                    logger.warning(f"  注意：此模型在 ModelScope 上不可用，只能从 HuggingFace 下载")
                    logger.warning(f"  如果网络不可达，请在有网络时预先下载:")
                    logger.warning(f"    export HF_ENDPOINT=https://hf-mirror.com  # 国内用户推荐")
                    logger.warning(f"    huggingface-cli download funasr/campplus --include campplus_cn_common.bin")
        
        original_hf_offline = os.environ.get('HF_HUB_OFFLINE', None)
        original_hf_local = os.environ.get('TRANSFORMERS_OFFLINE', None)
        original_hf_home = os.environ.get('HF_HOME', None)
        
        try:
            # 启用离线模式，优先使用本地缓存
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            # 确保使用正确的缓存目录
            if original_hf_home:
                os.environ['HF_HOME'] = original_hf_home
            logger.info("已设置离线模式环境变量，将优先使用本地缓存")
            logger.info("如果缓存中缺少模型文件，请在有网络时预先下载")
            
            # 在导入 IndexTTS2 之前，确保离线模式已设置
            # 并修复 ModelScope 路径问题（点号转换为下划线）
            ms_cache_dir = os.environ.get('MODELSCOPE_CACHE', os.path.expanduser('~/.cache/modelscope'))
            w2v_orig = Path(ms_cache_dir) / "hub" / "facebook" / "w2v-bert-2.0"
            w2v_alt = Path(ms_cache_dir) / "hub" / "facebook" / "w2v-bert-2___0"
            
            # 确保 ModelScope 转换路径有必要的文件
            if w2v_orig.exists() and w2v_alt.exists():
                needed_files = ['model.safetensors', 'preprocessor_config.json']
                for fname in needed_files:
                    orig_file = w2v_orig / fname
                    alt_file = w2v_alt / fname
                    if orig_file.exists() and not alt_file.exists():
                        try:
                            import shutil
                            shutil.copy2(orig_file, alt_file)
                            logger.debug(f"已复制 {fname} 到 ModelScope 转换路径")
                        except:
                            pass
            
            # 添加 index-tts 到 Python 路径
            project_root = Path(__file__).parent.parent.parent
            indextts_path = project_root / "index-tts"
            if indextts_path.exists():
                if str(indextts_path) not in sys.path:
                    sys.path.insert(0, str(indextts_path))
                logger.info(f"添加 index-tts 路径: {indextts_path}")
            else:
                logger.warning(f"index-tts 目录不存在: {indextts_path}")
                logger.warning("尝试使用已安装的 indextts 包")
            
            # 使用 IndexTTS2 官方代码加载模型
            logger.info("使用 IndexTTS2 官方代码加载模型（本地模式）...")
            # 导入前再次确认离线模式
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            from indextts.infer_v2 import IndexTTS2
            
            # 确保使用绝对路径，并移除尾随斜杠
            config_path_abs = Path(config_path).resolve()
            model_dir_abs = Path(self.model_dir).resolve()
            
            # 移除路径末尾的斜杠，避免 transformers 误判
            model_dir_str = str(model_dir_abs).rstrip('/\\')  # 同时移除 Unix 和 Windows 路径分隔符
            config_path_str = str(config_path_abs)
            
            logger.info(f"使用配置文件: {config_path_str}")
            logger.info(f"使用模型目录: {model_dir_str}")
            
            # 验证模型目录存在必要的文件
            model_dir_path = Path(model_dir_str)
            if not model_dir_path.exists():
                raise FileNotFoundError(f"模型目录不存在: {model_dir_str}")
            
            # 检查 Qwen 模型路径（如果配置中有的话）
            # 根据 IndexTTS2 的结构，qwen_emo_path 是相对路径，指向 model_dir 下的子目录
            qwen_emo_expected = model_dir_path / "qwen0.6bemo4-merge"
            if qwen_emo_expected.exists():
                qwen_emo_path_str = str(qwen_emo_expected).rstrip('/\\')
                logger.info(f"✓ 检测到 Qwen 模型路径: {qwen_emo_path_str}")
                # 验证关键文件是否存在
                tokenizer_config = qwen_emo_expected / "tokenizer_config.json"
                if tokenizer_config.exists():
                    logger.info(f"  - tokenizer_config.json 存在")
                else:
                    logger.warning(f"  - tokenizer_config.json 不存在，可能会影响加载")
            else:
                logger.warning(f"未找到预期的 Qwen 模型路径: {qwen_emo_expected}")
                # 列出实际存在的子目录，帮助调试
                logger.info(f"模型目录下的子目录:")
                try:
                    for item in model_dir_path.iterdir():
                        if item.is_dir():
                            logger.info(f"  - {item.name}/")
                except Exception as e:
                    logger.warning(f"无法列出目录内容: {e}")
            
            # 初始化 IndexTTS2（本地模式）
            logger.info("正在初始化 IndexTTS2（使用本地模型路径）...")
            try:
                self.tts_model = IndexTTS2(
                    cfg_path=config_path_str,
                    model_dir=model_dir_str,  # 使用清理过的路径，确保没有尾随斜杠
                    use_fp16=self.config.get('use_fp16', False),
                    device=self.device,
                    use_cuda_kernel=self.config.get('use_cuda_kernel', None),
                )
                
                logger.info("✓ IndexTTS2 模型加载成功")
            except (OSError, ConnectionError, Exception) as e:
                error_msg = str(e)
                if "huggingface.co" in error_msg or "Network is unreachable" in error_msg:
                    logger.error("=" * 60)
                    logger.error("网络连接失败：无法从 HuggingFace 下载模型")
                    logger.error("=" * 60)
                    logger.error("解决方案：")
                    logger.error("1. 在有网络的环境中预先下载所需模型：")
                    logger.error("   huggingface-cli download facebook/w2v-bert-2.0")
                    logger.error("   huggingface-cli download amphion/MaskGCT")
                    logger.error("   huggingface-cli download funasr/campplus")
                    logger.error("")
                    logger.error("2. 或者使用代理/镜像访问 HuggingFace")
                    logger.error("")
                    logger.error("3. 或者临时启用网络连接以下载缺失的模型")
                    logger.error("=" * 60)
                raise
        finally:
            # 恢复原始环境变量
            if original_hf_offline is None:
                os.environ.pop('HF_HUB_OFFLINE', None)
            else:
                os.environ['HF_HUB_OFFLINE'] = original_hf_offline
            
            if original_hf_local is None:
                os.environ.pop('TRANSFORMERS_OFFLINE', None)
            else:
                os.environ['TRANSFORMERS_OFFLINE'] = original_hf_local
    
    def synthesize(
        self,
        text: str,
        reference_audio: Optional[np.ndarray] = None,
        reference_audio_path: Optional[str] = None,
        emotion: Optional[str] = None,
        emotion_strength: float = 1.0,
        speed: Optional[float] = None,
        **kwargs
    ) -> np.ndarray:
        """
        语音合成
        
        Args:
            text: 要合成的文本
            reference_audio: 参考音频数组 (可选)
            reference_audio_path: 参考音频文件路径 (可选)
            emotion: 情感标签 (可选): 'neutral', 'happiness', 'sadness', 'anger', etc.
            emotion_strength: 情感强度 (0-1)
            speed: 语速倍率（注意：IndexTTS2 可能不支持动态语速调整）
            **kwargs: 其他参数
        
        Returns:
            audio: 音频数组
        """
        try:
            import tempfile
            import soundfile as sf
            import torch
            
            # 准备参考音频路径
            temp_file = None
            if reference_audio_path is None and reference_audio is not None:
                # 保存临时文件
                temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                sf.write(temp_file.name, reference_audio, self.sample_rate)
                reference_audio_path = temp_file.name
            
            # 如果没有提供参考音频，尝试使用默认参考音频
            if reference_audio_path is None:
                # 检查是否有默认参考音频（可以从配置中读取）
                default_ref_audio = self.config.get('default_reference_audio')
                
                # 如果是相对路径，转换为绝对路径
                if default_ref_audio:
                    if not os.path.isabs(default_ref_audio):
                        project_root = Path(__file__).parent.parent.parent
                        default_ref_audio = str(project_root / default_ref_audio)
                
                if default_ref_audio and os.path.exists(default_ref_audio):
                    reference_audio_path = default_ref_audio
                    logger.info(f"使用默认参考音频: {reference_audio_path}")
                else:
                    # 尝试自动查找参考音频
                    project_root = Path(__file__).parent.parent.parent
                    possible_paths = [
                        project_root / "index-tts" / "examples" / "test_voice.wav",
                        project_root / "data" / "audio_input" / "input_20251103_110735_0000.wav",
                        project_root / "index-tts" / "examples" / "voice_01.wav",
                    ]
                    
                    for path in possible_paths:
                        if path.exists() and path.stat().st_size > 1024:
                            try:
                                with open(path, 'rb') as f:
                                    header = f.read(12)
                                    if header[:4] == b'RIFF' and header[8:12] == b'WAVE':
                                        reference_audio_path = str(path)
                                        logger.info(f"自动找到参考音频: {reference_audio_path}")
                                        break
                            except:
                                continue
                    
                    if reference_audio_path is None:
                        logger.warning("未提供参考音频，IndexTTS2 需要参考音频才能工作")
                        logger.warning("请提供 reference_audio 或 reference_audio_path")
                        raise ValueError("IndexTTS2 需要参考音频才能进行语音合成")
            
            # 创建临时输出文件
            output_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            output_path = output_temp.name
            output_temp.close()
            
            try:
                # 调用 IndexTTS2 的 infer 方法
                # infer 方法需要: spk_audio_prompt, text, output_path
                result = self.tts_model.infer(
                    spk_audio_prompt=reference_audio_path,
                    text=text,
                    output_path=output_path,
                    emo_audio_prompt=None,  # 可以后续支持
                    emo_alpha=emotion_strength if emotion else 1.0,
                    emo_vector=None,  # 可以后续支持
                    verbose=False,
                    **kwargs
                )
                
                # 读取生成的音频文件
                if os.path.exists(output_path):
                    audio, sr = sf.read(output_path)
                    # 确保采样率匹配
                    if sr != self.sample_rate:
                        import librosa
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
                    
                    # 转换为 float32
                    if audio.dtype != np.float32:
                        audio = audio.astype(np.float32)
                    
                    logger.info(f"合成成功，音频长度: {len(audio)/self.sample_rate:.2f}秒")
                    return audio
                else:
                    raise FileNotFoundError(f"生成的音频文件不存在: {output_path}")
                
            finally:
                # 清理临时文件
                if temp_file is not None and os.path.exists(temp_file.name):
                    try:
                        os.unlink(temp_file.name)
                    except:
                        pass
                if os.path.exists(output_path):
                    try:
                        os.unlink(output_path)
                    except:
                        pass
            
        except Exception as e:
            logger.error(f"语音合成失败: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            raise
    
    def synthesize_to_file(
        self,
        text: str,
        output_path: str,
        **kwargs
    ) -> str:
        """
        合成语音并保存到文件
        
        Args:
            text: 要合成的文本
            output_path: 输出文件路径
            **kwargs: 其他参数
        
        Returns:
            output_path: 输出文件路径
        """
        import soundfile as sf
        
        # 生成音频
        audio = self.synthesize(text, **kwargs)
        
        # 保存
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, audio, self.sample_rate)
        
        logger.info(f"音频已保存到: {output_path}")
        return output_path
    
    def clone_voice(
        self,
        reference_audio: np.ndarray,
        text: str,
        **kwargs
    ) -> np.ndarray:
        """
        零样本语音克隆
        
        Args:
            reference_audio: 参考音频
            text: 要合成的文本
            **kwargs: 其他参数
        
        Returns:
            audio: 合成的音频
        """
        return self.synthesize(
            text=text,
            reference_audio=reference_audio,
            **kwargs
        )
    
    def set_speed(self, speed: float):
        """设置语速"""
        self.speed = speed
        logger.info(f"设置语速: {speed}")


# 改进的官方模型包装器：完整的接口兼容
class IndexTTS2ModelScopeWrapper(IndexTTS2ModelScope):
    """
    改进的官方模型包装器
    提供与复现模型完全兼容的接口
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 默认参数
        self.speaker_id = config.get('speaker_id', 0)
        self.emotion = config.get('emotion', 'neutral')
        self.pitch = config.get('pitch', 1.0)
    
    def synthesize(
        self,
        text: str,
        timbre_prompt: Optional[np.ndarray] = None,
        style_prompt: Optional[np.ndarray] = None,
        emotion: Optional[str] = None,
        target_duration: Optional[float] = None,
        speed: Optional[float] = None,
        speaker_id: Optional[int] = None,
        reference_audio: Optional[np.ndarray] = None,
        reference_audio_path: Optional[str] = None,
        **kwargs
    ) -> np.ndarray:
        """
        语音合成（兼容复现模型接口）
        
        Args:
            text: 要合成的文本
            timbre_prompt: 音色参考音频（兼容参数）
            style_prompt: 风格参考音频（兼容参数）
            emotion: 情感标签
            target_duration: 目标时长（官方模型可能不支持）
            speed: 语速倍率
            speaker_id: 说话人ID（兼容参数）
            reference_audio: 参考音频（官方接口）
            reference_audio_path: 参考音频路径（官方接口）
            **kwargs: 其他参数
        
        Returns:
            audio: 音频数组
        """
        # 统一参考音频处理
        ref_audio = reference_audio or timbre_prompt
        ref_path = reference_audio_path
        
        # 如果没有参考音频，尝试使用style_prompt
        if ref_audio is None and style_prompt is not None:
            ref_audio = style_prompt
        
        # 使用默认情感
        if emotion is None:
            emotion = self.emotion
        
        # 调用父类方法
        return super().synthesize(
            text=text,
            reference_audio=ref_audio,
            reference_audio_path=ref_path,
            emotion=emotion,
            speed=speed or self.speed,
            **kwargs
        )
    
    def synthesize_batch(
        self,
        texts: list,
        **kwargs
    ) -> list:
        """
        批量合成（兼容接口）
        
        Args:
            texts: 文本列表
            **kwargs: 其他参数
        
        Returns:
            audios: 音频数组列表
        """
        audios = []
        for i, text in enumerate(texts):
            logger.info(f"批量合成进度: {i+1}/{len(texts)}")
            audio = self.synthesize(text, **kwargs)
            audios.append(audio)
        return audios
    
    def set_speaker(self, speaker_id: int):
        """设置说话人（兼容接口）"""
        self.speaker_id = speaker_id
        logger.info(f"设置说话人ID: {speaker_id}（官方模型通过参考音频控制）")
    
    def set_emotion(self, emotion: str):
        """设置默认情感"""
        self.emotion = emotion
        logger.info(f"设置默认情感: {emotion}")
    
    def set_pitch(self, pitch: float):
        """设置音高（兼容接口，官方模型可能不支持）"""
        self.pitch = pitch
        logger.info(f"设置音高: {pitch}（注意：官方模型可能不支持此参数）")


# 兼容性别名
IndexTTSModule = IndexTTS2ModelScopeWrapper
