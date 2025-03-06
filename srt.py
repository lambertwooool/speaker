import torch
import whisper
from typing import Optional, Union

class SRTGenerator:
    """Whisper 字幕生成器（支持 SRT 格式输出）"""
    
    def __init__(self, 
                 model_name: str = "small",
                 language: Optional[str] = None,
                 device: str = None,
                 verbose: bool = False):
        """
        初始化字幕生成器
        
        参数：
            model_name: 预训练模型名称 (tiny/base/small/medium/large)
            language: 目标语言代码 (zh/en/ja等)，None 表示自动检测
            device: 推理设备
            verbose: 是否显示详细处理信息
        """
        self.model_name = model_name
        self.language = language
        self.device = torch.device(device or( "cuda" if torch.cuda.is_available() else "cpu"))
        self.verbose = verbose
        
        # 加载模型
        self.model = self._load_model()

    def _determine_device(self, device: str=None) -> str:
        """自动检测可用设备"""
        if device == "auto":
            return "cuda" if whisper.utils.available_flash() else "cpu"
        return device

    def _load_model(self) -> whisper.Whisper:
        """加载指定 Whisper 模型"""
        try:
            model = whisper.load_model(
                name=self.model_name,
                device=self.device,
                in_memory=True  # 减少磁盘占用
            )
            if self.verbose:
                print(f"✅ 已加载模型 {self.model_name}，推理设备：{self.device.upper()}")
            return model
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {str(e)}")

    def transcribe_audio(self, 
                        audio_path: str,
                        word_timestamps: bool = False) -> dict:
        """
        执行音频转录
        
        参数：
            audio_path: 音频文件路径
            word_timestamps: 是否生成逐字时间戳
        """
        try:
            result = self.model.transcribe(
                audio=audio_path,
                language=self.language,
                verbose=self.verbose,
                word_timestamps=word_timestamps
            )
            return result
        except FileNotFoundError:
            raise ValueError(f"音频文件不存在: {audio_path}")
        except Exception as e:
            raise RuntimeError(f"转录失败: {str(e)}")

    def save_srt(self, 
                    srt_dict: dict,
                    output_path: Optional[str] = None,
                    word_level: bool = False) -> Union[str, None]:
        """
        生成 SRT 字幕内容
        
        参数：
            srt_dict: 转录结果字典
            output_path: 输出文件路径（None 则返回字符串）
            word_level: 是否使用逐字时间戳
        """
            
        srt_content = []
        segments = self._get_segments(srt_dict, word_level)
        
        for idx, seg in enumerate(segments, 1):
            start = seg["start"]
            end = seg["end"]
            text = seg["text"].strip()
            
            srt_content.append(
                f"{idx}\n"
                f"{self._format_time(start)} --> {self._format_time(end)}\n"
                f"{text}\n"
            )

        full_content = "\n".join(srt_content)
        
        if output_path:
            self._save_srt(full_content, output_path)
            if self.verbose:
                print(f"SRT 文件已保存至：{output_path}")
        else:
            return full_content

    def _get_segments(self, srt_dict: dict, word_level: bool) -> list:
        """获取时间戳分段"""
        if word_level and "words" in srt_dict:
            return srt_dict["words"]
        return srt_dict["segments"]

    @staticmethod
    def _format_time(seconds: float) -> str:
        """格式化时间戳 (HH:MM:SS,mmm)"""
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}".replace(".", ",")

    def _save_srt(self, content: str, path: str):
        """保存 SRT 文件"""
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        except IOError as e:
            raise RuntimeError(f"文件保存失败: {str(e)}")

    def __repr__(self):
        return (f"<WhisperSRTGenerator model={self.model_name} "
                f"lang={self.language} device={self.device}>")
