import datetime
import shutil
import torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional, Union
from pyannote.audio import Audio, Model, Inference
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.core import Annotation, Segment
import torchaudio

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpeakerIdentifier:
    def __init__(
            self,
            database_root: Union[str, Path] = "speaker_db",
            segmentation_model: str = "./models/segmentation-3.0.bin",
            embedding_model: str = "./models/wespeaker-voxceleb-resnet34-LM.bin",
            auth_token: str = None,
            device: str = None,
            sample_rate: int = 16000,
            min_duration: float = 0.5
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.database_root = Path(database_root)
        self.segmentation_model = segmentation_model
        self.embedding_model = Model.from_pretrained(embedding_model, use_auth_token=auth_token)
        self.inference = Inference(self.embedding_model, window="whole", device=self.device)
        self.embeddings_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self.sample_rate = sample_rate
        self.min_duration = min_duration

        self._load_speakers()

    @torch.no_grad()
    @torch.inference_mode()
    def diarization(self, audio_file: str, add_new_speaker: bool = False, threshold: float = 0.65):
        # 获取管道名称和参数
        param_config = {
            "clustering": {
                "method": "centroid",
                "min_cluster_size": 12,
                "threshold": 0.7045654963945799,
            },
            "segmentation": {
                "min_duration_off": 0.0,
            },
        }

        # 初始化管道
        pipeline = SpeakerDiarization(**{
            "segmentation": self.segmentation_model,
            "embedding": self.embedding_model
        })
        pipeline.instantiate(param_config)
        pipeline.to(self.device)

        diarization = pipeline(audio_file)

        audio_tool = Audio(sample_rate=16000, mono=True)
        # 获取文件实际时长
        info = torchaudio.info(audio_file)
        total_duration = info.num_frames / info.sample_rate

        speakers = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # 提取当前片段的音频波形
            start = max(0.0, turn.start)
            end = min(turn.end, total_duration)
            duration = end - start
            waveform, sr = audio_tool.crop(audio_file, Segment(start, end))

            if duration > 0.5:
                # 识别说话人（不自动添加新说话人）
                speaker_id = self.identify_speaker(
                    query_audio=waveform,
                    sample_rate=sr,
                    threshold=threshold
                )

            if speaker not in speakers:
                speakers[speaker] = {}
            speakers[speaker][speaker_id] = speakers[speaker].get(speaker_id, 0) + duration

        speakers_name = {s: max(speakers[s], key=speakers[s].get) for s in speakers}

        new_diarization = Annotation()

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if add_new_speaker and speakers_name.get(speaker) is None:
                waveform, sr = audio_tool.crop(audio_file, turn)
                speaker_id = self._add_new_speaker(waveform, sr)
                speakers_name[speaker] = speaker_id
            new_diarization[turn] = speakers_name[speaker] or speaker

        return new_diarization

    def _get_speaker_dir(self, speaker_id: str) -> Path:
        return self.database_root / speaker_id

    def _get_audio_dir(self, speaker_id: str) -> Path:
        return self._get_speaker_dir(speaker_id) / "audios"

    def _get_embeddings_dir(self, speaker_id: str) -> Path:
        embeddings_dir = self._get_speaker_dir(speaker_id) / "embeddings"
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        return embeddings_dir

    def _get_embedding_path(self, speaker_id: str, audio_name: str) -> Path:
        return self._get_embeddings_dir(speaker_id) / f"{audio_name}.npy"

    def _generate_embeddings(self, speaker_id: str) -> bool:
        """为指定说话人生成声纹文件"""
        audio_dir = self._get_audio_dir(speaker_id)

        # 检查音频目录是否存在
        if not audio_dir.exists():
            logger.warning(f"跳过 {speaker_id}: 缺少音频目录 {audio_dir}")
            return False

        # 收集所有音频文件
        audio_files = list(audio_dir.glob("*.*"))
        valid_exts = [".wav", ".flac", ".mp3"]  # 支持的音频格式
        audio_files = [f for f in audio_files if f.suffix.lower() in valid_exts]

        if not audio_files:
            logger.warning(f"跳过 {speaker_id}: 目录中没有有效音频文件")
            return False

        # 提取所有声纹
        success_count = 0
        for audio_path in audio_files:
            try:
                embedding = self.inference(str(audio_path))
                audio_name = audio_path.stem
                embedding_path = self._get_embedding_path(speaker_id, audio_name)
                np.save(embedding_path, embedding)
                self.embeddings_cache.setdefault(speaker_id, {})[audio_name] = embedding
                logger.info(f"成功处理并保存声纹: {audio_path.name}")
                success_count += 1
            except Exception as e:
                logger.error(f"处理失败 {audio_path}: {str(e)}")

        if success_count == 0:
            logger.error(f"无法生成 {speaker_id}: 所有音频处理失败")
            return False

        logger.info(f"已为 {speaker_id} 生成声纹文件，共 {success_count} 个样本")
        return True

    def _load_speakers(self) -> None:
        """加载或自动生成所有说话人数据"""
        # 确保根目录存在
        self.database_root.mkdir(parents=True, exist_ok=True)

        # 遍历所有说话人目录
        for speaker_dir in self.database_root.iterdir():
            if not speaker_dir.is_dir():
                continue

            speaker_id = speaker_dir.name
            audio_dir = self._get_audio_dir(speaker_id)
            if not audio_dir.exists():
                continue

            # 收集所有音频文件
            audio_files = list(audio_dir.glob("*.*"))
            valid_exts = [".wav", ".flac", ".mp3"]  # 支持的音频格式
            audio_files = [f for f in audio_files if f.suffix.lower() in valid_exts]

            for audio_path in audio_files:
                audio_name = audio_path.stem
                embedding_path = self._get_embedding_path(speaker_id, audio_name)
                # 情况1: 已存在声纹文件
                if embedding_path.exists():
                    try:
                        embedding = np.load(embedding_path)
                        self.embeddings_cache.setdefault(speaker_id, {})[audio_name] = embedding
                        logger.info(f"已加载预计算声纹: {speaker_id}/{audio_name}")
                    except Exception as e:
                        logger.error(f"加载失败 {speaker_id}/{audio_name}: {str(e)}")
                    continue

                # 情况2: 需要自动生成声纹
                logger.info(f"检测到缺失声纹: {speaker_id}/{audio_name}，尝试自动生成...")
                if self._generate_embeddings(speaker_id):
                    # 生成成功后重新加载
                    try:
                        embedding = np.load(embedding_path)
                        self.embeddings_cache.setdefault(speaker_id, {})[audio_name] = embedding
                    except Exception as e:
                        logger.error(f"加载生成的文件失败: {str(e)}")
                else:
                    logger.error(f"无法自动生成声纹: {speaker_id}/{audio_name}")

    def identify_speaker(
            self,
            query_audio: Union[str, Path, torch.Tensor],
            sample_rate: Optional[int] = None,
            threshold: float = 0.7
    ) -> Optional[str]:
        """识别说话人"""
        if isinstance(query_audio, (str, Path)):
            # 处理文件路径
            waveform, sr = torchaudio.load(str(query_audio))
            waveform = self._preprocess_waveform(waveform, sr, self.sample_rate)
        elif isinstance(query_audio, torch.Tensor):
            # 处理波形张量
            if self.sample_rate is None:
                raise ValueError("必须提供 sample_rate 以处理波形输入")
            waveform = self._preprocess_waveform(query_audio, sample_rate, self.sample_rate)
        else:
            raise TypeError("输入类型不支持，应为路径或张量")

        # 检查有效长度
        if not self._validate_length(waveform):
            logger.warning("音频长度不足，需要至少%.1f秒", self.min_duration)
            return None

        query_emb = self.inference({"waveform": waveform.to(self.inference.device), "sample_rate": self.sample_rate})

        best_match = None
        max_sim = 0.0
        for speaker_id, audio_embeddings in self.embeddings_cache.items():
            for audio_name, mean_emb in audio_embeddings.items():
                sim = self._cosine_similarity(query_emb, mean_emb)
                if sim > max_sim and sim > threshold:
                    max_sim = sim
                    best_match = speaker_id

        return best_match

    def _add_new_speaker(self, query_audio: Union[str, Path, torch.Tensor], sample_rate: int) -> str:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        new_speaker_id = f"speaker_{timestamp}_{len(self.embeddings_cache)}"
        audio_dir = self._get_audio_dir(new_speaker_id)
        audio_dir.mkdir(parents=True, exist_ok=True)

        try:
            dest_path = f"{audio_dir}/{timestamp}.wav"
            if isinstance(query_audio, (str, Path)):
                # 直接复制原文件
                shutil.copyfile(query_audio, dest_path)
            else:
                # 保存张量为音频文件
                torchaudio.save(str(dest_path), query_audio, sample_rate)

            if self._generate_embeddings(new_speaker_id):
                audio_name = Path(dest_path).stem
                embedding_path = self._get_embedding_path(new_speaker_id, audio_name)
                embedding = np.load(embedding_path)
                self.embeddings_cache.setdefault(new_speaker_id, {})[audio_name] = embedding
                logger.info(f"新增说话人: {new_speaker_id}")
            else:
                logger.error("生成声纹失败")
        except Exception as e:
            logger.error(f"添加失败: {str(e)}")
        
        return new_speaker_id

    def _preprocess_waveform(self,
                             waveform: torch.Tensor,
                             original_rate: int,
                             target_rate: int
                             ) -> torch.Tensor:
        """音频预处理流水线"""
        # 重采样
        if original_rate != target_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=original_rate,
                new_freq=target_rate
            )

        # 转为单声道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        return waveform

    def _validate_length(self, waveform: torch.Tensor) -> bool:
        """验证音频长度有效性"""
        required_samples = int(self.min_duration * self.sample_rate)
        return waveform.shape[1] >= required_samples

    def write_rttm(self, file, annotation):
        """
        将 pyannote.core.Annotation 对象保存为 RTTM 格式的文件

        :param file: 文件对象或文件路径，用于保存 RTTM 数据
        :param annotation: pyannote.core.Annotation 对象，包含说话人分割信息
        """
        if isinstance(file, str):
            with open(file, 'w', encoding="utf-8") as f:
                self._write_rttm_to_file(f, annotation)
        else:
            self._write_rttm_to_file(file, annotation)

    def _write_rttm_to_file(self, file, annotation):
        """
        实际将 Annotation 对象写入文件的方法

        :param file: 文件对象
        :param annotation: pyannote.core.Annotation 对象
        """
        uri = annotation.uri if annotation.uri else 'unknown'
        for segment, track, label in annotation.itertracks(yield_label=True):
            start_time = segment.start
            duration = segment.duration
            line = f'SPEAKER {uri} 1 {start_time:.3f} {duration:.3f} {label} <NA> <NA> <NA> <NA>\n'
            file.write(line)

    @staticmethod
    def _cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """计算余弦相似度"""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
