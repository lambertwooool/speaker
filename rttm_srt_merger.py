from dataclasses import dataclass
from typing import List
from bisect import bisect_right

@dataclass
class RTTMEntry:
    speaker: str
    start: float
    end: float

@dataclass
class SRTEntry:
    index: int
    start: float
    end: float
    text: str

class RTTM_SRT_Merger:
    def __init__(self, rttm_path: str, srt_path: str):
        self.rttm_entries = self._parse_rttm(rttm_path)
        self.srt_entries = self._parse_srt(srt_path)
        self._preprocess_rttm()

    def _preprocess_rttm(self):
        """预处理RTTM数据，建立时间索引"""
        self.rttm_entries.sort(key=lambda x: x.start)
        self.rttm_starts = [r.start for r in self.rttm_entries]
        self.rttm_ends = [r.end for r in self.rttm_entries]

    def _parse_rttm(self, path: str) -> List[RTTMEntry]:
        entries = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                try:
                    speaker = parts[5]
                    start = float(parts[3])
                    duration = float(parts[4])
                    entries.append(RTTMEntry(
                        speaker=f"[{speaker}]",
                        start=start,
                        end=start + duration
                    ))
                except (IndexError, ValueError):
                    continue
        return entries

    def _parse_srt(self, path: str) -> List[SRTEntry]:
        entries = []
        with open(path, 'r', encoding='utf-8') as f:
            blocks = f.read().split('\n\n')
            for block in blocks:
                if not block.strip():
                    continue
                lines = block.split('\n')
                try:
                    index = int(lines[0])
                    timecode = lines[1]
                    text = ' '.join(lines[2:])
                    start_str, end_str = timecode.split(' --> ')
                    start = self._time_to_seconds(start_str)
                    end = self._time_to_seconds(end_str)
                    entries.append(SRTEntry(index, start, end, text))
                except (IndexError, ValueError):
                    continue
        return entries

    def _time_to_seconds(self, time_str: str) -> float:
        h, m, s = time_str.split(':')
        s, ms = s.split(',')
        return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000

    def _find_overlapping_speakers(self, srt_entry: SRTEntry) -> List[str]:
        speakers = []
        seen = set()
        
        end_idx = bisect_right(self.rttm_starts, srt_entry.end)
        
        # 遍历所有可能重叠的条目
        for rttm in self.rttm_entries[:end_idx]:
            # 检查条目是否实际重叠
            if rttm.end >= srt_entry.start:
                overlap = min(srt_entry.end, rttm.end) - max(srt_entry.start, rttm.start)
                if overlap > 0.1:
                    if rttm.speaker not in seen:
                        seen.add(rttm.speaker)
                        speakers.append(rttm.speaker)
        
        return speakers if speakers else ["[UNKNOWN]"]

    def _is_overlap(self, srt: SRTEntry, rttm: RTTMEntry) -> bool:
        """判断时间是否重叠（包含部分重叠）"""
        return not (srt.end <= rttm.start or srt.start >= rttm.end)

    def merge(self, output_path: str, min_overlap: float = 0.1):
        """生成合并后的SRT"""
        output = []
        for srt in self.srt_entries:
            speakers = self._find_overlapping_speakers(srt)
            speaker_tags = "".join(speakers)
            new_text = f"{speaker_tags} {srt.text}"
            
            start_str = self._seconds_to_time(srt.start)
            end_str = self._seconds_to_time(srt.end)
            
            output.append(
                f"{srt.index}\n"
                f"{start_str} --> {end_str}\n"
                f"{new_text}\n"
            )
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output))

    def _seconds_to_time(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace('.', ',')