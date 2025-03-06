import argparse
from datetime import datetime
from speaker import SpeakerIdentifier
from srt import SRTGenerator
from rttm_srt_merger import RTTM_SRT_Merger
import os

if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="音频处理脚本")
    parser.add_argument("audio_file", type=str, help="输入的音频文件路径")
    parser.add_argument("--rttm_file", type=str, default=None, help="已有的 RTTM 文件路径，若提供则跳过 RTTM 生成")
    parser.add_argument("--srt_file", type=str, default=None, help="已有的 SRT 文件路径，若提供则跳过 SRT 生成")
    parser.add_argument("--output_file", type=str, default=None, help="合并后的输出文件路径，若不提供则使用输入文件名修改后缀")

    # 解析命令行参数
    args = parser.parse_args()
    
    base_name, _ = os.path.splitext(args.audio_file)
    # 处理输出文件名
    if args.output_file is None:
        # 获取当前时间并格式化为目录名
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("output", current_time)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

    # 初始化相关类
    speaker_identifier = SpeakerIdentifier()
    generator = SRTGenerator()

    # 处理 RTTM 文件
    if args.rttm_file is None:
        diarization = speaker_identifier.diarization(args.audio_file, add_new_speaker=True)
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            print(f"start={turn.start:.3f}s end={turn.end:.3f}s duration={turn.end - turn.start:.1f}s speaker={speaker}")
        rttm_file = os.path.join(output_dir, f"{base_name}.rttm")
        speaker_identifier.write_rttm(rttm_file, diarization)
    else:
        rttm_file = args.rttm_file

    # 处理 SRT 文件
    if args.srt_file is None:
        srt_dict = generator.transcribe_audio(args.audio_file, word_timestamps=True)
        srt_file = os.path.join(output_dir, f"{base_name}.srt")
        generator.save_srt(srt_dict, srt_file)
    else:
        srt_file = args.srt_file

    # 处理输出文件名
    if args.output_file is None:
        output_file = os.path.join(output_dir, f"{base_name}_merge.srt")
    else:
        output_file = args.output_file

    # 合并 RTTM 和 SRT 文件
    merger = RTTM_SRT_Merger(
        rttm_path=rttm_file,
        srt_path=srt_file
    )
    merger.merge(output_file)

    print(f"合并后的文件已保存到 {output_file}")