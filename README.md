# 音频处理脚本：生成带演讲人的字幕文件
本项目提供的音频处理脚本，旨在从音频文件出发，生成包含演讲人信息的字幕文件，适用于会议记录、视频字幕添加等场景，助力高效的语音内容整理与展示。

## 一、功能概述
1. **说话人分割**：精准识别音频中的不同演讲人，将每个演讲人的发言时间段及对应身份信息提取出来，并保存为行业标准的RTTM（Rich Transcription Time Marked）格式文件。
2. **音频转录**：把音频中的语音内容准确转录为文本，并按照SRT（SubRip Text）字幕格式生成字幕文件，记录每个单词或语句出现的时间戳。
3. **结果合并**：巧妙地将说话人分割结果与音频转录结果进行融合，生成最终的SRT字幕文件。在这个文件里，每一段字幕都明确标注了对应的演讲人，极大地提升了字幕的可读性与信息完整性。

## 二、安装依赖
在运行脚本前，请确保安装了所需的依赖库。可使用以下命令进行安装：
```bash
pip install -r requirements.txt
```
PyTorch 的安装需要根据你的系统环境（如操作系统、CUDA 版本等）进行选择。你可以访问 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/) 来获取适合你环境的安装命令。

## 三、模型下载
在运行脚本前，你需要下载两个模型文件，并将其放置在 `models` 目录下：
1. 从 [https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM/resolve/main/pytorch_model.bin](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM/resolve/main/pytorch_model.bin) 下载模型文件。下载完成后，将其重命名为 `wespeaker-voxceleb-resnet34-LM.bin`。
2. 从 [https://huggingface.co/pyannote/segmentation-3.0/resolve/main/pytorch_model.bin](https://huggingface.co/pyannote/segmentation-3.0/resolve/main/pytorch_model.bin) 下载模型文件。下载完成后，将其重命名为 `segmentation-3.0.bin`。

请确保 `models` 目录存在，你可以手动创建该目录：
```bash
mkdir models
```
然后将重命名后的两个模型文件移动到 `models` 目录中。

## 四、使用方法
### （一）命令行参数说明
1. `audio_file`：此为必需参数，用于指定输入的音频文件路径，支持常见的音频格式，如 `.wav`。
2. `--rttm_file`：该参数可选，若提供已有的RTTM文件路径，脚本将跳过说话人分割及RTTM文件生成步骤，直接使用该文件进行后续处理。
3. `--srt_file`：同样为可选参数，若指定已有的SRT文件路径，脚本会跳过音频转录及SRT文件生成环节，采用此文件参与合并流程。
4. `--ignore_new_speaker`: 该参数用于控制是否忽略新说话人，默认为 `False`，即自动添加新说话人。
5. `--output_file`：此参数用于指定合并后的输出文件路径。若未提供，脚本将依据输入音频文件名，修改后缀生成默认的输出文件名。

新的说话人会添加到 /speaker_db 目录下，修改 speaker 的目录名称为你想要的人名，您也可以将多个 wav 文件放在同一用户下面，增强识别效果。

### （二）示例命令
1. **基本使用**
```bash
python script.py audio.wav
```
执行该命令，脚本会对 `audio.wav` 依次执行说话人分割、音频转录和结果合并操作，最终生成默认命名的带演讲人字幕文件。
2. **跳过RTTM生成**
```bash
python script.py audio.wav --rttm_file existing.rttm
```
此命令下，脚本将使用已有的 `existing.rttm` 文件，不再进行说话人分割及新RTTM文件的生成，直接进入后续流程。
3. **自定义输出文件名**
```bash
python script.py audio.wav --output_file custom_output.srt
```
该命令指定合并后的输出文件名为 `custom_output.srt`，方便用户按照需求自定义输出。

## 五、贡献与反馈
若在使用过程中发现问题，或有优化建议，欢迎提交Issues阐述问题，或通过Pull Requests提交代码改进方案。我们珍视每一份贡献，共同推动项目不断完善。