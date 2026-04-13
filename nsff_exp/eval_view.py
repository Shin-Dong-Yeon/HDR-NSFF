import os
import glob
import subprocess

# 현재 작업 디렉토리 기준 (필요하면 변경하세요)
base_dir = "/home/shindy/projects/hdr-4d/logs/hdr-nsff/_gopro/250913/bag_swing_3_1_i2_of_hdr_250913_F00-36/evaluation/hdr_tm"
import pdb; pdb.set_trace()

# PNG 파일들이 들어 있는 폴더들 (00000, 00001, ...)
folders = sorted([d for d in os.listdir(base_dir) if os.path.isdir(d)])

# 첫 폴더를 기준으로 프레임 이름들 추출
frame_names = sorted([os.path.basename(p) for p in glob.glob(os.path.join(folders[0], "*.png"))])

for frame in frame_names:
    print(f"프레임 {frame} 영상 생성 중...")

    # 리스트 파일 생성
    list_file = f"frames_{frame}.txt"
    with open(list_file, "w") as f:
        for folder in folders:
            img_path = os.path.join(folder, frame)
            if os.path.exists(img_path):
                f.write(f"file '{img_path}'\n")

    # ffmpeg 실행
    output_video_name = f"video_{os.path.splitext(frame)[0]}.mp4"
    output_video = os.path.join(base_dir, output_video_name)
    
    cmd = [
        "ffmpeg",
        "-y",  # 덮어쓰기
        "-f", "concat",
        "-safe", "0",
        "-i", list_file,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_video,
    ]
    subprocess.run(cmd)

print("✅ 모든 영상 생성 완료")