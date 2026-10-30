"""Make a captioned <60-second edit from real localhost browser frames.

Requires Pillow and FFmpeg; never generates or modifies model outputs. A
capture.json with status=done and the matching saved run are required.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageStat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capstone_demo.video_storyboard import build_storyboard, concat_manifest


def mean_difference(a, b):
    with Image.open(a) as first, Image.open(b) as second:
        left = first.convert('L').resize((80, 48))
        right = second.convert('L').resize((80, 48))
        return ImageStat.Stat(ImageChops.difference(left, right)).mean[0]


def caption_frame(source, destination, caption, number, font):
    with Image.open(source) as image:
        image = image.convert('RGB')
        canvas = Image.new('RGB', (image.width, image.height + 84), '#1F3A5F')
        canvas.paste(image, (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.rounded_rectangle((22, image.height + 18, 90, image.height + 64), radius=8, fill='#E0A526')
        draw.text((39, image.height + 27), f'{number:02d}', fill='#1A1A1A', font=font)
        draw.text((112, image.height + 26), caption, fill='#FFFFFF', font=font)
        canvas.save(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, default=55)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f'Refusing to overwrite {args.output}')
    capture = json.loads((args.capture / 'capture.json').read_text())
    if capture.get('status') != 'done' or not capture.get('run_id'):
        parser.error('Capture is not a completed real run')
    result = json.loads((args.runs / capture['run_id'] / 'result.json').read_text())
    if result['status'] != 'done' or any(v['status'] != 'done' for v in result['variants']):
        parser.error('Saved model run is not fully done')
    if result['input']['sha256'] != capture['input_sha256']:
        parser.error('Capture input hash does not match the saved run')
    records = json.loads((args.capture / 'timeline.json').read_text())
    frames = [args.capture / 'frames' / record['file'] for record in records]
    if len(frames) != capture['frame_count'] or not all(p.is_file() for p in frames):
        parser.error('Incomplete frame timeline')
    clips = build_storyboard(records,
        difference=lambda i, j: mean_difference(frames[i], frames[j]),
        max_seconds=args.max_seconds)
    duration = sum(c['seconds'] for c in clips)
    if duration >= 60:
        parser.error(f'Edit is too long: {duration:.2f}s')
    font_path = subprocess.run(['fc-match', '-f', '%{file}', 'Times New Roman'],
                               capture_output=True, text=True, check=True).stdout.strip()
    font = ImageFont.truetype(font_path, 25)
    edit_dir = args.capture / 'compact-edit'
    edit_dir.mkdir(exist_ok=False)
    annotated = []
    for number, clip in enumerate(clips, 1):
        path = edit_dir / f'keyframe_{number:03d}.png'
        caption_frame(frames[clip['index']], path, clip['caption'], number, font)
        annotated.append(path)
    manifest = concat_manifest(
        [{'index': i, 'seconds': clip['seconds']} for i, clip in enumerate(clips)], annotated)
    (edit_dir / 'ffmpeg-concat.txt').write_text(manifest)
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        parser.error('FFmpeg not found')
    command = [ffmpeg, '-loglevel', 'error', '-safe', '0', '-f', 'concat',
               '-i', str(edit_dir / 'ffmpeg-concat.txt'), '-vf',
               'fps=24,scale=1280:-2,pad=ceil(iw/2)*2:ceil(ih/2)*2',
               '-c:v', 'libx264', '-preset', 'medium', '-crf', '26',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(args.output)]
    subprocess.run(command, check=True, timeout=300)
    probe = json.loads(subprocess.check_output(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration,size:stream=codec_name,width,height',
         '-of', 'json', str(args.output)], text=True))
    actual_duration = float(probe['format']['duration'])
    if not 0 < actual_duration < 60:
        raise RuntimeError(f'Encoded video duration out of bounds: {actual_duration}')
    provenance = {
        'source': 'real laptop localhost Chromium capture; unchanged model outputs',
        'run_id': capture['run_id'], 'input_sha256': capture['input_sha256'],
        'source_frame_count': len(frames), 'selected_keyframes': len(clips),
        'waiting_shortened': True, 'captions': 'Added to an 84-pixel navy strip below genuine browser frames',
        'theme': 'NMIMS Capstone PPTX navy #1F3A5F, teal #2E7D8C, gold #E0A526',
        'clip_timeline': [{**clip, 'source_utc': records[clip['index']]['utc']} for clip in clips],
        'video': str(args.output), 'bytes': args.output.stat().st_size,
        'sha256': hashlib.file_digest(args.output.open('rb'), 'sha256').hexdigest(),
        'ffprobe': probe, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Synthetic smoke page, not an accuracy benchmark or all-input guarantee',
    }
    (args.capture / 'compact-video-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'video': str(args.output), 'duration': actual_duration,
                      'bytes': args.output.stat().st_size,
                      'keyframes': len(clips), 'run_id': capture['run_id']}))


if __name__ == '__main__':
    main()
