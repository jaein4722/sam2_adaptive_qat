#!/usr/bin/env python3
# training/scripts/sav_frame_extraction.py
import cv2, os, json, argparse, multiprocessing as mp
from pathlib import Path
from typing import List, Tuple

def list_videos(video_dir: Path, exts=(".mp4", ".mov", ".avi", ".mkv")) -> List[Path]:
    return sorted([p for p in video_dir.rglob("*") if p.suffix.lower() in exts])

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def extract_one(args: Tuple[str, str, int, int, str, int]):
    vid_path, out_root, every_n, fps, pattern, jpeg_quality = args
    vid_path = Path(vid_path)
    rel_id = vid_path.stem  # 필요 시 디렉터리 구조 반영해 수정
    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        return {"video": str(vid_path), "ok": False, "frames": 0, "reason": "open_failed"}

    # fps 강제 샘플링: every_n == 0 이면 fps 기반, 아니면 N프레임 간격
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    step = every_n if every_n > 0 else max(int(round(native_fps / max(fps, 1))), 1)
    out_dir = Path(out_root) / rel_id
    ensure_dir(out_dir)

    i = saved = 0
    ok, img = cap.read()
    while ok:
        if i % step == 0:
            fname = pattern.format(video_id=rel_id, frame_idx=saved)
            out_path = out_dir / fname
            ensure_dir(out_path.parent)
            cv2.imwrite(str(out_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            saved += 1
        i += 1
        ok, img = cap.read()
    cap.release()
    return {"video": str(vid_path), "ok": True, "frames": saved, "native_fps": native_fps, "total": frame_count}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", type=Path, required=True, help="비디오 루트 디렉토리")
    ap.add_argument("--out-dir", type=Path, required=True, help="프레임 출력 루트")
    ap.add_argument("--manifest", type=Path, default=None, help="처리 결과를 기록할 JSON 경로")
    ap.add_argument("--num-workers", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--every-n", type=int, default=0, help="N프레임 간격 추출(>0). 0이면 --fps 사용")
    ap.add_argument("--fps", type=int, default=2, help="초당 추출 프레임 수(every-n이 0일 때만)")
    ap.add_argument("--pattern", type=str, default="{frame_idx:05d}.jpg",
                    help="저장 파일명 패턴. 사용가능 키: video_id, frame_idx")
    ap.add_argument("--jpeg-quality", type=int, default=95)
    args = ap.parse_args()

    vids = list_videos(args.video_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(str(v), str(args.out_dir), args.every_n, args.fps, args.pattern, args.jpeg_quality) for v in vids]

    with mp.Pool(processes=args.num_workers) as pool:
        results = list(pool.imap_unordered(extract_one, jobs))

    if args.manifest:
        with open(args.manifest, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    ok = sum(1 for r in results if r.get("ok"))
    print(f"[done] {ok}/{len(results)} videos extracted to {args.out_dir}")

if __name__ == "__main__":
    main()
