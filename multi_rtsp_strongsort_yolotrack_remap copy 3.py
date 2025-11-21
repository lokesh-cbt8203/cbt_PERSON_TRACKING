# # multi_rtsp_strongsort_reid_patched.py
# import cv2
# import torch
# import torch.nn.functional as F
# import numpy as np
# from pathlib import Path
# from ultralytics import YOLO
# from boxmot import StrongSort
# import threading
# import time
# import csv
# import os
# import math
# from collections import deque
# from torchvision import transforms

# # ---------------------------
# # CONFIG - edit as needed
# # ---------------------------
# RTSP_STREAMS = [
#     # "rtsp://localhost:8554/cam1",
#     # "rtsp://localhost:8554/cam2",

#     "rtmp://localhost/live/stream4",
#     # "rtsp://admin:rolex@123@192.168.1.111:554/Streaming/channels/101",

# ]

# REID_CHECKPOINT = Path("osnet_x0_25_msmt17.pt")  # place here if available
# USE_DEEP_REID = True

# SAVE_CSV = True
# CSV_PATH = "detections_log.csv"
# TRAJECTORY_CSV = "trajectories.csv"
# SAVE_PER_CAMERA = True
# SAVE_GRID = True
# SAVE_CROPS = True

# TILE_W, TILE_H = 800, 800
# GRID_COLS = 2
# GRID_FPS = 20.0

# # ---------------------------
# # Devices
# # ---------------------------
# if torch.cuda.is_available() and torch.cuda.device_count() > 0:
#     TRACKER_DEVICE = 0
#     YOLO_DEVICE = "cuda:0"
#     REID_DEVICE = "cuda:0"
# else:
#     TRACKER_DEVICE = "cpu"
#     YOLO_DEVICE = "cpu"
#     REID_DEVICE = "cpu"

# print("YOLO device:", YOLO_DEVICE, "Tracker device:", TRACKER_DEVICE, "ReID device:", REID_DEVICE)

# # ---------------------------
# # YOLO model
# # ---------------------------
# yolo_model = YOLO("yolo12m.pt")
# try:
#     yolo_model.to(YOLO_DEVICE)
# except Exception:
#     pass

# # ---------------------------
# # CSV writers + lock
# # ---------------------------
# csv_lock = threading.Lock()
# if SAVE_CSV:
#     csv_file = open(CSV_PATH, "w", newline="")
#     csv_writer = csv.writer(csv_file)
#     csv_writer.writerow(["timestamp", "stream_id", "persistent_id", "strongsort_id", "x1", "y1", "x2", "y2", "conf"])
#     traj_file = open(TRAJECTORY_CSV, "w", newline="")
#     traj_writer = csv.writer(traj_file)
#     traj_writer.writerow(["timestamp", "stream_id", "persistent_id", "x1", "y1", "x2", "y2", "conf"])

# # ---------------------------
# # Helper dirs
# # ---------------------------
# os.makedirs("StrongSort_saved_output/crops", exist_ok=True)
# os.makedirs("StrongSort_saved_output/saved_videos", exist_ok=True)

# # ---------------------------
# # Utils
# # ---------------------------
# def compute_hsv_hist(image, bbox):
#     x1, y1, x2, y2 = bbox
#     h, w = image.shape[:2]
#     x1 = max(0, min(w - 1, int(x1)))
#     x2 = max(0, min(w, int(x2)))
#     y1 = max(0, min(h - 1, int(y1)))
#     y2 = max(0, min(h, int(y2)))
#     if x2 <= x1 or y2 <= y1:
#         return None
#     crop = image[y1:y2, x1:x2]
#     if crop.size == 0:
#         return None
#     hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
#     hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
#     cv2.normalize(hist, hist)
#     return hist.flatten().astype(np.float32)

# def hist_distance(a, b):
#     if a is None or b is None:
#         return 1.0
#     try:
#         return cv2.compareHist(a.astype('float32'), b.astype('float32'), cv2.HISTCMP_BHATTACHARYYA)
#     except Exception:
#         return float(np.linalg.norm(a - b))

# def centroid(bbox):
#     x1, y1, x2, y2 = bbox
#     return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

# def euclidean(a, b):
#     return math.hypot(a[0] - b[0], a[1] - b[1])

# # ---------------------------
# # Robust ReID loader (tries multiple import paths & torch.jit)
# # ---------------------------
# class ReIDExtractor:
#     def __init__(self, checkpoint_path: Path = None, device="cpu"):
#         self.device = torch.device(device if device != "cpu" else "cpu")
#         self.model = None
#         self.transform = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Resize((256, 128)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406],
#                                  [0.229, 0.224, 0.225])
#         ])
#         self.loaded_backend = None

#         # Try BoxMOT-based loader paths for different boxmot versions
#         if checkpoint_path is not None and checkpoint_path.exists():
#             tried = False
#             # Try newer path
#             try:
#                 from boxmot.trackers.strongsort.appearance.reid.reid_model import ReidModel  # type: ignore
#                 try:
#                     # instantiate via ReidModel if signature matches. Support depends on version.
#                     self.model = ReidModel(model_path=str(checkpoint_path), device=self.device)
#                     self.loaded_backend = "boxmot.reid_model"
#                     tried = True
#                     print("[ReID] Loaded via boxmot.trackers...reid_model")
#                 except Exception:
#                     self.model = None
#             except Exception:
#                 pass

#             # Try alternative older import
#             if not tried:
#                 try:
#                     from boxmot.trackers.strongsort.appearance.reid import ReidModel  # type: ignore
#                     try:
#                         self.model = ReidModel(model_path=str(checkpoint_path), device=self.device)
#                         self.loaded_backend = "boxmot.trackers.strongsort.appearance.reid"
#                         tried = True
#                         print("[ReID] Loaded via boxmot.trackers.strongsort.appearance.reid")
#                     except Exception:
#                         self.model = None
#                 except Exception:
#                     pass

#             # Try torch.jit scripted model
#             if not tried:
#                 try:
#                     m = torch.jit.load(str(checkpoint_path), map_location=self.device)
#                     m.to(self.device)
#                     m.eval()
#                     self.model = m
#                     self.loaded_backend = "torchscript"
#                     tried = True
#                     print("[ReID] Loaded TorchScript ReID model.")
#                 except Exception:
#                     self.model = None

#         if self.model is None:
#             print("[ReID] No usable deep ReID model found; falling back to HSV hist features.")
#             self.model = None

#     def extract(self, frame_bgr, bbox):
#         if self.model is None:
#             return None
#         x1, y1, x2, y2 = bbox
#         h, w = frame_bgr.shape[:2]
#         x1 = max(0, min(w - 1, int(x1)))
#         x2 = max(0, min(w, int(x2)))
#         y1 = max(0, min(h - 1, int(y1)))
#         y2 = max(0, min(h, int(y2)))
#         if x2 <= x1 or y2 <= y1:
#             return None
#         crop = frame_bgr[y1:y2, x1:x2]
#         if crop.size == 0:
#             return None
#         try:
#             if self.loaded_backend and "boxmot" in str(self.loaded_backend):
#                 # Many boxmot backends accept numpy images via a .feature() or .get_feature() method.
#                 # Try common method names.
#                 if hasattr(self.model, "feature"):
#                     feat = self.model.feature(crop)
#                 elif hasattr(self.model, "features"):
#                     feat = self.model.features(crop)
#                 elif hasattr(self.model, "get_feature"):
#                     feat = self.model.get_feature(crop)
#                 else:
#                     # fallback: try calling model directly with preprocessed tensor
#                     inp = self.transform(crop).unsqueeze(0).to(self.device)
#                     with torch.no_grad():
#                         feat = self.model(inp)
#                 if isinstance(feat, torch.Tensor):
#                     feat = feat.detach().cpu().numpy().reshape(-1)
#                 feat = np.asarray(feat, dtype=np.float32)
#             else:
#                 inp = self.transform(crop).unsqueeze(0).to(self.device)
#                 with torch.no_grad():
#                     out = self.model(inp)
#                 if isinstance(out, torch.Tensor):
#                     feat = out.cpu().numpy().reshape(-1)
#                 elif isinstance(out, dict):
#                     for k in ("feat", "features", "feature"):
#                         if k in out and isinstance(out[k], torch.Tensor):
#                             feat = out[k].cpu().numpy().reshape(-1)
#                             break
#                     else:
#                         return None
#                 else:
#                     return None
#                 feat = np.asarray(feat, dtype=np.float32)

#             if feat.size == 0 or not np.isfinite(feat).all():
#                 return None
#             # normalize
#             n = np.linalg.norm(feat)
#             if n == 0 or not np.isfinite(n):
#                 return None
#             return (feat / n).astype(np.float32)
#         except Exception:
#             return None

# # ---------------------------
# # TrackManager (persistent id remapping)
# # ---------------------------
# class TrackManager:
#     def __init__(self, max_age_frames=150, feat_match_thresh=0.45, hist_match_thresh=0.45, max_centroid_dist=160):
#         self.strong2persist = {}
#         self.persist_data = {}
#         self.next_persistent_id = 1
#         self.max_age_frames = max_age_frames
#         self.feat_match_thresh = feat_match_thresh
#         self.hist_match_thresh = hist_match_thresh
#         self.max_centroid_dist = max_centroid_dist

#     def _create_persistent(self, strong_id, bbox, feat, hist, frame_idx):
#         pid = self.next_persistent_id
#         self.next_persistent_id += 1
#         self.strong2persist[strong_id] = pid
#         self.persist_data[pid] = {
#             "last_bbox": bbox,
#             "last_centroid": centroid(bbox),
#             "last_frame": frame_idx,
#             "feats": deque([feat], maxlen=16) if feat is not None else deque(maxlen=16),
#             "hists": deque([hist], maxlen=8) if hist is not None else deque(maxlen=8),
#             "active": True
#         }
#         return pid

#     def _update_persistent(self, pid, strong_id, bbox, feat, hist, frame_idx):
#         self.strong2persist[strong_id] = pid
#         data = self.persist_data.get(pid, None)
#         if data is None:
#             self.persist_data[pid] = {
#                 "last_bbox": bbox,
#                 "last_centroid": centroid(bbox),
#                 "last_frame": frame_idx,
#                 "feats": deque([feat], maxlen=16) if feat is not None else deque(maxlen=16),
#                 "hists": deque([hist], maxlen=8) if hist is not None else deque(maxlen=8),
#                 "active": True
#             }
#             return
#         data["last_bbox"] = bbox
#         data["last_centroid"] = centroid(bbox)
#         data["last_frame"] = frame_idx
#         if feat is not None:
#             data["feats"].append(feat)
#         if hist is not None:
#             data["hists"].append(hist)
#         data["active"] = True

#     def cleanup_old(self, current_frame_idx):
#         for pid, d in self.persist_data.items():
#             if current_frame_idx - d["last_frame"] > self.max_age_frames:
#                 d["active"] = False

#     def _cosine_dist(self, a, b):
#         if a is None or b is None:
#             return 1.0
#         try:
#             denom = (np.linalg.norm(a) * np.linalg.norm(b))
#             if denom == 0:
#                 return 1.0
#             sim = float(np.dot(a, b) / denom)
#             return 1.0 - sim
#         except Exception:
#             return 1.0

#     def match_to_existing(self, bbox, feat, hist, frame_idx):
#         best_pid = None
#         best_score = 10.0
#         c = centroid(bbox)
#         for pid, d in self.persist_data.items():
#             age = frame_idx - d["last_frame"]
#             if age > self.max_age_frames:
#                 continue
#             dist_cent = euclidean(c, d["last_centroid"])
#             if dist_cent > self.max_centroid_dist:
#                 continue
#             if feat is not None and len(d["feats"]) > 0:
#                 feat_dist = min(self._cosine_dist(feat, f) for f in d["feats"])
#             else:
#                 feat_dist = 1.0
#             if hist is not None and len(d["hists"]) > 0:
#                 hist_dist = min(hist_distance(hist, h) for h in d["hists"])
#             else:
#                 hist_dist = 1.0

#             if feat_dist < self.feat_match_thresh:
#                 score = feat_dist + (dist_cent / (self.max_centroid_dist * 4.0))
#             elif hist_dist < self.hist_match_thresh:
#                 score = hist_dist + (dist_cent / (self.max_centroid_dist * 3.0))
#             else:
#                 score = (dist_cent / (self.max_centroid_dist * 1.5)) + 0.5

#             if score < best_score:
#                 best_score = score
#                 best_pid = pid
#         return best_pid, best_score

#     def register_track(self, strong_id, bbox, feat, hist, frame_idx):
#         remapped = False
#         remapped_from = None
#         if strong_id in self.strong2persist:
#             pid = self.strong2persist[strong_id]
#             self._update_persistent(pid, strong_id, bbox, feat, hist, frame_idx)
#             return pid, remapped, remapped_from

#         matched_pid, score = self.match_to_existing(bbox, feat, hist, frame_idx)
#         if matched_pid is not None:
#             self._update_persistent(matched_pid, strong_id, bbox, feat, hist, frame_idx)
#             remapped = True
#             remapped_from = matched_pid
#             return matched_pid, remapped, remapped_from

#         pid = self._create_persistent(strong_id, bbox, feat, hist, frame_idx)
#         return pid, remapped, remapped_from

#     def strong_id_removed(self, strong_id):
#         if strong_id in self.strong2persist:
#             pid = self.strong2persist.pop(strong_id, None)
#             return pid
#         return None

# # ---------------------------
# # StreamWorker
# # ---------------------------
# class StreamWorker:
#     def __init__(self, stream_url, stream_id, reid_extractor=None):
#         self.stream_url = stream_url
#         self.stream_id = stream_id
#         self.latest_frame = None
#         self.output_frame = None
#         self.stopped = False
#         self.frame_idx = 0
#         self.reid_extractor = reid_extractor

#         self.cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
#         try:
#             self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 4)
#         except Exception:
#             pass
#         if not self.cap.isOpened():
#             print(f"[Stream {self.stream_id}] ERROR: Cannot open stream!")

#         self.tracker = StrongSort(
#             reid_weights=Path("osnet_x1_0_msmt17.pt"),
#             device=TRACKER_DEVICE,
#             half=(TRACKER_DEVICE != "cpu"),
#             max_dist=0.45,
#             max_iou_dist=0.75,
#             match_thresh=0.6,
#             nn_budget=200,
#             n_init=3,
#             max_age=120,
#             min_hits=2,
#             mc_lambda=0.99,
#             ema_alpha=0.9
#         )

#         self.tm = TrackManager(max_age_frames=150, feat_match_thresh=0.45, hist_match_thresh=0.45, max_centroid_dist=160)

#         self.writer = None
#         if SAVE_PER_CAMERA:
#             fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#             out_path = f"StrongSort_saved_output/saved_videos/stream_{self.stream_id}_output.mp4"
#             self.writer = cv2.VideoWriter(out_path, fourcc, GRID_FPS, (TILE_W, TILE_H))

#         self.remap_visuals = {}

#         self.grab_thread = threading.Thread(target=self.grab_frames, daemon=True)
#         self.grab_thread.start()
#         self.process_thread = threading.Thread(target=self.process_frames, daemon=True)
#         self.process_thread.start()

#     def grab_frames(self):
#         while not self.stopped:
#             ret, frame = self.cap.read()
#             if not ret or frame is None:
#                 print(f"[Stream {self.stream_id}] WARNING: Unable to read frame")
#                 time.sleep(1)
#                 continue
#             # sanitize frame numeric anomalies
#             if np.isnan(frame).any() or np.isinf(frame).any():
#                 frame = np.nan_to_num(frame, nan=0, posinf=255, neginf=0).astype(np.uint8)
#             self.latest_frame = frame

#     def process_frames(self):
#         while not self.stopped:
#             if self.latest_frame is None:
#                 time.sleep(0.01)
#                 continue

#             frame = self.latest_frame.copy()
#             self.frame_idx += 1
#             frame_count = self.frame_idx

#             # YOLO detection
#             try:
#                 results = yolo_model(frame, conf=0.22, iou=0.25, verbose=False)
#             except Exception as e:
#                 print(f"[Stream {self.stream_id}] YOLO inference error:", e)
#                 time.sleep(0.01)
#                 continue

#             if results is None or len(results) == 0 or len(results[0].boxes) == 0:
#                 det = np.empty((0, 6), dtype=np.float32)
#             else:
#                 boxes = results[0].boxes
#                 xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
#                 conf = boxes.conf.cpu().numpy().astype(np.float32)
#                 cls = boxes.cls.cpu().numpy().astype(np.float32)
#                 idx = cls == 0
#                 xyxy, conf, cls = xyxy[idx], conf[idx], cls[idx]
#                 if len(xyxy) > 0:
#                     det = np.concatenate([xyxy, conf[:, None], cls[:, None]], axis=1)
#                 else:
#                     det = np.empty((0, 6), dtype=np.float32)

#             # ----- SANITIZE detections -----
#             if det is not None and len(det) > 0:
#                 # drop NaN/Inf rows
#                 det = det[np.isfinite(det).all(axis=1)]
#                 # require x2>x1 and y2>y1 (positive width/height)
#                 mask = (det[:, 2] > det[:, 0]) & (det[:, 3] > det[:, 1])
#                 det = det[mask]
#                 if det.size == 0:
#                     det = np.empty((0, 6), dtype=np.float32)

#             # debug print det for first few frames if invalid
#             if det is not None and len(det) > 0:
#                 if np.isnan(det).any() or np.isinf(det).any():
#                     print(f"[Stream {self.stream_id}] DEBUG: det contains invalid numeric entries:\n{det}")

#             # tracker update (protected)
#             tracked = []
#             try:
#                 if det is None or len(det) == 0:
#                     # StrongSort expects an array; pass empty
#                     tracked = self.tracker.update(np.empty((0,6), dtype=np.float32), frame)
#                 else:
#                     # debug print - FIRST line shows det array (helpful to find bad inputs)
#                     if frame_count % 100 == 1:
#                         print(f"[Stream {self.stream_id}] DEBUG frame {frame_count} det_array (first 8 rows):\n{det[:8]}")
#                     tracked = self.tracker.update(det, frame)
#             except Exception as e:
#                 # Dont reset internals; just log and continue
#                 print(f"[Stream {self.stream_id}] Tracker update error: {e}")
#                 # try to detect bad embeddings used by tracker
#                 try:
#                     if hasattr(self.tracker, "metric") and hasattr(self.tracker.metric, "samples"):
#                         emb = self.tracker.metric.samples
#                         if isinstance(emb, np.ndarray) and (np.isnan(emb).any() or np.isinf(emb).any()):
#                             print(f"[Stream {self.stream_id}] Tracker internal embeddings contain NaN/Inf - clearing if possible")
#                             try:
#                                 if isinstance(self.tracker.metric.samples, dict):
#                                     self.tracker.metric.samples.clear()
#                             except Exception:
#                                 pass
#                 except Exception:
#                     pass
#                 time.sleep(0.01)
#                 continue

#             # cleanup persistent entries
#             self.tm.cleanup_old(frame_count)

#             out_frame = frame.copy()
#             timestamp = time.time()
#             seen_strong_ids = set()

#             for t in tracked:
#                 # typical layout: x1,y1,x2,y2, strong_id, conf, cls, ind
#                 try:
#                     x1, y1, x2, y2, strong_id, conf_val, cls_val, ind = t
#                 except Exception:
#                     x1, y1, x2, y2 = t[0], t[1], t[2], t[3]
#                     strong_id = int(t[4]) if len(t) > 4 else -1
#                     conf_val = float(t[5]) if len(t) > 5 else 0.0

#                 # skip invalid numbers
#                 if not np.isfinite([x1, y1, x2, y2]).all():
#                     continue

#                 x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])
#                 seen_strong_ids.add(int(strong_id))

#                 # Deep ReID feature if available, else hist
#                 feat = None
#                 if USE_DEEP_REID and self.reid_extractor is not None:
#                     feat = self.reid_extractor.extract(frame, (x1_i, y1_i, x2_i, y2_i))
#                     if feat is not None and (np.isnan(feat).any() or np.isinf(feat).any()):
#                         # bad feature -> drop to None
#                         feat = None

#                 hist = None
#                 if feat is None:
#                     hist = compute_hsv_hist(frame, (x1_i, y1_i, x2_i, y2_i))

#                 pid, remapped, remapped_from = self.tm.register_track(int(strong_id), (x1, y1, x2, y2), feat, hist, frame_count)

#                 color = (0, 255, 0)
#                 if remapped:
#                     color = (0, 165, 255)
#                     prev_cent = self.tm.persist_data.get(pid, {}).get("last_centroid", None)
#                     cur_cent = centroid((x1, y1, x2, y2))
#                     if prev_cent is not None:
#                         p1 = (int(prev_cent[0]), int(prev_cent[1]))
#                         p2 = (int(cur_cent[0]), int(cur_cent[1]))
#                         h_f, w_f = out_frame.shape[:2]
#                         p1 = (max(0, min(w_f-1, p1[0])), max(0, min(h_f-1, p1[1])))
#                         p2 = (max(0, min(w_f-1, p2[0])), max(0, min(h_f-1, p2[1])))
#                         cv2.arrowedLine(out_frame, p1, p2, color, 2, tipLength=0.2)

#                 label = f" S{int(strong_id)} {conf_val:.2f}"#P{int(pid)}
#                 cv2.rectangle(out_frame, (x1_i, y1_i), (x2_i, y2_i), color, 2)
#                 cv2.putText(out_frame, label, (x1_i, max(0, y1_i-8)), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)

#                 if SAVE_CSV:
#                     with csv_lock:
#                         csv_writer.writerow([timestamp, self.stream_id, int(pid), int(strong_id), float(x1), float(y1), float(x2), float(y2), float(conf_val)])
#                         traj_writer.writerow([timestamp, self.stream_id, int(pid), float(x1), float(y1), float(x2), float(y2), float(conf_val)])

#                 if SAVE_CROPS:
#                     try:
#                         crop_dir = Path(f"StrongSort_saved_output/crops/stream{self.stream_id}/persist_{int(pid)}")
#                         crop_dir.mkdir(parents=True, exist_ok=True)
#                         crop_img = frame[y1_i:y2_i, x1_i:x2_i]
#                         if crop_img.size != 0:
#                             crop_name = crop_dir / f"{int(timestamp*1000)}.jpg"
#                             cv2.imwrite(str(crop_name), crop_img)
#                     except Exception as e:
#                         print(f"[Stream {self.stream_id}] Crop save error:", e)

#             # drop stale strong->persist mappings not updated recently to allow remap
#             mapped_strong = list(self.tm.strong2persist.keys())
#             for s in mapped_strong:
#                 pid = self.tm.strong2persist.get(s)
#                 if pid is None:
#                     continue
#                 last_frame = self.tm.persist_data.get(pid, {}).get("last_frame", 0)
#                 if frame_count - last_frame > 6:
#                     self.tm.strong2persist.pop(s, None)

#             try:
#                 resized = cv2.resize(out_frame, (TILE_W, TILE_H))
#             except Exception:
#                 resized = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
#             self.output_frame = resized

#             if self.writer is not None:
#                 try:
#                     self.writer.write(resized)
#                 except Exception as e:
#                     print(f"[Stream {self.stream_id}] Video write error:", e)

#             time.sleep(0.001)

#     def stop(self):
#         self.stopped = True
#         try:
#             self.grab_thread.join(timeout=1.0)
#         except Exception:
#             pass
#         try:
#             self.process_thread.join(timeout=1.0)
#         except Exception:
#             pass
#         try:
#             self.cap.release()
#         except Exception:
#             pass
#         if self.writer is not None:
#             try:
#                 self.writer.release()
#             except Exception:
#                 pass

# # ---------------------------
# # init reid extractor (best-effort)
# # ---------------------------
# reid_extractor = None
# if USE_DEEP_REID:
#     try:
#         reid_extractor = ReIDExtractor(REID_CHECKPOINT if REID_CHECKPOINT.exists() else None, device=REID_DEVICE)
#         if reid_extractor.model is None:
#             reid_extractor = None
#     except Exception as e:
#         print("[Main] ReID init failed:", e)
#         reid_extractor = None

# # ---------------------------
# # Launch streams
# # ---------------------------
# workers = []
# for i, url in enumerate(RTSP_STREAMS):
#     w = StreamWorker(url, i, reid_extractor)
#     workers.append(w)

# time.sleep(1.0)

# # ---------------------------
# # Grid writer init
# # ---------------------------
# grid_writer = None
# if SAVE_GRID:
#     rows = (len(workers) + GRID_COLS - 1) // GRID_COLS
#     grid_w = TILE_W * GRID_COLS
#     grid_h = TILE_H * rows
#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     grid_writer = cv2.VideoWriter("StrongSort_saved_output/saved_videos/combined_grid.mp4", fourcc, GRID_FPS, (grid_w, grid_h))

# def make_grid(frames, cols=GRID_COLS, tile_w=TILE_W, tile_h=TILE_H):
#     if len(frames) == 0:
#         return None
#     rows = (len(frames) + cols - 1) // cols
#     grid = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
#     for idx, frame in enumerate(frames):
#         r = idx // cols
#         c = idx % cols
#         grid[r*tile_h:(r+1)*tile_h, c*tile_w:(c+1)*tile_w] = frame
#     return grid

# # ---------------------------
# # Main display + save loop
# # ---------------------------
# try:
#     while True:
#         tiles = []
#         for w in workers:
#             if hasattr(w, "output_frame") and w.output_frame is not None:
#                 tiles.append(w.output_frame)
#             else:
#                 tiles.append(np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8))

#         grid = make_grid(tiles)
#         if grid is not None:
#             cv2.imshow("Multi-Stream Grid", grid)
#             if SAVE_GRID and grid_writer is not None:
#                 try:
#                     grid_writer.write(grid)
#                 except Exception as e:
#                     print("Grid write error:", e)

#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

# except KeyboardInterrupt:
#     print("Stopping...")

# # ---------------------------
# # Cleanup
# # ---------------------------
# for w in workers:
#     w.stop()

# if grid_writer is not None:
#     grid_writer.release()

# cv2.destroyAllWindows()
# if SAVE_CSV:
#     csv_file.close()
#     traj_file.close()

# print("All outputs saved to folder: saved_videos/, crops/, and", CSV_PATH)


# multi_rtsp_strongsort_reid_async.py
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from boxmot import StrongSort
import threading
import time
import csv
import os
import math
from collections import deque
from torchvision import transforms
import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor

# ---------------------------
# CONFIG - edit as needed
# ---------------------------
RTSP_STREAMS = [
    # "rtsp://localhost:8554/cam1",
    # "rtsp://localhost:8554/cam2",

    # "rtmp://localhost/live/stream4",
    # "rtsp://admin:rolex@123@192.168.1.111:554/Streaming/channels/101",
    "rtsp://admin:rolex@123@192.168.1.110:554/Streaming/channels/101",

]

REID_CHECKPOINT = Path("osnet_x0_25_msmt17.pt")  # place here if available
USE_DEEP_REID = True

SAVE_CSV = True
CSV_PATH = "detections_log.csv"
TRAJECTORY_CSV = "trajectories.csv"
SAVE_PER_CAMERA = True
SAVE_GRID = True
SAVE_CROPS = True

TILE_W, TILE_H = 800, 800
GRID_COLS = 2
GRID_FPS = 20.0

# ---------------------------
# Devices
# ---------------------------
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    TRACKER_DEVICE = 0
    YOLO_DEVICE = "cuda:0"
    REID_DEVICE = "cuda:0"
else:
    TRACKER_DEVICE = "cpu"
    YOLO_DEVICE = "cpu"
    REID_DEVICE = "cpu"

print("YOLO device:", YOLO_DEVICE, "Tracker device:", TRACKER_DEVICE, "ReID device:", REID_DEVICE)

# ---------------------------
# YOLO model (global)
# ---------------------------
# Keep model load outside of per-worker threads to reuse GPU memory
try:
    yolo_model = YOLO("yolo12m.pt")
    try:
        yolo_model.to(YOLO_DEVICE)
    except Exception:
        pass
except Exception as e:
    print("YOLO model load error:", e)
    yolo_model = None

# ---------------------------
# CSV writers + lock
# ---------------------------
csv_lock = threading.Lock()
if SAVE_CSV:
    csv_file = open(CSV_PATH, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["timestamp", "stream_id", "persistent_id", "strongsort_id", "x1", "y1", "x2", "y2", "conf"])
    traj_file = open(TRAJECTORY_CSV, "w", newline="")
    traj_writer = csv.writer(traj_file)
    traj_writer.writerow(["timestamp", "stream_id", "persistent_id", "x1", "y1", "x2", "y2", "conf"])

# ---------------------------
# Helper dirs
# ---------------------------
os.makedirs("StrongSort_saved_output/crops", exist_ok=True)
os.makedirs("StrongSort_saved_output/saved_videos", exist_ok=True)

# ---------------------------
# Utils
# ---------------------------
def compute_hsv_hist(image, bbox):
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    x1 = max(0, min(w - 1, int(x1)))
    x2 = max(0, min(w, int(x2)))
    y1 = max(0, min(h - 1, int(y1)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten().astype(np.float32)

def hist_distance(a, b):
    if a is None or b is None:
        return 1.0
    try:
        return cv2.compareHist(a.astype('float32'), b.astype('float32'), cv2.HISTCMP_BHATTACHARYYA)
    except Exception:
        return float(np.linalg.norm(a - b))

def centroid(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

# ---------------------------
# Robust ReID loader (tries multiple import paths & torch.jit)
# ---------------------------
class ReIDExtractor:
    def __init__(self, checkpoint_path: Path = None, device="cpu"):
        self.device = torch.device(device if device != "cpu" else "cpu")
        self.model = None
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.loaded_backend = None

        if checkpoint_path is not None and checkpoint_path.exists():
            tried = False
            try:
                from boxmot.trackers.strongsort.appearance.reid.reid_model import ReidModel  # type: ignore
                try:
                    self.model = ReidModel(model_path=str(checkpoint_path), device=self.device)
                    self.loaded_backend = "boxmot.reid_model"
                    tried = True
                    print("[ReID] Loaded via boxmot.trackers...reid_model")
                except Exception:
                    self.model = None
            except Exception:
                pass

            if not tried:
                try:
                    from boxmot.trackers.strongsort.appearance.reid import ReidModel  # type: ignore
                    try:
                        self.model = ReidModel(model_path=str(checkpoint_path), device=self.device)
                        self.loaded_backend = "boxmot.trackers.strongsort.appearance.reid"
                        tried = True
                        print("[ReID] Loaded via boxmot.trackers.strongsort.appearance.reid")
                    except Exception:
                        self.model = None
                except Exception:
                    pass

            if not tried:
                try:
                    m = torch.jit.load(str(checkpoint_path), map_location=self.device)
                    m.to(self.device)
                    m.eval()
                    self.model = m
                    self.loaded_backend = "torchscript"
                    tried = True
                    print("[ReID] Loaded TorchScript ReID model.")
                except Exception:
                    self.model = None

        if self.model is None:
            print("[ReID] No usable deep ReID model found; falling back to HSV hist features.")
            self.model = None

    def extract(self, frame_bgr, bbox):
        if self.model is None:
            return None
        x1, y1, x2, y2 = bbox
        h, w = frame_bgr.shape[:2]
        x1 = max(0, min(w - 1, int(x1)))
        x2 = max(0, min(w, int(x2)))
        y1 = max(0, min(h - 1, int(y1)))
        y2 = max(0, min(h, int(y2)))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        try:
            if self.loaded_backend and "boxmot" in str(self.loaded_backend):
                if hasattr(self.model, "feature"):
                    feat = self.model.feature(crop)
                elif hasattr(self.model, "features"):
                    feat = self.model.features(crop)
                elif hasattr(self.model, "get_feature"):
                    feat = self.model.get_feature(crop)
                else:
                    inp = self.transform(crop).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        feat = self.model(inp)
                if isinstance(feat, torch.Tensor):
                    feat = feat.detach().cpu().numpy().reshape(-1)
                feat = np.asarray(feat, dtype=np.float32)
            else:
                inp = self.transform(crop).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    out = self.model(inp)
                if isinstance(out, torch.Tensor):
                    feat = out.cpu().numpy().reshape(-1)
                elif isinstance(out, dict):
                    for k in ("feat", "features", "feature"):
                        if k in out and isinstance(out[k], torch.Tensor):
                            feat = out[k].cpu().numpy().reshape(-1)
                            break
                    else:
                        return None
                else:
                    return None
                feat = np.asarray(feat, dtype=np.float32)

            if feat.size == 0 or not np.isfinite(feat).all():
                return None
            n = np.linalg.norm(feat)
            if n == 0 or not np.isfinite(n):
                return None
            return (feat / n).astype(np.float32)
        except Exception:
            return None

# ---------------------------
# TrackManager (persistent id remapping)
# ---------------------------
class TrackManager:
    def __init__(self, max_age_frames=150, feat_match_thresh=0.45, hist_match_thresh=0.45, max_centroid_dist=160):
        self.strong2persist = {}
        self.persist_data = {}
        self.next_persistent_id = 1
        self.max_age_frames = max_age_frames
        self.feat_match_thresh = feat_match_thresh
        self.hist_match_thresh = hist_match_thresh
        self.max_centroid_dist = max_centroid_dist

    def _create_persistent(self, strong_id, bbox, feat, hist, frame_idx):
        pid = self.next_persistent_id
        self.next_persistent_id += 1
        self.strong2persist[strong_id] = pid
        self.persist_data[pid] = {
            "last_bbox": bbox,
            "last_centroid": centroid(bbox),
            "last_frame": frame_idx,
            "feats": deque([feat], maxlen=16) if feat is not None else deque(maxlen=16),
            "hists": deque([hist], maxlen=8) if hist is not None else deque(maxlen=8),
            "active": True
        }
        return pid

    def _update_persistent(self, pid, strong_id, bbox, feat, hist, frame_idx):
        self.strong2persist[strong_id] = pid
        data = self.persist_data.get(pid, None)
        if data is None:
            self.persist_data[pid] = {
                "last_bbox": bbox,
                "last_centroid": centroid(bbox),
                "last_frame": frame_idx,
                "feats": deque([feat], maxlen=16) if feat is not None else deque(maxlen=16),
                "hists": deque([hist], maxlen=8) if hist is not None else deque(maxlen=8),
                "active": True
            }
            return
        data["last_bbox"] = bbox
        data["last_centroid"] = centroid(bbox)
        data["last_frame"] = frame_idx
        if feat is not None:
            data["feats"].append(feat)
        if hist is not None:
            data["hists"].append(hist)
        data["active"] = True

    def cleanup_old(self, current_frame_idx):
        for pid, d in self.persist_data.items():
            if current_frame_idx - d["last_frame"] > self.max_age_frames:
                d["active"] = False

    def _cosine_dist(self, a, b):
        if a is None or b is None:
            return 1.0
        try:
            denom = (np.linalg.norm(a) * np.linalg.norm(b))
            if denom == 0:
                return 1.0
            sim = float(np.dot(a, b) / denom)
            return 1.0 - sim
        except Exception:
            return 1.0

    def match_to_existing(self, bbox, feat, hist, frame_idx):
        best_pid = None
        best_score = 10.0
        c = centroid(bbox)
        for pid, d in self.persist_data.items():
            age = frame_idx - d["last_frame"]
            if age > self.max_age_frames:
                continue
            dist_cent = euclidean(c, d["last_centroid"])
            if dist_cent > self.max_centroid_dist:
                continue
            if feat is not None and len(d["feats"]) > 0:
                feat_dist = min(self._cosine_dist(feat, f) for f in d["feats"])
            else:
                feat_dist = 1.0
            if hist is not None and len(d["hists"]) > 0:
                hist_dist = min(hist_distance(hist, h) for h in d["hists"])
            else:
                hist_dist = 1.0

            if feat_dist < self.feat_match_thresh:
                score = feat_dist + (dist_cent / (self.max_centroid_dist * 4.0))
            elif hist_dist < self.hist_match_thresh:
                score = hist_dist + (dist_cent / (self.max_centroid_dist * 3.0))
            else:
                score = (dist_cent / (self.max_centroid_dist * 1.5)) + 0.5

            if score < best_score:
                best_score = score
                best_pid = pid
        return best_pid, best_score

    def register_track(self, strong_id, bbox, feat, hist, frame_idx):
        remapped = False
        remapped_from = None
        if strong_id in self.strong2persist:
            pid = self.strong2persist[strong_id]
            self._update_persistent(pid, strong_id, bbox, feat, hist, frame_idx)
            return pid, remapped, remapped_from

        matched_pid, score = self.match_to_existing(bbox, feat, hist, frame_idx)
        if matched_pid is not None:
            self._update_persistent(matched_pid, strong_id, bbox, feat, hist, frame_idx)
            remapped = True
            remapped_from = matched_pid
            return matched_pid, remapped, remapped_from

        pid = self._create_persistent(strong_id, bbox, feat, hist, frame_idx)
        return pid, remapped, remapped_from

    def strong_id_removed(self, strong_id):
        if strong_id in self.strong2persist:
            pid = self.strong2persist.pop(strong_id, None)
            return pid
        return None

# ---------------------------
# StreamWorker (threaded grab + async processing)
# ---------------------------
class StreamWorker:
    def __init__(self, stream_url, stream_id, reid_extractor=None, executor=None):
        self.stream_url = stream_url
        self.stream_id = stream_id
        self.latest_frame = None
        self.output_frame = None
        self.stopped = False
        self.frame_idx = 0
        self.reid_extractor = reid_extractor
        self.executor = executor or ThreadPoolExecutor(max_workers=2)

        # VideoCapture (blocking) in a thread
        self.cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 4)
        except Exception:
            pass
        if not self.cap.isOpened():
            print(f"[Stream {self.stream_id}] ERROR: Cannot open stream!")

        # instantiate a StrongSort tracker instance per stream (keeps its own state)
        try:
            self.tracker = StrongSort(
                reid_weights=Path("osnet_x1_0_msmt17.pt"),
                device=TRACKER_DEVICE,
                half=(TRACKER_DEVICE != "cpu"),
                max_dist=0.45,
                max_iou_dist=0.75,
                match_thresh=0.6,
                nn_budget=200,
                n_init=3,
                max_age=120,
                min_hits=2,
                mc_lambda=0.99,
                ema_alpha=0.9
            )
        except Exception as e:
            print(f"[Stream {self.stream_id}] StrongSort init error: {e}")
            self.tracker = None

        self.tm = TrackManager(max_age_frames=150, feat_match_thresh=0.45, hist_match_thresh=0.45, max_centroid_dist=160)

        self.writer = None
        if SAVE_PER_CAMERA:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_path = f"StrongSort_saved_output/saved_videos/stream_{self.stream_id}_output.mp4"
            self.writer = cv2.VideoWriter(out_path, fourcc, GRID_FPS, (TILE_W, TILE_H))

        self.remap_visuals = {}

        # start grab thread (keeps reading frames)
        self.grab_thread = threading.Thread(target=self.grab_frames, daemon=True)
        self.grab_thread.start()

    def grab_frames(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                # print once per second to avoid spam
                print(f"[Stream {self.stream_id}] WARNING: Unable to read frame")
                time.sleep(1)
                continue
            # sanitize frame numeric anomalies
            if np.isnan(frame).any() or np.isinf(frame).any():
                frame = np.nan_to_num(frame, nan=0, posinf=255, neginf=0).astype(np.uint8)
            self.latest_frame = frame

    async def process_loop(self, stop_event: asyncio.Event):
        """
        Async process loop:
        - Waits for latest_frame
        - Runs YOLO and tracker in background threads using asyncio.to_thread
        - Writes outputs & composes output_frame
        """
        while not stop_event.is_set() and not self.stopped:
            if self.latest_frame is None:
                # short sleep to yield control
                await asyncio.sleep(0.01)
                continue

            frame = self.latest_frame.copy()
            self.frame_idx += 1
            frame_count = self.frame_idx

            # YOLO detection (blocking) -> run in thread to avoid blocking loop
            try:
                if yolo_model is None:
                    results = None
                else:
                    # call synchronous inference inside thread pool
                    results = await asyncio.to_thread(yolo_model, frame, conf=0.22, iou=0.25, verbose=False)
            except Exception as e:
                print(f"[Stream {self.stream_id}] YOLO inference error:", e)
                await asyncio.sleep(0.01)
                continue

            # Parse detections
            if results is None or len(results) == 0 or len(results[0].boxes) == 0:
                det = np.empty((0, 6), dtype=np.float32)
            else:
                boxes = results[0].boxes
                xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
                conf = boxes.conf.cpu().numpy().astype(np.float32)
                cls = boxes.cls.cpu().numpy().astype(np.float32)
                idx = cls == 0
                xyxy, conf, cls = xyxy[idx], conf[idx], cls[idx]
                if len(xyxy) > 0:
                    det = np.concatenate([xyxy, conf[:, None], cls[:, None]], axis=1)
                else:
                    det = np.empty((0, 6), dtype=np.float32)

            # sanitize detections
            if det is not None and len(det) > 0:
                det = det[np.isfinite(det).all(axis=1)]
                mask = (det[:, 2] > det[:, 0]) & (det[:, 3] > det[:, 1])
                det = det[mask]
                if det.size == 0:
                    det = np.empty((0, 6), dtype=np.float32)

            # tracker update (blocking) -> run in thread
            tracked = []
            try:
                # call tracker.update in thread to avoid blocking asyncio
                if self.tracker is None:
                    tracked = []
                else:
                    tracked = await asyncio.to_thread(self.tracker.update, (det if det is not None else np.empty((0,6), dtype=np.float32)), frame)
            except Exception as e:
                print(f"[Stream {self.stream_id}] Tracker update error:", e)
                await asyncio.sleep(0.01)
                continue

            # cleanup persistent entries
            self.tm.cleanup_old(frame_count)

            out_frame = frame.copy()
            timestamp = time.time()
            seen_strong_ids = set()

            for t in tracked:
                try:
                    x1, y1, x2, y2, strong_id, conf_val, cls_val, ind = t
                except Exception:
                    x1, y1, x2, y2 = t[0], t[1], t[2], t[3]
                    strong_id = int(t[4]) if len(t) > 4 else -1
                    conf_val = float(t[5]) if len(t) > 5 else 0.0

                if not np.isfinite([x1, y1, x2, y2]).all():
                    continue

                x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])
                seen_strong_ids.add(int(strong_id))

                # Deep ReID feature if available, else hist
                feat = None
                if USE_DEEP_REID and self.reid_extractor is not None:
                    # run feature extraction in thread (since some reid backends may block)
                    try:
                        feat = await asyncio.to_thread(self.reid_extractor.extract, frame, (x1_i, y1_i, x2_i, y2_i))
                        if feat is not None and (np.isnan(feat).any() or np.isinf(feat).any()):
                            feat = None
                    except Exception:
                        feat = None

                hist = None
                if feat is None:
                    hist = compute_hsv_hist(frame, (x1_i, y1_i, x2_i, y2_i))

                pid, remapped, remapped_from = self.tm.register_track(int(strong_id), (x1, y1, x2, y2), feat, hist, frame_count)

                color = (0, 255, 0)
                if remapped:
                    color = (0, 165, 255)
                    prev_cent = self.tm.persist_data.get(pid, {}).get("last_centroid", None)
                    cur_cent = centroid((x1, y1, x2, y2))
                    if prev_cent is not None:
                        p1 = (int(prev_cent[0]), int(prev_cent[1]))
                        p2 = (int(cur_cent[0]), int(cur_cent[1]))
                        h_f, w_f = out_frame.shape[:2]
                        p1 = (max(0, min(w_f-1, p1[0])), max(0, min(h_f-1, p1[1])))
                        p2 = (max(0, min(w_f-1, p2[0])), max(0, min(h_f-1, p2[1])))
                        cv2.arrowedLine(out_frame, p1, p2, color, 2, tipLength=0.2)

                label = f" S{int(strong_id)} {conf_val:.2f}"
                cv2.rectangle(out_frame, (x1_i, y1_i), (x2_i, y2_i), color, 2)
                cv2.putText(out_frame, label, (x1_i, max(0, y1_i-8)), cv2.FONT_HERSHEY_SIMPLEX, 1.25, color, 2)

                if SAVE_CSV:
                    # write in thread to avoid blocking (but protect CSV with lock)
                    def _write_csv():
                        with csv_lock:
                            csv_writer.writerow([timestamp, self.stream_id, int(pid), int(strong_id), float(x1), float(y1), float(x2), float(y2), float(conf_val)])
                            traj_writer.writerow([timestamp, self.stream_id, int(pid), float(x1), float(y1), float(x2), float(y2), float(conf_val)])
                    await asyncio.to_thread(_write_csv)

                if SAVE_CROPS:
                    try:
                        crop_dir = Path(f"StrongSort_saved_output/crops/stream{self.stream_id}/persist_{int(pid)}")
                        crop_dir.mkdir(parents=True, exist_ok=True)
                        crop_img = frame[y1_i:y2_i, x1_i:x2_i]
                        if crop_img.size != 0:
                            crop_name = crop_dir / f"{int(timestamp*1000)}.jpg"
                            # save in thread
                            await asyncio.to_thread(cv2.imwrite, str(crop_name), crop_img)
                    except Exception as e:
                        print(f"[Stream {self.stream_id}] Crop save error:", e)

            # drop stale strong->persist mappings not updated recently to allow remap
            mapped_strong = list(self.tm.strong2persist.keys())
            for s in mapped_strong:
                pid = self.tm.strong2persist.get(s)
                if pid is None:
                    continue
                last_frame = self.tm.persist_data.get(pid, {}).get("last_frame", 0)
                if frame_count - last_frame > 6:
                    self.tm.strong2persist.pop(s, None)

            try:
                resized = cv2.resize(out_frame, (TILE_W, TILE_H))
            except Exception:
                resized = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
            self.output_frame = resized

            if self.writer is not None:
                # write frame in thread
                try:
                    await asyncio.to_thread(self.writer.write, resized)
                except Exception as e:
                    print(f"[Stream {self.stream_id}] Video write error:", e)

            # tiny sleep to yield control (tunable)
            await asyncio.sleep(0.001)

    def stop(self):
        self.stopped = True
        try:
            self.grab_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self.cap.release()
        except Exception:
            pass
        if self.writer is not None:
            try:
                self.writer.release()
            except Exception:
                pass

# ---------------------------
# init reid extractor (best-effort)
# ---------------------------
reid_extractor = None
if USE_DEEP_REID:
    try:
        reid_extractor = ReIDExtractor(REID_CHECKPOINT if REID_CHECKPOINT.exists() else None, device=REID_DEVICE)
        if reid_extractor.model is None:
            reid_extractor = None
    except Exception as e:
        print("[Main] ReID init failed:", e)
        reid_extractor = None

# ---------------------------
# Launch streams (create workers)
# ---------------------------
workers = []
# shared executor for blocking calls (optional)
shared_executor = ThreadPoolExecutor(max_workers=max(4, len(RTSP_STREAMS) * 2))
for i, url in enumerate(RTSP_STREAMS):
    w = StreamWorker(url, i, reid_extractor, executor=shared_executor)
    workers.append(w)

# wait a short moment for grab threads to start filling frames
time.sleep(0.5)

# ---------------------------
# Grid writer init
# ---------------------------
grid_writer = None
if SAVE_GRID:
    rows = (len(workers) + GRID_COLS - 1) // GRID_COLS
    grid_w = TILE_W * GRID_COLS
    grid_h = TILE_H * rows
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    grid_writer = cv2.VideoWriter("StrongSort_saved_output/saved_videos/combined_grid.mp4", fourcc, GRID_FPS, (grid_w, grid_h))

def make_grid(frames, cols=GRID_COLS, tile_w=TILE_W, tile_h=TILE_H):
    if len(frames) == 0:
        return None
    rows = (len(frames) + cols - 1) // cols
    grid = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    for idx, frame in enumerate(frames):
        r = idx // cols
        c = idx % cols
        try:
            grid[r*tile_h:(r+1)*tile_h, c*tile_w:(c+1)*tile_w] = frame
        except Exception:
            # if frame shape mismatch, fill with blank tile
            grid[r*tile_h:(r+1)*tile_h, c*tile_w:(c+1)*tile_w] = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    return grid

# ---------------------------
# Async main loop
# ---------------------------
async def main_loop():
    stop_event = asyncio.Event()
    # schedule worker processing tasks
    tasks = []
    for w in workers:
        t = asyncio.create_task(w.process_loop(stop_event))
        tasks.append(t)

    try:
        while True:
            # build tiles for grid
            tiles = []
            for w in workers:
                if hasattr(w, "output_frame") and w.output_frame is not None:
                    tiles.append(w.output_frame)
                else:
                    tiles.append(np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8))

            grid = make_grid(tiles)
            if grid is not None:
                cv2.imshow("Multi-Stream Grid", grid)
                if SAVE_GRID and grid_writer is not None:
                    # write in thread to avoid blocking (but small overhead)
                    await asyncio.to_thread(grid_writer.write, grid)

            # check for 'q' press in a non-blocking way
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("q pressed -> stopping")
                break

            # also allow interrupt via asyncio cancellation (ctrl-c)
            await asyncio.sleep(0.01)
    except KeyboardInterrupt:
        print("KeyboardInterrupt -> stopping")
    finally:
        # signal stop to workers
        stop_event.set()
        # give tasks short grace to finish
        await asyncio.sleep(0.25)
        # cancel remaining tasks
        for t in tasks:
            if not t.done():
                t.cancel()
        # stop threads & release resources
        for w in workers:
            w.stop()
        if grid_writer is not None:
            try:
                grid_writer.release()
            except Exception:
                pass
        if SAVE_CSV:
            try:
                csv_file.close()
                traj_file.close()
            except Exception:
                pass
        cv2.destroyAllWindows()
        print("All outputs saved to folder: StrongSort_saved_output/saved_videos/, crops/, and", CSV_PATH)

if __name__ == "__main__":
    # run asyncio main loop
    try:
        asyncio.run(main_loop())
    except Exception as e:
        print("Fatal error in main loop:", e)
        # attempt final cleanup
        for w in workers:
            try:
                w.stop()
            except Exception:
                pass
        if SAVE_CSV:
            try:
                csv_file.close()
                traj_file.close()
            except Exception:
                pass
        if grid_writer is not None:
            try:
                grid_writer.release()
            except Exception:
                pass
        cv2.destroyAllWindows()
        sys.exit(1)
