# File: multi_rtsp_strongsort_reid_patched.py
# Debug-friendly multi-RTSP + YOLO + StrongSort + InsightFace + gallery fusion pipeline
import os
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
import math
from collections import deque
from torchvision import transforms
import glob
import traceback

# ---------------------------
# CONFIG - edit as needed
# ---------------------------
RTSP_STREAMS = [
    # example streams
    # "rtsp://admin:rolex%40123@192.168.1.112:554/Streaming/channels/101",
   "rtsp://admin:rolex%40123@192.168.1.110:554/Streaming/channels/101",
   "rtsp://admin:rolex%40123@192.168.1.108:554/Streaming/channels/101",

    # "rtsp://admin:rolex%40123@192.168.1.111:554/Streaming/channels/101",

    # "rtmp://localhost/live/stream4"
]

REID_CHECKPOINT = Path("osnet_x0_25_msmt17.pt")  # if available
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

# ----- GALLERY INPUTS (from user) -----
GALLERY_FACE_INPUT = r"D:\clever_bridge\person_tracking\gallery"  # face gallery root (per-person subfolders recommended)
GALLERY_BODY_INPUT = r"D:\clever_bridge\person_tracking\gallery_face"  # body gallery root (per-person subfolders recommended)

# Matching / output choices (from user)
MATCH_RULE = "B"  # fused (we use 0.7 body + 0.3 face)
OUTPUT_LABEL = "B"  # P{pid} (name)
FACE_MATCH_THRESHOLD = 30 # percent similarity (converted to 0..1)
BODY_MATCH_THRESHOLD = 70  # percent similarity

# ---------------------------
# Devices
# ---------------------------
GPU_AVAILABLE = torch.cuda.is_available()
if GPU_AVAILABLE and torch.cuda.device_count() > 0:
    TRACKER_DEVICE = 0
    YOLO_DEVICE = "cuda:0"
    REID_DEVICE = "cuda:0"
else:
    TRACKER_DEVICE = "cpu"
    YOLO_DEVICE = "cpu"
    REID_DEVICE = "cpu"

print(f"[INIT] YOLO device: {YOLO_DEVICE}, Tracker device: {TRACKER_DEVICE}, ReID device: {REID_DEVICE}, GPU_AVAILABLE={GPU_AVAILABLE}")

# ---------------------------
# YOLO model
# ---------------------------
try:
    yolo_model = YOLO("yolo12m.pt")
    try:
        yolo_model.to(YOLO_DEVICE)
    except Exception:
        pass
    print("[INIT] YOLO loaded.")
except Exception as e:
    print("[ERROR] YOLO load failed:", e)
    yolo_model = None

# ---------------------------
# CSV writers + lock
# ---------------------------
csv_lock = threading.Lock()
if SAVE_CSV:
    csv_file = open(CSV_PATH, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["timestamp", "stream_id", "persistent_id", "strongsort_id", "x1", "y1", "x2", "y2", "conf", "label"])
    traj_file = open(TRAJECTORY_CSV, "w", newline="", encoding="utf-8")
    traj_writer = csv.writer(traj_file)
    traj_writer.writerow(["timestamp", "stream_id", "persistent_id", "x1", "y1", "x2", "y2", "conf", "label"])

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
    try:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten().astype(np.float32)
    except Exception:
        return None

def hist_distance(a, b):
    if a is None or b is None:
        return 1.0
    try:
        d = cv2.compareHist(a.astype('float32'), b.astype('float32'), cv2.HISTCMP_BHATTACHARYYA)
        return d
    except Exception:
        try:
            return float(np.linalg.norm(a - b))
        except Exception:
            return 1.0

def centroid(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def safe_cosine_sim(a, b):
    if a is None or b is None:
        return None
    try:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return None
        return float(np.dot(a, b) / denom)
    except Exception:
        return None

# ---------------------------
# Robust ReID loader (from your original code)
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
# Face extractor (INSIGHTFACE primary, fallback to face_recognition -> Haar+hist)
# ---------------------------
class FaceExtractor:
    def __init__(self, use_gpu=True):
        self.backend = None
        self.face_detector = None
        self.insight_app = None
        self.use_gpu = use_gpu and GPU_AVAILABLE
        # Try InsightFace first
        try:
            import insightface
            from insightface.app import FaceAnalysis
            # create app
            self.insight_app = FaceAnalysis(name="buffalo_l")  # buffalo_l is robust/descriptive; falls back to available models
            # ctx_id = 0 for GPU, -1 for CPU (FaceAnalysis uses mxnet by default)
            ctx_id = 0 if self.use_gpu else -1
            try:
                # prepare may raise if model or providers not available; det_size tuned for speed
                self.insight_app.prepare(ctx_id=ctx_id, det_size=(640, 640))
                self.backend = "insightface"
                print(f"[FaceExtractor] Using InsightFace backend (gpu={self.use_gpu}, ctx_id={ctx_id}).")
            except Exception as e:
                print("[FaceExtractor] InsightFace.prepare failed:", e)
                self.insight_app = None
                self.backend = None
        except Exception as e:
            print("[FaceExtractor] InsightFace import failed:", e)
            self.insight_app = None
            self.backend = None

        # If not insightface, try face_recognition
        if self.backend is None:
            try:
                import face_recognition
                self.fr = face_recognition
                self.backend = "face_recognition"
                print("[FaceExtractor] Using face_recognition backend.")
            except Exception:
                self.fr = None
                # try Haar cascade for face detection
                try:
                    haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                    self.face_detector = cv2.CascadeClassifier(haar_path)
                    if self.face_detector.empty():
                        self.face_detector = None
                        self.backend = None
                    else:
                        self.backend = "haar_hist"
                        print("[FaceExtractor] Using Haar face detector + HSV hist fallback.")
                except Exception:
                    self.face_detector = None
                    self.backend = None
                    print("[FaceExtractor] No face backend available; will not extract face deep features.")

    def extract(self, frame_bgr, bbox):
        """
        Returns:
            - numpy vector (normalized) for deep embedding, OR
            - tuple ("hist", hist_vector) if only hist fallback available, OR
            - None if nothing
        """
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame_bgr.shape[:2]
        x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        # InsightFace path
        if self.backend == "insightface" and self.insight_app is not None:
            try:
                # insightface expects BGR or RGB? FaceAnalysis.get expects BGR image (it converts internally)
                faces = self.insight_app.get(roi)
                if not faces or len(faces) == 0:
                    return None
                # Choose largest detection (if multiple)
                faces = sorted(faces, key=lambda f: f.bbox[2]*f.bbox[3], reverse=True)
                emb = faces[0].normed_embedding  # numpy normalized embedding
                if emb is None:
                    return None
                emb = np.asarray(emb, dtype=np.float32)
                # ensure normalization (insightface often returns already normalized)
                n = np.linalg.norm(emb)
                if n == 0 or not np.isfinite(n):
                    return None
                return (emb / n).astype(np.float32)
            except Exception as e:
                print("[FaceExtractor][InsightFace] extract error:", e)
                return None

        # face_recognition path
        if self.backend == "face_recognition" and self.fr is not None:
            try:
                rgb = roi[:, :, ::-1]  # BGR->RGB
                locs = self.fr.face_locations(rgb)
                if len(locs) == 0:
                    return None
                encs = self.fr.face_encodings(rgb, locs)
                if len(encs) == 0:
                    return None
                v = np.asarray(encs[0], dtype=np.float32)
                n = np.linalg.norm(v)
                if n == 0:
                    return None
                return (v / n).astype(np.float32)
            except Exception as e:
                print("[FaceExtractor][face_recognition] extract error:", e)
                return None

        # Haar + hist fallback
        if self.backend == "haar_hist" and self.face_detector is not None:
            try:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30,30))
                if len(faces) == 0:
                    # return hist of whole roi
                    hst = compute_hsv_hist(frame_bgr, (x1, y1, x2, y2))
                    return ("hist", hst) if hst is not None else None
                faces = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
                fx, fy, fw, fh = faces[0]
                face_crop = roi[fy:fy+fh, fx:fx+fw]
                hst = compute_hsv_hist(face_crop, (0,0,face_crop.shape[1], face_crop.shape[0]))
                return ("hist", hst) if hst is not None else None
            except Exception as e:
                print("[FaceExtractor][haar] extract error:", e)
                return None

        return None

# ---------------------------
# Gallery manager
# ---------------------------
class GalleryManager:
    def __init__(self, face_extractor: FaceExtractor, reid_extractor: ReIDExtractor = None,
                 max_gallery_per_person=32):
        self.face_extractor = face_extractor
        self.reid_extractor = reid_extractor
        self.max_gallery_per_person = max_gallery_per_person
        self.gallery = {}  # label -> dict of feats/hists

    def _add_face_image(self, label, img_path):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[Gallery] Failed read face image {img_path}")
                return
            f = self.face_extractor.extract(img, (0,0,img.shape[1], img.shape[0]))
            if f is None:
                return
            if isinstance(f, tuple) and f[0] == "hist":
                self.gallery.setdefault(label, {}).setdefault("face_hists", []).append(f[1])
            else:
                self.gallery.setdefault(label, {}).setdefault("face_feats", []).append(f)
        except Exception:
            print("[Gallery] Exception adding face image:", img_path)
            traceback.print_exc()

    def _add_body_image(self, label, img_path):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[Gallery] Failed read body image {img_path}")
                return
            if self.reid_extractor is not None and self.reid_extractor.model is not None:
                feat = self.reid_extractor.extract(img, (0,0,img.shape[1], img.shape[0]))
                if feat is not None:
                    self.gallery.setdefault(label, {}).setdefault("body_feats", []).append(feat)
                    return
            # fallback hist
            h = compute_hsv_hist(img, (0,0,img.shape[1], img.shape[0]))
            if h is not None:
                self.gallery.setdefault(label, {}).setdefault("body_hists", []).append(h)
        except Exception:
            print("[Gallery] Exception adding body image:", img_path)
            traceback.print_exc()

    def build_from_inputs(self, face_input: str, body_input: str):
        def normalize_paths(inp):
            if inp is None:
                return []
            inp = inp.strip()
            if inp == "":
                return []
            parts = [p.strip() for p in inp.split(",") if p.strip()]
            expanded = []
            for p in parts:
                p = Path(p)
                if p.is_dir():
                    expanded.append(str(p))
                elif p.exists():
                    expanded.append(str(p))
                else:
                    g = glob.glob(str(p))
                    expanded.extend(g)
            return expanded

        face_paths = normalize_paths(face_input)
        body_paths = normalize_paths(body_input)

        def add_path_to_gallery(path_list, is_face=True):
            for p in path_list:
                p = Path(p)
                if p.is_file():
                    label = p.parent.name
                    if is_face:
                        self._add_face_image(label, p)
                    else:
                        self._add_body_image(label, p)
                elif p.is_dir():
                    # subfolders = persons
                    for sub in p.iterdir():
                        if sub.is_dir():
                            label = sub.name
                            imgs = list(sub.glob("*.*"))
                            for img in imgs[:self.max_gallery_per_person]:
                                if is_face:
                                    self._add_face_image(label, img)
                                else:
                                    self._add_body_image(label, img)
                        else:
                            # files directly in p -> treat as label = p.name
                            label = p.name
                            if is_face:
                                self._add_face_image(label, sub)
                            else:
                                self._add_body_image(label, sub)
                else:
                    continue

        add_path_to_gallery(face_paths, is_face=True)
        add_path_to_gallery(body_paths, is_face=False)

        # prune
        for label, data in self.gallery.items():
            if "face_feats" in data:
                data["face_feats"] = data["face_feats"][:self.max_gallery_per_person]
            if "body_feats" in data:
                data["body_feats"] = data["body_feats"][:self.max_gallery_per_person]
            if "face_hists" in data:
                data["face_hists"] = data["face_hists"][:self.max_gallery_per_person]
            if "body_hists" in data:
                data["body_hists"] = data["body_hists"][:self.max_gallery_per_person]

        print("[GalleryManager] Built gallery for labels:", list(self.gallery.keys()))

    def match(self, feat_body, feat_face):
        """
        Input:
            feat_body: numpy vector or None
            feat_face: numpy vector OR ("hist", hist) OR None
        Returns:
            (best_label_or_None, fused_similarity (0..1), face_sim, body_sim)
        """
        best_label = None
        best_fused = 0.0
        best_face_sim = None
        best_body_sim = None

        face_thresh = FACE_MATCH_THRESHOLD / 100.0
        body_thresh = BODY_MATCH_THRESHOLD / 100.0
        fused_required = 0.7 * body_thresh + 0.3 * face_thresh
        # fused_required = 0.8 * body_thresh + 0.2 * face_thresh


        if not self.gallery:
            return None, 0.0, None, None

        for label, data in self.gallery.items():
            label_body_sim = None
            label_face_sim = None

            # body deep features (cosine)
            if feat_body is not None and data.get("body_feats"):
                sims = [safe_cosine_sim(feat_body, b) for b in data["body_feats"]]
                sims = [s for s in sims if s is not None]
                if sims:
                    label_body_sim = max(sims)

            # face deep features
            if feat_face is not None and not isinstance(feat_face, tuple):
                if data.get("face_feats"):
                    sims = [safe_cosine_sim(feat_face, f) for f in data["face_feats"]]
                    sims = [s for s in sims if s is not None]
                    if sims:
                        label_face_sim = max(sims)

            # face hist comparison fallback
            if label_face_sim is None and data.get("face_hists"):
                fh = None
                if isinstance(feat_face, tuple) and feat_face[0] == "hist":
                    fh = feat_face[1]
                # if face deep missing, we could try hist of whole body or skip
                if fh is not None:
                    sims = []
                    for gh in data["face_hists"]:
                        d = hist_distance(fh, gh)
                        sim = max(0.0, 1.0 - float(d))
                        sims.append(sim)
                    if sims:
                        label_face_sim = max(sims)

            # body hist fallback (if no deep)
            if label_body_sim is None and data.get("body_hists"):
                # we don't have body hist from tracker here; skip unless feat_body was actually a hist tuple (not implemented)
                pass

            # treat missing as 0.0
            b_sim = label_body_sim if label_body_sim is not None else 0.0
            f_sim = label_face_sim if label_face_sim is not None else 0.0

            # fused
            if (label_body_sim is not None) and (label_face_sim is not None):
                fused = 0.7 * b_sim + 0.3 * f_sim
            elif label_body_sim is not None:
                fused = b_sim
            elif label_face_sim is not None:
                fused = f_sim
            else:
                fused = 0.0

            # debug
            print(f"[Gallery.match] {label}: body={b_sim:.3f} face={f_sim:.3f} fused={fused:.3f}")

            if fused > best_fused:
                best_fused = fused
                best_label = label
                best_face_sim = f_sim
                best_body_sim = b_sim

        if best_label is None:
            return None, 0.0, None, None
        if best_fused >= fused_required:
            return best_label, best_fused, best_face_sim, best_body_sim
        return None, best_fused, best_face_sim, best_body_sim

# ---------------------------
# TrackManager (persistent id remapping) — extended to store assigned label
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
        self.pid_label = {}  # mapping pid -> assigned label

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
        print(f"[TrackManager] Created PID {pid} for strong_id {strong_id}")
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
            print(f"[TrackManager] Re-created persistent entry for PID {pid}")
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
        for pid, d in list(self.persist_data.items()):
            if current_frame_idx - d["last_frame"] > self.max_age_frames:
                d["active"] = False
                # keep pid_label for history; optionally remove to free memory
                # self.pid_label.pop(pid, None)

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
                feat_dist = min(self._cosine_dist(feat, f) for f in d["feats"] if f is not None)
            else:
                feat_dist = 1.0
            if hist is not None and len(d["hists"]) > 0:
                hist_dist = min(hist_distance(hist, h) for h in d["hists"] if h is not None)
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
            print(f"[TrackManager] Strong {strong_id} remapped to existing PID {matched_pid} (score {score:.3f})")
            return matched_pid, remapped, remapped_from

        pid = self._create_persistent(strong_id, bbox, feat, hist, frame_idx)
        return pid, remapped, remapped_from

    def strong_id_removed(self, strong_id):
        if strong_id in self.strong2persist:
            pid = self.strong2persist.pop(strong_id, None)
            print(f"[TrackManager] Removed strong->persist mapping for strong {strong_id} -> pid {pid}")
            return pid
        return None

# ---------------------------
# StreamWorker (integrates face/gallery matching)
# ---------------------------
class StreamWorker:
    def __init__(self, stream_url, stream_id, reid_extractor=None, face_extractor=None, gallery_manager=None):
        self.stream_url = stream_url
        self.stream_id = stream_id
        self.latest_frame = None
        self.output_frame = None
        self.stopped = False
        self.frame_idx = 0
        self.reid_extractor = reid_extractor
        self.face_extractor = face_extractor
        self.gallery_manager = gallery_manager

        self.cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 4)
        except Exception:
            pass
        if not self.cap.isOpened():
            print(f"[Stream {self.stream_id}] ERROR: Cannot open stream!")

        # StrongSort instance (keep your original params)
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
            print(f"[Stream {self.stream_id}] StrongSort initialized.")
        except Exception as e:
            print(f"[Stream {self.stream_id}] StrongSort init failed:", e)
            self.tracker = None

        self.tm = TrackManager(max_age_frames=150, feat_match_thresh=0.45, hist_match_thresh=0.45, max_centroid_dist=160)

        self.writer = None
        if SAVE_PER_CAMERA:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_path = f"StrongSort_saved_output/saved_videos/stream_{self.stream_id}_output.mp4"
            try:
                self.writer = cv2.VideoWriter(out_path, fourcc, GRID_FPS, (TILE_W, TILE_H))
            except Exception as e:
                print(f"[Stream {self.stream_id}] VideoWriter init failed:", e)
                self.writer = None

        self.remap_visuals = {}

        self.grab_thread = threading.Thread(target=self.grab_frames, daemon=True)
        self.grab_thread.start()
        self.process_thread = threading.Thread(target=self.process_frames, daemon=True)
        self.process_thread.start()

    def grab_frames(self):
        while not self.stopped:
            try:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    print(f"[Stream {self.stream_id}] WARNING: Unable to read frame")
                    time.sleep(1)
                    continue
                if np.isnan(frame).any() or np.isinf(frame).any():
                    frame = np.nan_to_num(frame, nan=0, posinf=255, neginf=0).astype(np.uint8)
                self.latest_frame = frame
            except Exception as e:
                print(f"[Stream {self.stream_id}] grab_frames exception:", e)
                time.sleep(0.5)

    def process_frames(self):
        while not self.stopped:
            if self.latest_frame is None:
                time.sleep(0.01)
                continue

            frame = self.latest_frame.copy()
            self.frame_idx += 1
            frame_count = self.frame_idx

            # YOLO detection
            try:
                if yolo_model is None:
                    det = np.empty((0, 6), dtype=np.float32)
                else:
                    results = yolo_model(frame, conf=0.22, iou=0.25, verbose=False)
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
            except Exception as e:
                print(f"[Stream {self.stream_id}] YOLO inference error:", e)
                det = np.empty((0, 6), dtype=np.float32)

            # SANITIZE detections
            try:
                if det is not None and len(det) > 0:
                    det = det[np.isfinite(det).all(axis=1)]
                    mask = (det[:, 2] > det[:, 0]) & (det[:, 3] > det[:, 1])
                    det = det[mask]
                    if det.size == 0:
                        det = np.empty((0, 6), dtype=np.float32)
            except Exception as e:
                print(f"[Stream {self.stream_id}] Detection sanitize error:", e)
                det = np.empty((0, 6), dtype=np.float32)

            # tracker update
            tracked = []
            try:
                if self.tracker is None:
                    tracked = []
                else:
                    if det is None or len(det) == 0:
                        tracked = self.tracker.update(np.empty((0,6), dtype=np.float32), frame)
                    else:
                        if frame_count % 100 == 1:
                            print(f"[Stream {self.stream_id}] DEBUG frame {frame_count} det_array (first rows):\n{det[:6]}")
                        tracked = self.tracker.update(det, frame)
            except Exception as e:
                print(f"[Stream {self.stream_id}] Tracker update error: {e}")
                time.sleep(0.01)
                continue

            # cleanup persistent entries
            self.tm.cleanup_old(frame_count)

            out_frame = frame.copy()
            timestamp = time.time()
            seen_strong_ids = set()

            for t in tracked:
                # StrongSort tracked output format may vary; try to be defensive
                try:
                    x1, y1, x2, y2, strong_id, conf_val, cls_val, ind = t
                except Exception:
                    # fallback common layout
                    try:
                        x1, y1, x2, y2 = float(t[0]), float(t[1]), float(t[2]), float(t[3])
                        strong_id = int(t[4]) if len(t) > 4 else -1
                        conf_val = float(t[5]) if len(t) > 5 else 0.0
                    except Exception:
                        # skip malformed track
                        continue

                if not np.isfinite([x1, y1, x2, y2]).all():
                    continue

                x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])
                seen_strong_ids.add(int(strong_id))

                # Deep ReID feature if available, else hist
                feat = None
                if USE_DEEP_REID and self.reid_extractor is not None:
                    try:
                        feat = self.reid_extractor.extract(frame, (x1_i, y1_i, x2_i, y2_i))
                        if feat is not None and (np.isnan(feat).any() or np.isinf(feat).any()):
                            feat = None
                    except Exception as e:
                        print(f"[Stream {self.stream_id}] ReID extraction error:", e)
                        feat = None

                hist = None
                if feat is None:
                    hist = compute_hsv_hist(frame, (x1_i, y1_i, x2_i, y2_i))

                # Face feature (InsightFace or fallback)
                face_feat = None
                try:
                    if self.face_extractor is not None:
                        ffe = self.face_extractor.extract(frame, (x1_i, y1_i, x2_i, y2_i))
                        if isinstance(ffe, tuple) and ffe[0] == "hist":
                            face_feat = ffe  # ("hist", hist_vector)
                        else:
                            face_feat = ffe  # normalized vector or None
                except Exception as e:
                    print(f"[Stream {self.stream_id}] Face extraction error:", e)
                    face_feat = None

                # register into persistent TM
                pid, remapped, remapped_from = self.tm.register_track(int(strong_id), (x1, y1, x2, y2), feat, hist, frame_count)

                # attempt gallery match for label assignment (fusion)
                assigned_label = None
                try:
                    if self.gallery_manager is not None:
                        b_feat = feat  # deep body feat or None
                        f_feat = face_feat
                        label, fused_sim, f_sim, b_sim = self.gallery_manager.match(b_feat, f_feat)
                        if label is not None:
                            assigned_label = label
                            # store mapping pid->label
                            self.tm.pid_label[pid] = label
                            print(f"[Stream {self.stream_id}] PID {pid} assigned label '{label}' (fused_sim={fused_sim:.3f})")
                except Exception as e:
                    print(f"[Stream {self.stream_id}] Gallery match error:", e)
                    traceback.print_exc()

                # use previously assigned label if available
                stored_label = self.tm.pid_label.get(pid, None)
                display_label = stored_label if stored_label is not None else assigned_label

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

                label_text = f"S{int(strong_id)} {conf_val:.2f}"
                if display_label is not None:
                    if OUTPUT_LABEL == "B":
                        label_text = f"P{int(pid)} ({display_label})"
                    else:
                        label_text = f"P{int(pid)} {display_label}"

                # draw bounding box and label
                try:
                    cv2.rectangle(out_frame, (x1_i, y1_i), (x2_i, y2_i), color, 2)
                    cv2.putText(out_frame, label_text, (x1_i, max(0, y1_i-8)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
                except Exception as e:
                    print(f"[Stream {self.stream_id}] Draw error:", e)

                # CSV logging with label
                if SAVE_CSV:
                    try:
                        with csv_lock:
                            csv_writer.writerow([timestamp, self.stream_id, int(pid), int(strong_id), float(x1), float(y1), float(x2), float(y2), float(conf_val), display_label if display_label else ""])
                            traj_writer.writerow([timestamp, self.stream_id, int(pid), float(x1), float(y1), float(x2), float(y2), float(conf_val), display_label if display_label else ""])
                    except Exception as e:
                        print(f"[Stream {self.stream_id}] CSV write error:", e)

                # Save crops
                if SAVE_CROPS:
                    try:
                        crop_dir = Path(f"StrongSort_saved_output/crops/stream{self.stream_id}/persist_{int(pid)}")
                        crop_dir.mkdir(parents=True, exist_ok=True)
                        crop_img = frame[y1_i:y2_i, x1_i:x2_i]
                        if crop_img.size != 0:
                            crop_name = crop_dir / f"{int(timestamp*1000)}.jpg"
                            cv2.imwrite(str(crop_name), crop_img)
                    except Exception as e:
                        print(f"[Stream {self.stream_id}] Crop save error:", e)

            # drop stale strong->persist mappings not updated recently to allow remap
            try:
                mapped_strong = list(self.tm.strong2persist.keys())
                for s in mapped_strong:
                    pid = self.tm.strong2persist.get(s)
                    if pid is None:
                        continue
                    last_frame = self.tm.persist_data.get(pid, {}).get("last_frame", 0)
                    if frame_count - last_frame > 6:
                        self.tm.strong2persist.pop(s, None)
            except Exception as e:
                print(f"[Stream {self.stream_id}] cleanup mapping error:", e)

            try:
                resized = cv2.resize(out_frame, (TILE_W, TILE_H))
            except Exception:
                resized = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
            self.output_frame = resized

            if self.writer is not None:
                try:
                    self.writer.write(resized)
                except Exception as e:
                    print(f"[Stream {self.stream_id}] Video write error:", e)

            # small sleep to yield
            time.sleep(0.001)

    def stop(self):
        self.stopped = True
        try:
            self.grab_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self.process_thread.join(timeout=1.0)
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
# Init face extractor, reid extractor, gallery
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

face_extractor = None
try:
    face_extractor = FaceExtractor(use_gpu=True)
except Exception as e:
    print("[Main] FaceExtractor init failed:", e)
    face_extractor = None

gallery_manager = GalleryManager(face_extractor, reid_extractor)
gallery_manager.build_from_inputs(GALLERY_FACE_INPUT, GALLERY_BODY_INPUT)

# ---------------------------
# Launch streams
# ---------------------------
workers = []
for i, url in enumerate(RTSP_STREAMS):
    try:
        w = StreamWorker(url, i, reid_extractor, face_extractor, gallery_manager)
        workers.append(w)
    except Exception as e:
        print(f"[Main] Failed to start worker for stream {i} ({url}):", e)

time.sleep(1.0)

# ---------------------------
# Grid writer init
# ---------------------------
grid_writer = None
if SAVE_GRID:
    try:
        rows = (len(workers) + GRID_COLS - 1) // GRID_COLS
        grid_w = TILE_W * GRID_COLS
        grid_h = TILE_H * rows
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        grid_writer = cv2.VideoWriter("StrongSort_saved_output/saved_videos/combined_grid.mp4", fourcc, GRID_FPS, (grid_w, grid_h))
    except Exception as e:
        print("[Main] Grid writer init failed:", e)
        grid_writer = None

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
            # frame shape mismatch -> fill black
            pass
    return grid

# ---------------------------
# Main display + save loop
# ---------------------------
try:
    print("[Main] Entering main display loop. Press 'q' to quit.")
    while True:
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
                try:
                    grid_writer.write(grid)
                except Exception as e:
                    print("[Main] Grid write error:", e)

        # use q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("Stopping... (KeyboardInterrupt)")

except Exception as e:
    print("[Main] Exception in display loop:", e)
    traceback.print_exc()

# ---------------------------
# Cleanup
# ---------------------------
for w in workers:
    try:
        w.stop()
    except Exception:
        pass

if grid_writer is not None:
    try:
        grid_writer.release()
    except Exception:
        pass

cv2.destroyAllWindows()
if SAVE_CSV:
    try:
        csv_file.close()
        traj_file.close()
    except Exception:
        pass

print("All outputs saved to folder: StrongSort_saved_output/, and", CSV_PATH)
