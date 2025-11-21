# updated_reid_insightface_multicam.py  right main correct code after the copy 3
import asyncio
import os
import cv2
import numpy as np
import sqlite3
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO
import time
import threading
from queue import Queue
import traceback
import math
import sqlite3

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except Exception:
    INSIGHTFACE_AVAILABLE = False
    print("⚠️ insightface not available. Install with: pip install insightface")

# ---------------- CONFIG ----------------
YOLO_WEIGHTS = "yolo12m.pt"   # your YOLO weights file
GALLERY_DIR = "gallery_face"
CAMERA_SOURCES = ["rtsp://admin:rolex@123@192.168.1.111:554/Streaming/channels/101","rtsp://admin:rolex@123@192.168.1.112:554/Streaming/channels/101","rtsp://admin:rolex@123@192.168.1.110:554/Streaming/channels/101"]  # example
EMB_DB = "embeddings.db"
FACE_DB = "faces.db"
OUTPUT_DIR = "outputs"

CONF_THRESHOLD = 0.7
IOU_THRESHOLD = 0.4
MATCH_THRESHOLD = 0.90
FACE_MATCH_THRESHOLD = 0.80
MIN_BOX_AREA = 400
device = "cuda" if torch.cuda.is_available() else "cpu"
CTX_ID = 0 if torch.cuda.is_available() and INSIGHTFACE_AVAILABLE else -1


TRACK_COOLDOWN_SEC = 5.0

# ---------------- FEATURE EXTRACTOR ----------------
print("🔍 Loading ResNet50 feature extractor (body embeddings)...")
resnet = models.resnet50(pretrained=True)
feature_extractor = torch.nn.Sequential(*(list(resnet.children())[:-1]))
feature_extractor.eval().to(device)

preprocess = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


face_app = None
if INSIGHTFACE_AVAILABLE:
    try:
        print("🔎 Initializing InsightFace FaceAnalysis...")

        face_app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
        face_app.prepare(ctx_id=CTX_ID, det_size=(640, 640))
        print("✅ InsightFace ready (ctx_id=%s)" % str(CTX_ID))
    except Exception:
        traceback.print_exc()
        face_app = None


def _get_table_info(db_path, table_name):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    try:
        c.execute(f"PRAGMA table_info({table_name})")
        info = c.fetchall()
    except Exception:
        info = []
    conn.close()
    return info

def init_dbs():

    conn = sqlite3.connect(EMB_DB)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS people_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            embedding BLOB,
            ts REAL
        )
    """)
    conn.commit()
    conn.close()


    conn = sqlite3.connect(FACE_DB)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            embedding BLOB
        )
    """)
    conn.commit()
    conn.close()

def np_to_blob(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def blob_to_np(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)

def load_all_embeddings(db, table):
    conn = sqlite3.connect(db)
    c = conn.cursor()
    try:
        c.execute(f"SELECT name, embedding FROM {table}")
        rows = c.fetchall()
    except Exception:
        rows = []
    conn.close()
    out = []
    for name, blob in rows:
        try:
            emb = blob_to_np(blob).astype(np.float32).flatten()
            out.append((name, emb))
        except Exception:
            continue
    return out

def insert_body_embedding(name, emb: np.ndarray):
    # insert new exemplar (we keep multiple exemplars per name)
    conn = sqlite3.connect(EMB_DB)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO people_embeddings (name, embedding, ts) VALUES (?, ?, ?)",
                  (name, np_to_blob(emb.astype(np.float32)), time.time()))
        conn.commit()
    except Exception:
        traceback.print_exc()
    finally:
        conn.close()

# ---------------- FACE GALLERY (using InsightFace) ----------------
def build_face_gallery(gallery_dir):
    if not os.path.exists(gallery_dir):
        print("⚠️ Gallery not found:", gallery_dir)
        return []
    if face_app is None:
        print("⚠️ InsightFace not initialized; skipping face gallery.")
        return []

    entries = []
    conn = sqlite3.connect(FACE_DB)
    c = conn.cursor()

    for person in os.listdir(gallery_dir):
        pdir = os.path.join(gallery_dir, person)
        if not os.path.isdir(pdir):
            continue
        added = 0
        for fn in sorted(os.listdir(pdir)):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(pdir, fn)
            try:
                img = cv2.imread(path)
                if img is None:
                    continue
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                faces = face_app.get(rgb)
                if not faces:
                    continue
                # take first face
                emb = np.array(faces[0].embedding, dtype=np.float32)
                entries.append((person, np_to_blob(emb)))
                added += 1
            except Exception:
                traceback.print_exc()
                continue
        if added:
            print(f"📸 {person}: added {added} face embeddings")
    if entries:
        try:
            c.executemany("INSERT INTO face_embeddings(name, embedding) VALUES (?, ?)", entries)
            conn.commit()
        except Exception:
            traceback.print_exc()
    conn.close()
    return load_all_embeddings(FACE_DB, "face_embeddings")

# ---------------- EMBEDDING EXTRACTION ----------------
def extract_body_embedding(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    x1, x2 = max(0, x1), min(frame.shape[1] - 1, x2)
    y1, y2 = max(0, y1), min(frame.shape[0] - 1, y2)
    if x2 <= x1 or y2 <= y1 or (x2 - x1) * (y2 - y1) < MIN_BOX_AREA:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    try:
        img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    except Exception:
        return None
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = feature_extractor(tensor)
        feats = feats.reshape(1, -1)
        feats = F.normalize(feats, p=2, dim=1)
        return feats.cpu().numpy().flatten()

def extract_face_embedding_from_crop_insight(crop_rgb):
    if face_app is None:
        return None
    try:
        faces = face_app.get(crop_rgb)
        if not faces:
            return None
        # take primary face
        emb = np.array(faces[0].embedding, dtype=np.float32)
        # L2-normalize (InsightFace embeddings are typically normalized)
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        return emb
    except Exception:
        return None

# ---------------- MATCHING ----------------
def find_best_face_match_grouped(query_emb, face_memory):
    # Filter embeddings that have the same shape as query_emb
    valid_entries = [(name, e) for name, e in face_memory if e is not None and e.shape == query_emb.shape]

    if not valid_entries:
        return None, 1.0  # No valid match

    embs = np.stack([e for _, e in valid_entries])
    names = [name for name, _ in valid_entries]

    # Compute cosine distance
    dists = np.sum(query_emb * embs, axis=1) / (
        np.linalg.norm(query_emb) * np.linalg.norm(embs, axis=1)
    )
    best_idx = np.argmax(dists)
    best_name = names[best_idx]
    best_dist = 1 - dists[best_idx]  # Convert to distance-like score
    return best_name, best_dist

def find_best_body_match(emb, body_memory):
    # body_memory: list of (name, emb) with many exemplars allowed
    if emb is None or not body_memory:
        return None, -1.0
    embs = np.stack([e for _, e in body_memory])
    names = [n for n, _ in body_memory]
    # normalize
    embs_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
    emb_norm = emb / (np.linalg.norm(emb) + 1e-12)
    sims = np.dot(emb_norm, embs_norm.T)  # cosine similarities
    idx = int(np.argmax(sims))
    return names[idx], float(sims[idx])


class MultiCamReIDAsync:
    def __init__(self, sources):
        self.sources = sources
        self.body_memory = load_all_embeddings(EMB_DB, "people_embeddings")
        self.face_memory = load_all_embeddings(FACE_DB, "face_embeddings")
        self.frames = [None] * len(sources)
        self.stop_flag = False
        self.queues = [Queue(maxsize=10) for _ in sources]
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # track_id mapping: global key = (cam_idx, track_id) -> dict {name, last_seen, emb}
        self.track_map = {}
        # currently assigned names in active frame across cameras to avoid duplicates
        self.active_assigned_names = set()
        # lock for thread safety during writes
        self.lock = threading.Lock()

    def _capture_thread(self, cam_idx, source, queue: Queue):

        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        while not self.stop_flag:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            if queue.full():
                try:
                    queue.get_nowait()
                except Exception:
                    pass
            queue.put(frame)
        cap.release()
    def is_duplicate_embedding(self, name: str, new_emb: np.ndarray, threshold: float = 0.92) -> bool:
        """
        Returns True if the new body embedding is too similar to an existing one for the same person.
        Works whether self.body_memory is a dict{name: [embs]} or a flat list of (name, emb).
        """
        if not isinstance(self.body_memory, (dict, list)) or new_emb is None:
            return False


        existing_embs = []
        if isinstance(self.body_memory, dict):
            existing_embs = self.body_memory.get(name, [])
        else:

            existing_embs = [emb for n, emb in self.body_memory if n == name]

        if not existing_embs:
            return False

        new_norm = np.linalg.norm(new_emb)
        for old_emb in existing_embs:
            if old_emb is None:
                continue
            sim = float(np.dot(new_emb, old_emb) / (new_norm * np.linalg.norm(old_emb)))
            if sim >= threshold:
                return True
        return False

    async def process_camera(self, cam_idx, source):
        threading.Thread(target=self._capture_thread, args=(cam_idx, source, self.queues[cam_idx]), daemon=True).start()
        queue = self.queues[cam_idx]

        while queue.empty() and not self.stop_flag:
            await asyncio.sleep(0.01)
        if queue.empty():
            print(f"❌ Camera {cam_idx} has no frames")
            return

        first_frame = queue.get()
        height, width = first_frame.shape[:2]
        fps = 25
        out_path = os.path.join(OUTPUT_DIR, f"cam{cam_idx}.mp4")
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        yolo = YOLO(YOLO_WEIGHTS)
        yolo.to(device)
        print(f"🎥 Camera[{cam_idx}] started -> writing to {out_path}")

        small_w, small_h = 640, 360

        while not self.stop_flag:
            if queue.empty():
                await asyncio.sleep(0.005)
                continue

            frame = queue.get()
            annotated = cv2.resize(frame, (small_w, small_h))

            try:
                results = yolo.track(annotated, persist=True, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
            except Exception:
                await asyncio.sleep(0.01)
                continue

            frame_assigned = {}

            if results and len(results) > 0 and getattr(results[0], "boxes", None) is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                if ids is None:
                    ids = [None] * len(boxes)
                detections = []
                for bbox, cls_idx, track_id in zip(boxes, classes, ids):
                    cls_name = yolo.names[int(cls_idx)]
                    if cls_name.lower() != "person":
                        continue
                    x1, y1, x2, y2 = map(int, bbox)
                    if (x2 - x1) * (y2 - y1) < MIN_BOX_AREA:
                        continue
                    detections.append((bbox, track_id))

                # Extract embeddings
                det_infos = []
                for bbox, track_id in detections:
                    x1, y1, x2, y2 = map(int, bbox)
                    crop = annotated[y1:y2, x1:x2] if (0 <= y1 < y2 and 0 <= x1 < x2) else None
                    face_emb = None
                    if crop is not None:
                        face_emb = extract_face_embedding_from_crop_insight(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    body_emb = extract_body_embedding(annotated, bbox)
                    det_infos.append({
                        "bbox": bbox,
                        "track_id": track_id,
                        "face_emb": face_emb,
                        "body_emb": body_emb,
                        "assigned_name": None,
                        "best_score": None,
                        "face_confirmed": False
                    })

                # --- Matching ---
                candidates = []
                for i, det in enumerate(det_infos):
                    # --- FACE VERIFICATION ---
                    if det["face_emb"] is not None and self.face_memory:
                        fname, fdist = find_best_face_match_grouped(det["face_emb"], self.face_memory)
                        if fname is not None and fdist <= FACE_MATCH_THRESHOLD:
                            det["assigned_name"] = fname
                            det["face_confirmed"] = True
                            det["verified"] = True
                            continue


                    if det["body_emb"] is not None and self.body_memory:
                        bname, bsim = find_best_body_match(det["body_emb"], self.body_memory)
                        if bname is not None and bsim >= MATCH_THRESHOLD:
                            det["assigned_name"] = bname
                            det["best_score"] = bsim
                        else:
                            det["best_score"] = bsim


                now_ts = time.time()
                with self.lock:
                    for det in det_infos:
                        key = (cam_idx, det["track_id"])
                        prev = self.track_map.get(key, {})

                        prev_name = prev.get("name")
                        prev_emb = prev.get("emb")
                        prev_locked = prev.get("locked", False)
                        prev_verified = prev.get("verified", False)
                        last_seen = prev.get("last_seen", 0)

                        assigned_name = det.get("assigned_name")


                        if prev_locked and (now_ts - last_seen) < 15.0:
                            det["assigned_name"] = prev_name
                            assigned_name = prev_name


                        elif det.get("face_confirmed") or det.get("verified"):
                            det["assigned_name"] = assigned_name
                            prev_locked = True
                            prev_verified = True


                        elif not assigned_name and prev_name and (now_ts - last_seen) < 10.0:
                            det["assigned_name"] = prev_name
                            assigned_name = prev_name


                        colliding = False
                        for other in det_infos:
                            if other is det:
                                continue
                            bx1, by1, bx2, by2 = det["bbox"]
                            ox1, oy1, ox2, oy2 = other["bbox"]
                            inter_x1, inter_y1 = max(bx1, ox1), max(by1, oy1)
                            inter_x2, inter_y2 = min(bx2, ox2), min(by2, oy2)
                            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                            union_area = (bx2 - bx1) * (by2 - by1) + (ox2 - ox1) * (oy2 - oy1) - inter_area
                            iou = inter_area / (union_area + 1e-6)
                            if iou > 0.4:
                                colliding = True
                                break

                        if colliding and prev_name:
                            det["assigned_name"] = prev_name
                            assigned_name = prev_name


                        self.track_map[key] = {
                            "name": assigned_name,
                            "emb": det["body_emb"] if det["body_emb"] is not None else prev_emb,
                            "last_seen": now_ts,
                            "locked": prev_locked,
                            "verified": prev_verified,
                        }


                        if det.get("face_confirmed") and det["body_emb"] is not None:
                            if not self.is_duplicate_embedding(assigned_name, det["body_emb"]):
                                insert_body_embedding(assigned_name, det["body_emb"])
                                self.body_memory = load_all_embeddings(EMB_DB, "people_embeddings")
                                print(f"✅ Added new body embedding for {assigned_name}")
                            else:
                                print(f"⚠️ Skipped duplicate embedding for {assigned_name}")


                    for i, det in enumerate(det_infos):
                        bbox = det["bbox"]
                        x1, y1, x2, y2 = map(int, bbox)
                        assigned = det["assigned_name"]
                        color = (0, 255, 0) if det.get("face_confirmed") else (0, 200, 0)
                        label = assigned or f"Person_{cam_idx}_{det['track_id']}"
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(annotated, label, (x1, max(15, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                self.frames[cam_idx] = annotated
                out.write(cv2.resize(annotated, (width, height)))
                await asyncio.sleep(0.005)

        out.release()
        print(f"🛑 Camera[{cam_idx}] stopped -> {out_path}")



    async def show_all(self):
        try:
            while not self.stop_flag:
                # refresh global active_assigned_names each loop
                self.active_assigned_names = set()
                frames = [f for f in self.frames if f is not None]
                if frames:
                    n = len(frames)
                    cols = int(np.ceil(np.sqrt(n)))
                    rows = int(np.ceil(n / cols))
                    h, w = frames[0].shape[:2]
                    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
                    for idx, f in enumerate(frames):
                        r, c = divmod(idx, cols)
                        grid[r*h:(r+1)*h, c*w:(c+1)*w] = f

                    cv2.imshow("All Cameras", grid)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop_flag = True
                    break

                # cleanup stale track_map entries
                now_ts = time.time()
                with self.lock:
                    to_delete = []
                    for k, v in list(self.track_map.items()):
                        if now_ts - v.get("last_seen", 0) > (TRACK_COOLDOWN_SEC * 5):
                            to_delete.append(k)
                    for k in to_delete:
                        del self.track_map[k]

                await asyncio.sleep(0.03)
        finally:
            cv2.destroyAllWindows()

    async def start(self):

        self.body_memory = load_all_embeddings(EMB_DB, "people_embeddings")
        self.face_memory = load_all_embeddings(FACE_DB, "face_embeddings")
        tasks = [self.process_camera(i, src) for i, src in enumerate(self.sources)]
        tasks.append(self.show_all())
        await asyncio.gather(*tasks)


def main():
    init_dbs()
    print("🔁 Building / loading face gallery...")
    face_mem = build_face_gallery(GALLERY_DIR)
    mgr = MultiCamReIDAsync(CAMERA_SOURCES)
    mgr.face_memory = face_mem
    try:
        asyncio.run(mgr.start())
    except KeyboardInterrupt:
        print("🔚 KeyboardInterrupt received; stopping.")
    except Exception:
        traceback.print_exc()
    print("✅ Done. Check outputs directory:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
