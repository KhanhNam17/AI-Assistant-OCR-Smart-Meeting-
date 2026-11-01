"""
core/voiceid_database.py  — v2 (multi-file per ID)
---------------------------------------------------
THAY ĐỔI so với v1:
    - add_voiceid(): tên đã tồn tại → THÊM audio vào ID cũ (không tạo ID mới)
    - DB lưu `vector_paths: list[str]` + `preview_paths: list[str]`
      (backward compat: đọc cả key cũ string)
    - add_audio_to_voiceid(vid, wav_bytes): API mới
    - load_vectors_for_entry(entry): load tất cả embedding
    - delete_voiceid(): xoá tất cả file của entry
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import uuid
import wave
from pathlib import Path
from typing import Optional

import numpy as np

import ssl, certifi
_orig_ssl = ssl.create_default_context
def _patched_ssl(*a, **kw):
    if not any(k in kw for k in ('cafile','capath','cadata')): kw['cafile'] = certifi.where()
    return _orig_ssl(*a, **kw)
ssl.create_default_context = _patched_ssl

log = logging.getLogger(__name__)

ENROLL_DIR    = os.getenv("VOICEID_ENROLL_DIR", "enrollment_data")
DB_PATH       = os.path.join(ENROLL_DIR, "voice_db.json")
PREVIEW_DIR   = os.path.join(ENROLL_DIR, "audio_previews")
VECTOR_DIR    = os.path.join(ENROLL_DIR, "vectors")
TARGET_SR     = 16000
VOICEID_MODEL = os.getenv("VOICEID_MODEL", "models/speech_campplus_sv_zh-cn_16k-common")

_embed_pipeline = None

def _get_embed_pipeline():
    global _embed_pipeline
    if _embed_pipeline is not None: return _embed_pipeline
    try:
        from modelscope.pipelines import pipeline as ms_pipeline
        _embed_pipeline = ms_pipeline(task="speaker-verification", model=VOICEID_MODEL)
    except Exception as e:
        log.warning(f"[VoiceDB] pipeline: {e}"); _embed_pipeline = None
    return _embed_pipeline

def _extract_embedding_from_pipeline(wav_path):
    pipe = _get_embed_pipeline()
    if pipe is None: return None
    try:
        pre = pipe.preprocess([wav_path]); fwd = pipe.forward(pre); emb = None
        if hasattr(fwd,'detach'):          emb = fwd.detach().cpu().numpy()
        elif isinstance(fwd, np.ndarray):  emb = fwd
        elif isinstance(fwd, list):        emb = np.array(fwd)
        elif isinstance(fwd, dict):
            for k in ("spk_embedding","embedding","emb","output"):
                if k in fwd:
                    c = fwd[k]
                    if hasattr(c,'detach'): c = c.detach().cpu().numpy()
                    if isinstance(c,list):  c = np.array(c)
                    if isinstance(c,np.ndarray) and c.size>10: emb=c; break
            if emb is None:
                for v in fwd.values():
                    if hasattr(v,'detach'): v=v.detach().cpu().numpy()
                    if isinstance(v,np.ndarray) and v.size>10: emb=v; break
        if emb is None: return None
        emb = np.array(emb,np.float32).flatten(); n = np.linalg.norm(emb)
        return (emb/n).astype(np.float32) if n>=1e-8 else None
    except Exception as e: log.warning(f"[VoiceDB] emb: {e}"); return None

def _ensure_dirs():
    for d in [ENROLL_DIR, PREVIEW_DIR, VECTOR_DIR]: Path(d).mkdir(parents=True, exist_ok=True)

def load_db() -> dict:
    _ensure_dirs()
    if not Path(DB_PATH).exists(): return {}
    try:
        with open(DB_PATH,"r",encoding="utf-8") as f: return json.load(f)
    except Exception as e: log.error(f"[VoiceDB] read: {e}"); return {}

def save_db(db):
    _ensure_dirs(); tmp = DB_PATH+".tmp"
    with open(tmp,"w",encoding="utf-8") as f: json.dump(db,f,ensure_ascii=False,indent=2)
    os.replace(tmp, DB_PATH)

def _normalize_wav(src, dst):
    try:
        with wave.open(src,"rb") as wf:
            n_ch,sw,sr,raw = wf.getnchannels(),wf.getsampwidth(),wf.getframerate(),wf.readframes(wf.getnframes())
        if sw==2:   a = np.frombuffer(raw,np.int16).astype(np.float32)/32768.
        elif sw==4: a = np.frombuffer(raw,np.int32).astype(np.float32)/2147483648.
        elif sw==1: a = (np.frombuffer(raw,np.uint8).astype(np.float32)-128)/128.
        else: return False
        if n_ch==2:  a = a.reshape(-1,2).mean(1)
        elif n_ch>2: a = a.reshape(-1,n_ch).mean(1)
        if sr!=TARGET_SR:
            x0=np.linspace(0,1,len(a)); x1=np.linspace(0,1,int(len(a)*TARGET_SR/sr))
            a = np.interp(x1,x0,a).astype(np.float32)
        a16=(np.clip(a,-1.,1.)*32767).astype(np.int16)
        with wave.open(dst,"wb") as wf:
            wf.setnchannels(1);wf.setsampwidth(2);wf.setframerate(TARGET_SR);wf.writeframes(a16.tobytes())
        return True
    except Exception as e: log.error(f"[VoiceDB] norm_wav: {e}"); return False

def compute_embedding(wav_path):
    emb = _extract_embedding_from_pipeline(wav_path)
    if emb is not None:
        return emb
    log.warning(f"[VoiceDB] Không thể trích xuất embedding cho file: {wav_path}")
    return None

def _mel_fb(nf,fs,sr):
    hfm=2595*np.log10(1+(sr/2)/700); mp=np.linspace(0,hfm,nf+2)
    hp=700*(10**(mp/2595)-1); bp=np.floor((fs+1)*hp/sr).astype(int)
    fb=np.zeros((nf,fs//2+1))
    for i in range(1,nf+1):
        for j in range(bp[i-1],bp[i]): fb[i-1,j]=(j-bp[i-1])/(bp[i]-bp[i-1]+1e-10)
        for j in range(bp[i],bp[i+1]): fb[i-1,j]=(bp[i+1]-j)/(bp[i+1]-bp[i]+1e-10)
    return fb

def _dct(x):
    n=x.shape[1]
    return x@np.cos(np.pi*np.arange(n)[:,None]*(2*np.arange(x.shape[1])[None,:]+1)/(2*n)).T

# ── Multi-file helpers ────────────────────────────────────────────────────────
def _get_paths(entry, key_list, key_single):
    v = entry.get(key_list) or entry.get(key_single)
    if v is None: return []
    if isinstance(v,str): return [v]
    return [p for p in v if p]

def get_preview_paths(entry): return _get_paths(entry,"preview_paths","preview_path")
def get_vector_paths(entry):  return _get_paths(entry,"vector_paths","vector_path")

def load_vectors_for_entry(entry) -> list:
    out=[]
    for p in get_vector_paths(entry):
        if p and os.path.exists(p):
            try: out.append(np.load(p))
            except Exception as e: log.warning(f"[VoiceDB] load vec {p}: {e}")
    return out

def _next_slot(vid, folder, ext):
    idx=0
    while True:
        p=os.path.join(folder,f"{vid}_{idx}{ext}")
        if not os.path.exists(p): return p,idx
        idx+=1

# ── CRUD ──────────────────────────────────────────────────────────────────────
def add_voiceid(name:str, wav_bytes:bytes, source:str="upload", group:str="Nhóm 1: Toạ đàm") -> tuple[bool,str]:
    """
    Thêm VoiceID.
    - Tên đã tồn tại → gộp vào ID cũ (không tạo ID mới).
    - Chưa có → tạo ID mới (lưu kèm group).
    """
    name=name.strip()
    if not name:      return False,"Tên không được để trống"
    if not wav_bytes: return False,"Không có dữ liệu âm thanh"
    _ensure_dirs(); db=load_db()

    existing = next((vid for vid,e in db.items() if e.get("name","").lower()==name.lower()), None)
    if existing:
        ok,msg = add_audio_to_voiceid(existing, wav_bytes, source=source)
        return (True,existing) if ok else (False,msg)

    vid=f"v_{uuid.uuid4().hex[:8]}"
    prev_p,_=_next_slot(vid,PREVIEW_DIR,".wav"); vec_p,_=_next_slot(vid,VECTOR_DIR,".npy")
    tmp=tempfile.mktemp(suffix=".wav")
    try:
        with open(tmp,"wb") as f: f.write(wav_bytes)
        if not _normalize_wav(tmp,prev_p): return False,"File WAV không hợp lệ"
        emb=compute_embedding(prev_p); has_v=False
        if emb is not None: np.save(vec_p,emb); has_v=True
        
        # BỔ SUNG KEY "group" VÀO ĐÂY
        entry={"id":vid,"name":name,"group":group,"source":source,"status":"active","has_vector":has_v,
               "preview_paths":[prev_p],"vector_paths":[vec_p] if has_v else [],
               "preview_path":prev_p,"vector_path":vec_p if has_v else None,
               "created_at":time.strftime("%Y-%m-%d %H:%M:%S"),
               "updated_at":time.strftime("%Y-%m-%d %H:%M:%S")}
        db[vid]=entry; save_db(db)
        log.info(f"[VoiceDB] Tạo mới: {vid} ({name})")
        return True,vid
    except Exception as e: log.error(f"[VoiceDB] add: {e}"); return False,str(e)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def add_audio_to_voiceid(vid:str, wav_bytes:bytes, source:str="upload") -> tuple[bool,str]:
    """Thêm 1 file audio vào ID đã có → append embedding mới."""
    if not wav_bytes: return False,"Không có dữ liệu âm thanh"
    db=load_db()
    if vid not in db: return False,f"Không tìm thấy VoiceID '{vid}'"
    entry=db[vid]; _ensure_dirs()
    prev_p,slot=_next_slot(vid,PREVIEW_DIR,".wav"); vec_p,_=_next_slot(vid,VECTOR_DIR,".npy")
    tmp=tempfile.mktemp(suffix=".wav")
    try:
        with open(tmp,"wb") as f: f.write(wav_bytes)
        if not _normalize_wav(tmp,prev_p): return False,"File WAV không hợp lệ"
        emb=compute_embedding(prev_p)
        pl=get_preview_paths(entry); vl=get_vector_paths(entry)
        pl.append(prev_p)
        if emb is not None: np.save(vec_p,emb); vl.append(vec_p); entry["has_vector"]=True
        entry.update({"preview_paths":pl,"vector_paths":vl,"preview_path":pl[0],
                      "vector_path":vl[0] if vl else None,
                      "updated_at":time.strftime("%Y-%m-%d %H:%M:%S")})
        db[vid]=entry; save_db(db)
        name=entry.get("name",vid)
        log.info(f"[VoiceDB] +audio #{slot} → {vid} ({name}), tổng {len(pl)} file")
        return True,f"Đã thêm audio #{slot+1} cho '{name}' (tổng {len(pl)} file)"
    except Exception as e: log.error(f"[VoiceDB] add_audio: {e}"); return False,str(e)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def delete_voiceid(vid:str) -> tuple[bool,str]:
    db=load_db()
    if vid not in db: return False,f"Không tìm thấy '{vid}'"
    entry=db[vid]
    for p in get_preview_paths(entry)+get_vector_paths(entry):
        if p and os.path.exists(p):
            try: os.unlink(p)
            except OSError as e: log.warning(f"[VoiceDB] xoá {p}: {e}")
    del db[vid]; save_db(db)
    return True,f"Đã xoá '{entry.get('name',vid)}'"


def update_voiceid(vid:str, **fields) -> tuple[bool,str]:
    db=load_db()
    if vid not in db: return False,f"Không tìm thấy '{vid}'"
    entry=db[vid]
    if "name" in fields:
        n=fields["name"].strip()
        if not n: return False,"Tên không được để trống"
        entry["name"]=n
    if "status" in fields:
        if fields["status"] not in ("active","inactive"): return False,"status không hợp lệ"
        entry["status"]=fields["status"]
    # Bổ sung lưu nhóm (group)
    if "group" in fields:
        entry["group"] = fields["group"]
        
    entry["updated_at"]=time.strftime("%Y-%m-%d %H:%M:%S")
    db[vid]=entry; save_db(db)
    return True,"Cập nhật thành công"


def get_all_entries() -> list:
    return sorted(load_db().values(), key=lambda e:e.get("created_at",""), reverse=True)

def get_entry(vid:str): return load_db().get(vid)

def get_db_stats() -> dict:
    db=load_db(); vals=list(db.values())
    return {"total":len(vals),
            "active":sum(1 for v in vals if v.get("status")=="active"),
            "has_vector":sum(1 for v in vals if v.get("has_vector")),
            "total_files":sum(len(get_preview_paths(v)) for v in vals)}

# ── Quản lý danh sách NHÓM (Groups) ──────────────────────────────────────────
GROUPS_PATH     = os.path.join(ENROLL_DIR, "voiceid_groups.json")
DEFAULT_GROUPS  = ["Nhóm 1: Toạ đàm", "Nhóm 2: Podcast", "Nhóm 3: Meeting"]

def get_groups() -> list[str]:
    _ensure_dirs()
    if not Path(GROUPS_PATH).exists():
        save_groups(DEFAULT_GROUPS)
        return list(DEFAULT_GROUPS)
    try:
        with open(GROUPS_PATH, "r", encoding="utf-8") as f:
            groups = json.load(f)
        return groups if groups else list(DEFAULT_GROUPS)
    except Exception:
        return list(DEFAULT_GROUPS)

def save_groups(groups: list[str]) -> None:
    _ensure_dirs()
    with open(GROUPS_PATH, "w", encoding="utf-8") as f:
        json.dump(groups, f, ensure_ascii=False, indent=2)

def add_group(name: str) -> tuple[bool, str]:
    name = (name or "").strip()
    if not name:
        return False, "Tên nhóm không được để trống"
    groups = get_groups()
    if name in groups:
        return False, "Nhóm này đã tồn tại"
    groups.append(name)
    save_groups(groups)
    return True, f"Đã tạo nhóm '{name}'"