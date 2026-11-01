"""
validate_gt.py
===============

Kiem tra 1 file ground-truth co khop 'gt_document.schema.json' khong,
va tu dong cap nhat 'manifest.json' (danh sach tong hop moi tai lieu GT).

Cach dung:
    python validate_gt.py <duong_dan_gt.json> [--doc-type TYPE] [--table-style STYLE] [--scan-quality Q]

Vi du:
    python validate_gt.py benchmark_data/ground_truth/vanban1/gt.json \
        --doc-id vanban1 \
        --doc-type phu_luc_ky_thuat \
        --table-style bordered \
        --scan-quality clean
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

try:
    import jsonschema
except ImportError:
    jsonschema = None

SCHEMA_DIR = Path(__file__).resolve().parent
GT_SCHEMA_PATH = SCHEMA_DIR / "gt_document.schema.json"
MANIFEST_SCHEMA_PATH = SCHEMA_DIR / "manifest.schema.json"


def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def recompute_n_blocks(gt: dict) -> None:
    for page in gt.get("pages", []):
        page["n_blocks"] = len(page.get("blocks", []))


def check_all_verified(gt: dict) -> tuple[bool, list[str]]:
    problems = []
    for page in gt.get("pages", []):
        for b in page.get("blocks", []):
            if not b.get("verified", False):
                problems.append(
                    f"page {page.get('page')} order={b.get('order')} "
                    f"type={b.get('type')!r}: verified=False"
                )
    return (len(problems) == 0, problems)


def validate_schema(gt: dict, schema: dict) -> list[str]:
    if jsonschema is None:
        return ["[canh bao] khong co package 'jsonschema' -- bo qua kiem tra JSON Schema chi tiet, "
                "chi kiem tra thu cong cac field bat buoc. Cai bang: pip install jsonschema --break-system-packages"]
    validator = jsonschema.Draft7Validator(schema)
    errors = sorted(validator.iter_errors(gt), key=lambda e: list(e.path))
    return [f"{'/'.join(str(p) for p in e.path) or '(root)'}: {e.message}" for e in errors]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gt_path", type=Path)
    ap.add_argument("--doc-id", type=str, default=None, help="Ghi de/dien doc_id neu file GT chua co")
    ap.add_argument("--doc-type", type=str, default=None)
    ap.add_argument("--table-style", type=str, default=None,
                    choices=["bordered", "borderless", "mixed", "none"])
    ap.add_argument("--scan-quality", type=str, default=None,
                    choices=["clean", "medium", "noisy"])
    ap.add_argument("--annotated-by", type=str, default=None)
    ap.add_argument("--manifest", type=Path, default=SCHEMA_DIR / "manifest.json")
    args = ap.parse_args()

    gt = load_json(args.gt_path)

    # ── auto-fix nhung field thieu de tuong thich nguoc (vd file cu chua co doc_id) ──
    if "doc_id" not in gt:
        gt["doc_id"] = args.doc_id or args.gt_path.stem
        print(f"[tu-sua] them doc_id={gt['doc_id']!r} (thieu trong file goc)")
    elif args.doc_id and gt["doc_id"] != args.doc_id:
        print(f"[canh bao] doc_id trong file ({gt['doc_id']!r}) khac voi --doc-id truyen vao ({args.doc_id!r})")

    recompute_n_blocks(gt)
    args.gt_path.write_text(json.dumps(gt, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── kiem tra schema ──
    schema = load_json(GT_SCHEMA_PATH)
    errors = validate_schema(gt, schema)
    if errors:
        print(f"\n❌ {args.gt_path} KHONG khop schema:")
        for e in errors:
            print(f"   - {e}")
    else:
        print(f"✅ {args.gt_path} khop 'gt_document.schema.json'")

    # ── kiem tra tat ca block da verified chua ──
    all_verified, problems = check_all_verified(gt)
    if not all_verified:
        print(f"\n⚠️  {len(problems)} block CHUA verified -- KHONG dua vao benchmark chinh thuc:")
        for p in problems[:10]:
            print(f"   - {p}")
        if len(problems) > 10:
            print(f"   ... va {len(problems) - 10} block khac")
    else:
        print("✅ Tat ca block da verified=true")

    # ── cap nhat manifest.json ──
    manifest = {"schema_version": "1.0", "documents": []}
    if args.manifest.exists():
        manifest = load_json(args.manifest)

    n_pages = len(gt.get("pages", []))
    has_table = any(
        b.get("content_kind") == "table_html"
        for page in gt.get("pages", []) for b in page.get("blocks", [])
    )

    entry = {
        "doc_id": gt["doc_id"],
        "gt_path": str(args.gt_path),
        "source_pdf": gt.get("source_pdf", ""),
        "n_pages": n_pages,
        "has_table": has_table,
        "verified_complete": all_verified,
        "annotated_at": date.today().isoformat(),
    }
    if args.doc_type:
        entry["doc_type"] = args.doc_type
    if args.table_style:
        entry["table_style"] = args.table_style
    if args.scan_quality:
        entry["scan_quality"] = args.scan_quality
    if args.annotated_by:
        entry["annotated_by"] = args.annotated_by

    existing = {d["doc_id"]: i for i, d in enumerate(manifest["documents"])}
    if entry["doc_id"] in existing:
        manifest["documents"][existing[entry["doc_id"]]] = entry
        print(f"[manifest] cap nhat entry cho doc_id={entry['doc_id']!r}")
    else:
        manifest["documents"].append(entry)
        print(f"[manifest] them moi entry cho doc_id={entry['doc_id']!r}")

    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[manifest] da luu -> {args.manifest} ({len(manifest['documents'])} tai lieu)")

    return 0 if (not errors and all_verified) else 1


if __name__ == "__main__":
    sys.exit(main())
