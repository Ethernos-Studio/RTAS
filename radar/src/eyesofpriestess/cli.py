"""``radar`` 命令行：抽帧 -> 标注 -> 出框 -> 训练 -> 预测，一条链路到底。

    uv run radar extract   --src raw/1-7.mp4            # 视频 -> frames/ + manifest
    uv run radar ls-tasks                               # frames -> tasks/ls_tasks.json
    uv run radar ls-interface                           # configs/enemies.json -> interface.xml
    uv run radar import-ls --export ls_export.json      # LS 导出 -> clicks/*.jsonl
    uv run radar build                                  # clicks -> SAM 框 -> YOLO 数据集
    uv run radar train --epochs 50                      # 冒烟训练
    uv run radar export --fmt onnx
    uv run radar predict --weights runs/arknights/v0/weights/best.pt --src 战斗.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .catalog import load_catalog
from .dataset import BuildOptions, build
from .frames import extract, load_manifest
from .ls import build_tasks, convert
from .paths import Layout
from .schema import load_clicks


def _utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover
            pass


# -- 子命令 ---------------------------------------------------------------
def cmd_extract(args: argparse.Namespace) -> int:
    layout = Layout.discover(args.root)
    layout.ensure()
    src = Path(args.src) if args.src else layout.raw
    if not src.exists():
        print(f"错误: 找不到录像 {src}（把 mp4 放进 raw/ 或传 --src）", file=sys.stderr)
        return 1
    entries = extract(
        src,
        Path(args.out) if args.out else layout.frames,
        max_frames=args.max_frames,
        diff_thresh=args.diff_thresh,
        hash_max=args.hash_max,
        sample_fps=args.sample_fps,
        min_gap=args.min_gap,
        jpg_quality=args.jpg_quality,
        min_brightness=args.min_brightness,
        min_edges=args.min_edges,
        hwaccel=args.hwaccel,
        jobs=args.jobs,
        append=not args.rebuild,
    )
    manifest = load_manifest((Path(args.out) if args.out else layout.frames) / "manifest.jsonl")
    print(f"新增 {len(entries)} 帧；manifest 共 {len(manifest)} 帧")
    print(f"帧目录: {Path(args.out) if args.out else layout.frames}")
    return 0


def cmd_ls_serve(args: argparse.Namespace) -> int:
    """用本地文件存储启动 Label Studio（HANDOFF §6.1：别走 Cloud 转存）。"""
    import os
    import shutil
    import subprocess

    layout = Layout.discover(args.root)
    exe = shutil.which("label-studio")
    if not exe:
        print(
            "错误: 没装 label-studio。先跑:\n"
            "  uv tool install label-studio\n"
            "或 pip install label-studio",
            file=sys.stderr,
        )
        return 1

    env = os.environ.copy()
    env["LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED"] = "true"
    env["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"] = str(layout.root)
    env.setdefault("LABEL_STUDIO_BASE_DATA_DIR", str(layout.root / "ls_data"))
    # 走内网穿透/域名访问时，Django 会校验 Origin：把公网地址加进白名单，否则登录 POST 被 403
    origins = [origin.strip() for origin in (args.public_url or "").split(",") if origin.strip()]
    if origins:
        existing = [item for item in env.get("CSRF_TRUSTED_ORIGINS", "").split(",") if item]
        env["CSRF_TRUSTED_ORIGINS"] = ",".join(dict.fromkeys(existing + origins))

    # 注意：不要给 `label-studio start` 传项目名——LS 1.2x 会当成"启动已存在的项目"，
    # 项目还没建时会直接退出（Project "..." not found）。项目在网页里 Create 即可。
    command = [exe, "start", "--port", str(args.port), "--no-browser"]

    print("Label Studio 启动参数：")
    print(f"  LOCAL_FILES_DOCUMENT_ROOT = {env['LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT']}")
    print(f"  BASE_DATA_DIR             = {env['LABEL_STUDIO_BASE_DATA_DIR']}")
    if origins:
        print(f"  CSRF_TRUSTED_ORIGINS      = {env['CSRF_TRUSTED_ORIGINS']}")
    print(f"  标注界面                  = {layout.root / 'labelstudio' / 'interface.xml'}")
    print(f"  任务文件                  = {layout.tasks / 'ls_tasks.json'}")
    print(f"  账号                      = {args.email or '（首个注册的账号成为管理员）'}")
    print(f"  打开 http://127.0.0.1:{args.port}\n")

    if args.background:
        import time
        import urllib.error
        import urllib.request

        log_path = layout.root / "ls_serve.log"
        log_handle = log_path.open("a", encoding="utf-8", errors="replace")
        process = subprocess.Popen(command, env=env, stdout=log_handle,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        url = f"http://127.0.0.1:{args.port}/"
        deadline = time.time() + args.wait
        while time.time() < deadline:
            if process.poll() is not None:
                print(f"错误: Label Studio 退出了（码 {process.returncode}），日志 {log_path}",
                      file=sys.stderr)
                return 1
            try:
                with urllib.request.urlopen(url, timeout=3) as response:
                    if response.status == 200:
                        print(f"已启动: {url}（PID {process.pid}，日志 {log_path}）")
                        return 0
            except (urllib.error.URLError, OSError):
                time.sleep(2)
        print(f"错误: {args.wait}s 内没起来，看日志 {log_path}", file=sys.stderr)
        return 1

    try:
        return int(subprocess.call(command, env=env))
    except KeyboardInterrupt:  # pragma: no cover - Ctrl+C 退出
        return 0


def _ls_login(base: str, email: str, password: str):
    """登录 Label Studio，返回 (session, headers)；失败返回 (None, None)。"""
    import requests

    session = requests.Session()
    page = session.get(f"{base}/user/login/", timeout=20)
    if page.status_code != 200:
        print(f"错误: 打不开 {base}/user/login/（HTTP {page.status_code}）", file=sys.stderr)
        return None, None
    csrf = session.cookies.get("csrftoken", "")
    session.post(
        f"{base}/user/login/",
        data={"email": email, "password": password},
        headers={"Referer": f"{base}/user/login/", "X-CSRFToken": csrf},
        timeout=20,
    )
    if "sessionid" not in session.cookies:
        print("错误: 登录失败（邮箱或密码不对？）", file=sys.stderr)
        return None, None
    return session, {"X-CSRFToken": csrf, "Referer": base}


def _load_rules(layout: Layout, given: str | None) -> str:
    """读标注规矩（会作为「Instructions」显示在 LS 里）。"""
    path = Path(given) if given else layout.root / "labelstudio" / "labeling_rules.md"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _inline_html(text: str) -> str:
    import html
    import re

    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    return text


def rules_to_html(markdown: str) -> str:
    """把标注规矩的 markdown 子集转成 HTML。

    LS 的 Instructions 面板是按 **HTML** 渲染的（不认 markdown），直接塞原文就会变成
    截图里那种一坨 `#`/`**`。这里只支持我们实际用到的语法：# 标题、- 列表、**粗体**、`代码`。
    """
    import re

    blocks: list[str] = []
    in_list = False
    for raw in markdown.splitlines():
        line = raw.rstrip()
        if not line.strip():
            if in_list:
                blocks.append("</ul>")
                in_list = False
            continue

        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            if in_list:
                blocks.append("</ul>")
                in_list = False
            level = min(6, len(heading.group(1)) + 1)      # # -> h2，## -> h3
            blocks.append(f"<h{level}>{_inline_html(heading.group(2))}</h{level}>")
            continue

        bullet = re.match(r"^\s*[-*]\s+(.*)$", line)
        if bullet:
            if not in_list:
                blocks.append("<ul>")
                in_list = True
            blocks.append(f"<li>{_inline_html(bullet.group(1))}</li>")
            continue

        if in_list:
            blocks.append("</ul>")
            in_list = False
        blocks.append(f"<p>{_inline_html(line)}</p>")

    if in_list:
        blocks.append("</ul>")
    return "\n".join(blocks)


def cmd_add_code(args: argparse.Namespace) -> int:
    """往类别表里加一个图鉴编号（自动挑空热键、重新生成界面，可选推送到 LS）。"""
    layout = Layout.discover(args.root)
    catalog_path = Path(args.catalog) if args.catalog else layout.root / "configs" / "enemies.json"
    if not catalog_path.exists():
        print(f"错误: 找不到 {catalog_path}", file=sys.stderr)
        return 1
    data = json.loads(catalog_path.read_text(encoding="utf-8"))
    enemies = data.setdefault("enemies", {})

    code = args.code.strip()
    if not code:
        print("错误: --code 不能为空", file=sys.stderr)
        return 1
    if code in enemies or code in (data.get("special") or {}):
        print(f"错误: 编号 {code} 已存在（{enemies.get(code) or data['special'].get(code)}）",
              file=sys.stderr)
        return 1

    used = {str(item.get("hotkey")) for item in enemies.values() if item.get("hotkey")}
    used |= {str(item.get("hotkey")) for item in (data.get("special") or {}).values()
             if item.get("hotkey")}
    used |= {str(item.get("hotkey")) for item in (data.get("ignore") or {}).values()
             if item.get("hotkey")}
    hotkey = args.hotkey or next(
        (key for key in "oeruyipmnc" if key not in used), ""
    )
    if hotkey and hotkey in used:
        print(f"警告: 热键 {hotkey} 已被占用，这个标签只能鼠标点", file=sys.stderr)

    enemies[code] = {
        "name": args.name or "",
        "tier": args.tier,
        "cls": args.cls,
        "hotkey": hotkey,
    }
    catalog_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"已加入 {code}（{args.name or '未命名'}，{args.cls}，热键 {hotkey or '-'}）-> {catalog_path}")

    # 重新生成标注界面
    from .catalog import Catalog

    catalog = Catalog(data)
    interface = layout.root / "labelstudio" / "interface.xml"
    interface.write_text(catalog.interface_xml(), encoding="utf-8")
    print(f"标注界面已更新: {interface}")

    if args.push and args.email and args.password:
        base = (args.url or "http://127.0.0.1:8080").rstrip("/")
        session, headers = _ls_login(base, args.email, args.password)
        if session is None:
            return 1
        listing = session.get(f"{base}/api/projects", timeout=20).json()
        projects = listing.get("results") if isinstance(listing, dict) else listing
        project = next((item for item in projects or [] if item.get("title") == args.project), None)
        if not project:
            print(f"错误: 找不到项目 {args.project}", file=sys.stderr)
            return 1
        response = session.patch(
            f"{base}/api/projects/{project['id']}",
            json={"label_config": catalog.interface_xml()},
            headers=headers, timeout=30,
        )
        if response.status_code >= 300:
            print(f"错误: 推送失败 HTTP {response.status_code}: {response.text[:200]}",
                  file=sys.stderr)
            return 1
        print(f"已推送到 LS 项目 #{project['id']}「{args.project}」，刷新标注页生效")
    else:
        print("（要同步到 LS：加 --push --email ... --password ...）")
    return 0


def cmd_ls_sync_instruction(args: argparse.Namespace) -> int:
    """把 labeling_rules.md 推到 LS 项目的 Instructions 面板。"""
    layout = Layout.discover(args.root)
    base = (args.url or "http://127.0.0.1:8080").rstrip("/")
    session, headers = _ls_login(base, args.email, args.password)
    if session is None:
        return 1

    rules = _load_rules(layout, args.rules)
    if not rules:
        print("错误: 找不到标注规矩文件（labelstudio/labeling_rules.md）", file=sys.stderr)
        return 1

    listing = session.get(f"{base}/api/projects", timeout=20).json()
    projects = listing.get("results") if isinstance(listing, dict) else listing
    project = next(
        (item for item in projects or []
         if (args.project_id and item.get("id") == args.project_id)
         or (not args.project_id and item.get("title") == args.project)),
        None,
    )
    if not project:
        print(f"错误: 找不到项目 {args.project_id or args.project}", file=sys.stderr)
        return 1

    updated = session.patch(
        f"{base}/api/projects/{project['id']}",
        json={"expert_instruction": rules_to_html(rules), "show_instruction": True},
        headers=headers, timeout=30,
    )
    if updated.status_code >= 300:
        print(f"错误: 更新失败 HTTP {updated.status_code}: {updated.text[:200]}", file=sys.stderr)
        return 1
    print(f"已把标注规矩写入项目 #{project['id']}「{project.get('title')}」"
          f"（{len(rules)} 字，标注页右上「Instructions」可见）")
    return 0


def cmd_ls_init(args: argparse.Namespace) -> int:
    """登录 LS → 建项目 → 贴标注界面 → 挂本地存储 → 导入任务（一次到位）。"""
    layout = Layout.discover(args.root)
    base = (args.url or "http://127.0.0.1:8080").rstrip("/")
    session, headers = _ls_login(base, args.email, args.password)
    if session is None:
        return 1
    print(f"登录成功: {args.email} @ {base}")
    rules = _load_rules(layout, args.rules)

    interface = Path(args.interface) if args.interface else layout.root / "labelstudio" / "interface.xml"
    tasks_path = Path(args.tasks) if args.tasks else layout.tasks / "ls_tasks.json"
    if not interface.exists():
        print(f"错误: 找不到标注界面 {interface}（先跑 `radar ls-interface`）", file=sys.stderr)
        return 1
    if not tasks_path.exists():
        print(f"错误: 找不到任务文件 {tasks_path}（先跑 `radar ls-tasks`）", file=sys.stderr)
        return 1
    label_config = interface.read_text(encoding="utf-8")
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))

    def task_count(pid: int) -> int:
        detail = session.get(f"{base}/api/projects/{pid}", timeout=20).json()
        if isinstance(detail, dict) and "task_number" in detail:
            return int(detail["task_number"] or 0)
        payload = session.get(f"{base}/api/projects/{pid}/tasks?page_size=500", timeout=20).json()
        if isinstance(payload, dict):
            return int(payload.get("count") or 0)
        return len(payload or [])

    # 已经有同名项目就复用，避免重复建
    listing = session.get(f"{base}/api/projects", timeout=20).json()
    projects = listing.get("results") if isinstance(listing, dict) else listing
    project = next((item for item in projects or [] if item.get("title") == args.project), None)
    # 标注规矩一起推进 Instructions 面板（没这个文件就跳过）
    extra = {"expert_instruction": rules_to_html(rules), "show_instruction": True} if rules else {}
    if project:
        project_id = project["id"]
        session.patch(f"{base}/api/projects/{project_id}",
                      json={"label_config": label_config, "show_skip_button": True, **extra},
                      headers=headers, timeout=30)
        print(f"复用已有项目 #{project_id}「{args.project}」，已更新标注界面"
              + ("+标注规矩" if rules else ""))
    else:
        created = session.post(
            f"{base}/api/projects/",
            json={"title": args.project, "label_config": label_config,
                  "description": "EYESOFPRIESTESS：图鉴编号点选。先按热键选标签再点目标，ignore 只点坐标。",
                  **extra},
            headers=headers, timeout=30,
        )
        if created.status_code >= 300:
            print(f"错误: 建项目失败 HTTP {created.status_code}: {created.text[:200]}",
                  file=sys.stderr)
            return 1
        project_id = created.json()["id"]
        print(f"已建项目 #{project_id}「{args.project}」")

    existing_tasks = task_count(project_id)

    # LS 1.23 起，光有 LOCAL_FILES_SERVING_ENABLED 不够：/data/local-files/ 还会查
    # LocalFilesImportStorage（Cloud Storage → Local files），没有这条记录一律 404。
    # 这里给项目挂一条指向 frames/ 的本地存储，图片才能加载。
    frames_dir = str((Path(args.storage_path) if args.storage_path else layout.frames).resolve())
    storages = session.get(f"{base}/api/storages/localfiles?project={project_id}",
                           timeout=20).json()
    storages = storages.get("results") if isinstance(storages, dict) else storages
    known_paths = {(item.get("path") or "").rstrip("\\/").lower() for item in storages or []}
    if frames_dir.rstrip("\\/").lower() not in known_paths:
        created_storage = session.post(
            f"{base}/api/storages/localfiles",
            json={
                "project": project_id,
                "title": "frames",
                "path": frames_dir,
                "regex_filter": r".*[.]jpe?g$",
                "use_blob_urls": False,
                "recursive_scan": False,
                "presign": False,
            },
            headers=headers, timeout=30,
        )
        if created_storage.status_code >= 300:
            print(f"警告: 本地存储连接创建失败 HTTP {created_storage.status_code}: "
                  f"{created_storage.text[:200]}（图片会加载不出来）", file=sys.stderr)
        else:
            print(f"已挂本地存储连接: {frames_dir}")
    else:
        print(f"本地存储连接已存在: {frames_dir}")
    if existing_tasks and not args.force:
        # 已有任务时**只补新增的帧**（按 frame 去重），别整体跳过：新录的视频要能增量加进来
        existing_frames: set[str] = set()
        page = 1
        while True:
            chunk = session.get(
                f"{base}/api/projects/{project_id}/tasks?page={page}&page_size=500",
                timeout=60,
            ).json()
            items = chunk.get("results") if isinstance(chunk, dict) else chunk
            if not items:
                break
            for item in items:
                frame = str((item.get("data") or {}).get("frame") or "")
                if frame:
                    existing_frames.add(frame)
            if isinstance(chunk, dict) and not chunk.get("next"):
                break
            page += 1

        pending = [task for task in tasks
                   if str((task.get("data") or {}).get("frame") or "") not in existing_frames]
        if not pending:
            print(f"项目里已有 {existing_tasks} 个任务，没有新增帧，跳过导入")
            print(f"\n打开标注入口: {base}/projects/{project_id}/data")
            return 0
        print(f"项目里已有 {existing_tasks} 个任务；补导入 {len(pending)} 个新帧")
        tasks = pending

    imported = session.post(
        f"{base}/api/projects/{project_id}/import",
        data=json.dumps(tasks, ensure_ascii=False).encode("utf-8"),
        headers={**headers, "Content-Type": "application/json"},
        timeout=120,
    )
    if imported.status_code >= 300:
        print(f"错误: 导入任务失败 HTTP {imported.status_code}: {imported.text[:300]}",
              file=sys.stderr)
        return 1
    summary = imported.json() if imported.text else {}
    print(f"导入任务: {summary.get('task_count', len(tasks))} 条"
          f"（fails={summary.get('fails', 0)}）")

    print(f"\n打开标注入口: {base}/projects/{project_id}/data")
    return 0


def cmd_ls_rename_label(args: argparse.Namespace) -> int:
    """把已有标注里的某个标签批量改成另一个（可限定视频）。"""
    layout = Layout.discover(args.root)
    base = (args.url or "http://127.0.0.1:8080").rstrip("/")
    session, headers = _ls_login(base, args.email, args.password)
    if session is None:
        return 1

    listing = session.get(f"{base}/api/projects", timeout=20).json()
    projects = listing.get("results") if isinstance(listing, dict) else listing
    project = next((item for item in projects or [] if item.get("title") == args.project), None)
    if not project:
        print(f"错误: 找不到项目 {args.project}", file=sys.stderr)
        return 1

    from .frames import load_manifest

    manifest = load_manifest(layout.manifest)
    src, dst = args.src.strip(), args.dst.strip()
    if not src or not dst or src == dst:
        print("错误: --src/--dst 不能相同或为空", file=sys.stderr)
        return 1

    tasks = session.get(f"{base}/api/projects/{project['id']}/tasks?page_size=500",
                        timeout=60).json()
    tasks = tasks.get("results") if isinstance(tasks, dict) else tasks

    changed = 0
    touched = 0
    per_video: dict[str, int] = {}
    for task in tasks or []:
        frame = str((task.get("data") or {}).get("frame") or "")
        entry = manifest.get(frame)
        video = entry.video if entry else "?"
        if args.video and video != args.video:
            continue
        for annotation in task.get("annotations") or []:
            results = annotation.get("result") or []
            hit = 0
            for item in results:
                value = item.get("value") or {}
                key = "keypointlabels" if "keypointlabels" in value else "labels"
                labels = value.get(key) or []
                if src in [str(label) for label in labels]:
                    value[key] = [dst if str(label) == src else label for label in labels]
                    item["value"] = value
                    hit += 1
            if not hit:
                continue
            changed += hit
            touched += 1
            per_video[video] = per_video.get(video, 0) + hit
            if args.dry_run:
                continue
            response = session.patch(
                f"{base}/api/annotations/{annotation['id']}",
                json={"result": results}, headers=headers, timeout=30,
            )
            if response.status_code >= 300:
                print(f"警告: 更新标注 #{annotation['id']} 失败 HTTP {response.status_code}",
                      file=sys.stderr)

    verb = "将改" if args.dry_run else "已改"
    scope = f"（仅 {args.video}）" if args.video else ""
    print(f"{verb} {changed} 个点 {src} -> {dst}{scope}，涉及 {touched} 份标注")
    if per_video:
        print("按视频:", per_video)
    return 0


def cmd_ls_drop_labels(args: argparse.Namespace) -> int:
    """把某些标签的点从已有标注里删掉（例如已废弃的血条标签）。"""
    layout = Layout.discover(args.root)
    base = (args.url or "http://127.0.0.1:8080").rstrip("/")
    session, headers = _ls_login(base, args.email, args.password)
    if session is None:
        return 1

    listing = session.get(f"{base}/api/projects", timeout=20).json()
    projects = listing.get("results") if isinstance(listing, dict) else listing
    project = next((item for item in projects or [] if item.get("title") == args.project), None)
    if not project:
        print(f"错误: 找不到项目 {args.project}", file=sys.stderr)
        return 1

    drop = {name.strip() for name in args.labels.split(",") if name.strip()} if args.labels else set()
    if not drop and not args.dry_run:
        print("错误: --labels 不能为空（例如 hp_bar_red,hp_bar_ally）", file=sys.stderr)
        return 1

    tasks = session.get(f"{base}/api/projects/{project['id']}/tasks?page_size=500",
                        timeout=60).json()
    tasks = tasks.get("results") if isinstance(tasks, dict) else tasks

    removed = 0
    touched_annotations = 0
    for task in tasks or []:
        for annotation in task.get("annotations") or []:
            results = annotation.get("result") or []
            kept = []
            for item in results:
                value = item.get("value") or {}
                labels = value.get("keypointlabels") or value.get("labels") or []
                if any(str(label) in drop for label in labels):
                    removed += 1
                    continue
                kept.append(item)
            if len(kept) == len(results):
                continue
            touched_annotations += 1
            if args.dry_run:
                continue
            response = session.patch(
                f"{base}/api/annotations/{annotation['id']}",
                json={"result": kept}, headers=headers, timeout=30,
            )
            if response.status_code >= 300:
                print(f"警告: 更新标注 #{annotation['id']} 失败 HTTP {response.status_code}",
                      file=sys.stderr)

    verb = "将删除" if args.dry_run else "已删除"
    print(f"{verb} {removed} 个点（涉及 {touched_annotations} 份标注），标签: {sorted(drop)}")
    return 0


def cmd_ls_export(args: argparse.Namespace) -> int:
    """用 API 把标注导出成 LS 的 JSON，省得去网页点 Export。"""
    layout = Layout.discover(args.root)
    base = (args.url or "http://127.0.0.1:8080").rstrip("/")
    session, headers = _ls_login(base, args.email, args.password)
    if session is None:
        return 1

    listing = session.get(f"{base}/api/projects", timeout=20).json()
    projects = listing.get("results") if isinstance(listing, dict) else listing
    project = next(
        (item for item in projects or []
         if (args.project_id and item.get("id") == args.project_id)
         or (not args.project_id and item.get("title") == args.projects)),
        None,
    )
    if not project:
        print(f"错误: 找不到项目 {args.project_id or args.projects}", file=sys.stderr)
        return 1

    response = session.get(
        f"{base}/api/projects/{project['id']}/export",
        params={"exportType": "JSON", "download_all_tasks": "false"},
        timeout=120,
    )
    if response.status_code >= 300:
        print(f"错误: 导出失败 HTTP {response.status_code}: {response.text[:200]}", file=sys.stderr)
        return 1
    out = Path(args.out) if args.out else layout.root / "ls_export.json"
    out.write_text(response.text, encoding="utf-8")
    try:
        count = len(response.json())
    except ValueError:
        count = -1
    print(f"已导出 {count} 条任务 -> {out}")
    return 0


def cmd_ls_tasks(args: argparse.Namespace) -> int:
    layout = Layout.discover(args.root)
    frames_dir = Path(args.frames) if args.frames else layout.frames
    manifest = load_manifest(Path(args.manifest) if args.manifest else layout.manifest)
    if not manifest:
        print("错误: manifest 是空的，先跑 `radar extract`", file=sys.stderr)
        return 1
    tasks = build_tasks(
        manifest,
        frames_dir=frames_dir,
        local_prefix=args.local_prefix or "frames",
        absolute=args.absolute,
    )
    out = Path(args.out) if args.out else layout.tasks / "ls_tasks.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"任务数 {len(tasks)} -> {out}")
    if not args.absolute:
        print("LS 需要本地存储（HANDOFF §6.1：别用 Cloud 转存）：")
        print("  LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true")
        print(f"  LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT={layout.root}")
        print(f"  任务里的 d= 前缀是 {args.local_prefix or 'frames'}/（相对上面这个根目录）")
    return 0


def cmd_ls_interface(args: argparse.Namespace) -> int:
    layout = Layout.discover(args.root)
    catalog = load_catalog(args.catalog)
    out = Path(args.out) if args.out else layout.root / "labelstudio" / "interface.xml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(catalog.interface_xml(), encoding="utf-8")
    print(f"标注界面 -> {out}")
    print(catalog.cheat_sheet())
    return 0


def cmd_import_ls(args: argparse.Namespace) -> int:
    layout = Layout.discover(args.root)
    catalog = load_catalog(args.catalog)
    stats = convert(
        Path(args.export),
        Path(args.manifest) if args.manifest else layout.manifest,
        Path(args.out) if args.out else layout.clicks,
        catalog=catalog,
        operator=args.operator,
    )
    print(
        f"转换 {stats['clicks']} 个点，覆盖 {len(stats['frames'])} 帧；"
        f"跳过 {stats['skip_task']} 个任务"
    )
    if stats["unknown_labels"]:
        preview = ", ".join(f"{k}×{v}" for k, v in sorted(stats["unknown_labels"].items())[:8])
        print(f"注意: 有不在 catalog 里的标签（默认按 enemy 处理）: {preview}",
              file=sys.stderr)
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    layout = Layout.discover(args.root)
    catalog = load_catalog(args.catalog)

    from .sam import SamBoxer  # 延迟导入：没装 ultralytics 时其它子命令照样可用

    options = BuildOptions(
        val_ratio=args.val_ratio,
        seed=args.seed,
        val_videos=tuple(args.val_videos) if getattr(args, "val_videos", None) else None,
        backgrounds=tuple(getattr(args, "backgrounds", []) or ()),
        min_box_area=args.min_box_area,
        region=tuple(float(v) for v in args.region.split(",")) if args.region else None,
        downscale=args.downscale,
        crop_classes=tuple(args.crops.split(",")) if args.crops else ("enemy",),
    )
    report = build(
        clicks=Path(args.clicks) if args.clicks else layout.clicks,
        frames=Path(args.frames) if args.frames else layout.frames,
        manifest_path=Path(args.manifest) if args.manifest else layout.manifest,
        out=Path(args.out) if args.out else layout.dataset,
        catalog=catalog,
        boxer=SamBoxer(_sam_weights(layout, args.sam), device=args.device),
        options=options,
    )
    data = report.to_dict()
    print(json.dumps({k: data[k] for k in ("points", "boxes", "missing", "ignore",
                                           "box_ratio", "ignore_rate", "val_videos")},
                     ensure_ascii=False, indent=2))
    print(f"class_hist: {data['class_hist']}")
    for warning in data["warnings"]:
        print(f"[warn] {warning}", file=sys.stderr)
    out_dir = Path(args.out) if args.out else layout.dataset
    print(f"数据集 -> {out_dir}（报告 {out_dir / 'report.json'}）")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .train import train

    layout = Layout.discover(args.root)
    dataset_dir = Path(args.data) if args.data else layout.dataset
    model = train(
        dataset_dir,
        weights=_train_weights(layout, args.weights),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        freeze=args.freeze,
        patience=args.patience,
        project=Path(args.project) if args.project else None,
        name=args.name,
        device=args.device,
        workers=getattr(args, "workers", 2),
    )
    save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
    print(f"训练完成: {save_dir}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from .train import export

    layout = Layout.discover(args.root)
    weights = Path(args.weights) if args.weights else _default_weights(layout, args.name)
    path = export(weights, fmt=args.fmt, imgsz=args.imgsz, half=args.half)
    print(f"导出: {path}")
    return 0


def cmd_val(args: argparse.Namespace) -> int:
    from .train import validate

    layout = Layout.discover(args.root)
    weights = Path(args.weights) if args.weights else _default_weights(layout, args.name)
    summary = validate(
        weights,
        Path(args.data) if args.data else layout.dataset,
        imgsz=args.imgsz,
        split=args.split,
        device=args.device,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def cmd_embed_train(args: argparse.Namespace) -> int:
    """训练敌人身份的 embedding（ArcFace），按视频划分训练/验证。"""
    from .catalog import load_catalog
    from .embed import TrainConfig, collect_crops, split_by_video, train_embedder
    from .frames import load_manifest

    layout = Layout.discover(args.root)
    catalog = load_catalog(Path(args.catalog) if args.catalog else layout.root / "configs" / "enemies.json")
    codes = [code for code, label in catalog.labels.items() if label.cls in ("enemy", "ally")]

    items = collect_crops(layout.dataset / "crops", codes)
    if not items:
        print("错误: dataset/crops 里没有可用的裁图（先跑 radar build）", file=sys.stderr)
        return 1

    manifest = load_manifest(layout.manifest)
    val_videos = set(args.validate_videos or [])
    if not val_videos:
        # 默认拿最新那段录像当验证集：按 manifest 里的视频顺序取最后一个
        videos = []
        for entry in manifest.values():
            if entry.video not in videos:
                videos.append(entry.video)
        val_videos = {videos[-1]} if videos else set()

    train_items, val_items = split_by_video(items, val_videos)
    print(f"训练裁图 {len(train_items)} 张 / 验证裁图 {len(val_items)} 张（验证视频: {', '.join(sorted(val_videos))}）")

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        margin=args.margin,
        reference_dir=Path(args.reference) if args.reference else None,
    )
    out = Path(args.out) if args.out else layout.weights / "embedder.pt"
    report = train_embedder(train_items, val_items, codes, out, config)

    print(json.dumps({k: v for k, v in report.items() if k != "history"}, ensure_ascii=False, indent=2))
    print(f"embedding 权重 -> {out}")
    return 0


def cmd_gallery_build(args: argparse.Namespace) -> int:
    """从 dataset/crops/<编号>/ 建身份图库（embedding 原型）。"""
    from .identity import Encoder, Gallery

    layout = Layout.discover(args.root)
    crops_dir = Path(args.crops) if args.crops else layout.dataset / "crops"
    if not crops_dir.exists():
        print(f"错误: 找不到裁剪图目录 {crops_dir}（先跑 `radar build`）", file=sys.stderr)
        return 1

    encoder = Encoder(args.arch, device=args.device,
                      checkpoint=Path(args.embedder) if getattr(args, 'embedder', None) else None)
    keep = None
    if not args.include_all:
        from .catalog import load_catalog

        catalog = load_catalog(args.catalog)
        keep = [code for code in catalog.labels if catalog.cls_for(code) == "enemy"]
    gallery = Gallery.from_crops(
        crops_dir, encoder,
        max_per_code=args.max_per_code,
        min_crops=args.min_crops,
        prototypes_per_code=args.prototypes,
        keep_codes=keep,
        reference_dir=Path(args.reference) if args.reference else None,
    )
    out = Path(args.out) if args.out else layout.weights / "gallery.npz"
    gallery.save(out)

    print(f"图库 {len(gallery.codes)} 类 / {sum(gallery.samples.values())} 张 -> {out}")
    print(gallery.summary())
    print(f"编码器: {args.arch} ({encoder.device_name})")
    return 0


def cmd_identity_check(args: argparse.Namespace) -> int:
    """按真实管线量身份准确率：检测框裁图 -> 图库匹配 -> 对比人工编号。"""
    import cv2

    from .catalog import load_catalog
    from .frames import load_manifest
    from .identity import Encoder, Gallery, crop_box
    from .schema import load_clicks

    layout = Layout.discover(args.root)
    catalog = load_catalog(args.catalog)
    gallery = Gallery.load(Path(args.gallery) if args.gallery else layout.weights / "gallery.npz")
    encoder = Encoder(args.arch, device=args.device,
                      checkpoint=Path(args.embedder) if getattr(args, 'embedder', None) else None)

    if args.detector:
        from ultralytics import YOLO

        detector = YOLO(args.detector)
    else:
        detector = None

    manifest = load_manifest(layout.manifest)
    clicks = load_clicks(Path(args.clicks) if args.clicks else layout.clicks)

    def iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
        union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return 0.0 if union <= 0 else inter / union

    correct = 0
    total = 0
    ambiguous = 0
    unknown = 0
    per_code: dict[str, list[int]] = {}
    confusion: dict[tuple[str, str], int] = {}

    for frame, items in sorted(clicks.items()):
        entry = manifest.get(frame)
        if entry is None or (args.video and entry.video != args.video):
            continue
        image = cv2.imread(str(layout.frames / frame))
        if image is None:
            continue
        height, width = image.shape[:2]

        truth = [
            (click.cls or "", [click.x - max(8, width * 0.015), click.y - max(8, height * 0.03),
                               click.x + max(8, width * 0.015), click.y + max(8, height * 0.03)])
            for click in items
            if not click.is_ignore and catalog.cls_for(click.cls or "") == "enemy"
        ]
        if not truth:
            continue

        if detector is not None:
            result = detector.predict(image, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            boxes = []
            for box in result.boxes or []:
                if int(box.cls[0]) != 0:          # 只给 enemy 做身份
                    continue
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                boxes.append([x1, y1, x2 - x1, y2 - y1])
        else:
            boxes = [[t[1][0], t[1][1], t[1][2] - t[1][0], t[1][3] - t[1][1]] for t in truth]

        if not boxes:
            for expected, _box in truth:
                bucket = per_code.setdefault(expected, [0, 0])
                bucket[1] += 1
            continue

        crops = [crop_box(image, box, pad=args.pad) for box in boxes]
        matches = gallery.identify(encoder.embed(crops))

        for expected, gt_box in truth:
            # 找与这个 GT 点重叠最大的检测框，用它的身份结果
            best_index, best_iou = -1, 0.25
            for index, box in enumerate(boxes):
                x, y, w, h = box
                value = iou([gt_box[0], gt_box[1], gt_box[2], gt_box[3]],
                            [x, y, x + w, y + h])
                if value > best_iou:
                    best_index, best_iou = index, value
            # 只统计"检测到了并且能配上"的点：检测漏掉的属于检测器的问题，
            # 不该算进身份匹配的准确率
            if best_index < 0:
                continue
            total += 1
            bucket = per_code.setdefault(expected, [0, 0])
            bucket[1] += 1
            match = matches[best_index]
            if match.ambiguous:
                ambiguous += 1
            if not match.code:
                unknown += 1
            if match.code == expected:
                correct += 1
                bucket[0] += 1
            else:
                key = (expected, match.code or "?")
                confusion[key] = confusion.get(key, 0) + 1

    if not total:
        print("错误: 没有可评估的点（检查 clicks / --video / 检测器）", file=sys.stderr)
        return 1
    print(f"身份匹配 top-1: {correct}/{total} = {correct / total:.1%}")
    print(f"  认不出(低于阈值): {unknown}   top1/top2 接近: {ambiguous}")
    for code, (hit, count) in sorted(per_code.items(), key=lambda item: -item[1][1]):
        print(f"  {code:<4} {hit}/{count}")
    if confusion:
        worst = sorted(confusion.items(), key=lambda item: -item[1])[:6]
        print("  主要混淆:", "、".join(f"{a}→{b}×{n}" for (a, b), n in worst))
    return 0


def cmd_describe(args: argparse.Namespace) -> int:
    """把 scene.jsonl 翻成 VLM 能读的文字（数量 + 身份 + **坐标** + 方位）。

    给了 ``--src`` + ``--out-dir`` 时，还会把每条描述对应那一帧的**图片**一起存盘：
    ``0001.jpg`` + ``0001.txt``，外加 ``index.jsonl``。VLM 的输入就是"图 + 这段文字"，
    文字里带 center=(x,y)，它才能把"图里那个家伙"和坐标对上。
    """
    import cv2

    from .describe import Describer, load_scene

    scene = Path(args.scene)
    if not scene.exists():
        print(f"错误: 找不到场景图 {scene}（先跑 `radar predict`）", file=sys.stderr)
        return 1

    records = load_scene(scene)
    layout = Layout.discover(args.root)
    catalog_path = Path(args.catalog) if args.catalog else layout.root / "configs" / "enemies.json"
    catalog = load_catalog(catalog_path)
    names = {code: label.name for code, label in catalog.labels.items() if label.name}
    describer = Describer(width=args.width, height=args.height, names=names)
    buckets = _bucketize(records, describer, args.every)

    lines: list[str] = []
    for timestamp, units in buckets:
        lines.append(describer.line(timestamp, describer.summarize(units)))
        lines.extend(describer.unit_lines(units))

    if args.out:
        Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"描述 {len(buckets)} 段 -> {args.out}")

    if args.out_dir and args.src:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        wanted = {int(record["frame_idx"]): index for index, record in enumerate(records)}
        by_index = {}
        cap = cv2.VideoCapture(args.src) if not Path(args.src).is_dir() else None
        for index, record in enumerate(records, 1):
            frame_index = int(record.get("frame_idx", index))
            if cap is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index - 1)
                    ok, frame = cap.read()
                if ok:
                    by_index[frame_index] = frame
            else:
                path = sorted(Path(args.src).glob("*.jpg"))[frame_index - 1]
                image = cv2.imread(str(path))
                if image is not None:
                    by_index[frame_index] = image
        if cap is not None:
            cap.release()

        index_lines = []
        for number, (timestamp, units) in enumerate(buckets, 1):
            text = describer.line(timestamp, describer.summarize(units))
            unit_text = describer.unit_lines(units)
            frame_index = _bucket_frame_index(records, describer, args.every, number)
            frame = by_index.get(frame_index)
            if frame is not None:
                # 关键：图上也必须画出编号，VLM（和人）才能把文字里的 #3 和图上那个框对上
                frame = _annotate(frame, units)
                image_path = out_dir / f"{number:04d}.jpg"
                cv2.imwrite(str(image_path), frame)
            else:
                image_path = None
            text_path = out_dir / f"{number:04d}.txt"
            text_path.write_text("\n".join([text] + unit_text) + "\n", encoding="utf-8")
            index_lines.append(json.dumps({
                "t": round(timestamp, 2),
                "image": image_path.name if image_path else None,
                "text": text,
                "units": len(units),
            }, ensure_ascii=False))
        (out_dir / "index.jsonl").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
        print(f"图文对 {len(buckets)} 组 -> {out_dir}")

    for line in lines[: args.show]:
        print(line)
    return 0


def _annotate(frame, units):
    """在图上画出框 + 编号，编号与文字里的 ``#id`` 一致。

    VLM 没法把 "center=(989,240)" 和图上某个像素对上（人也做不到），
    所以文字里的每个 #id 必须在图上有个一眼能认出来的标记。
    """
    import cv2

    canvas = frame.copy()
    occupied: list[tuple[int, int, int, int]] = []
    for unit in units:
        if unit.get("cls") != "enemy":
            continue
        x, y, w, h = (int(v) for v in unit["box"])
        track = unit.get("track_id")
        code = unit.get("code") or "enemy"
        label = f"#{track} {code}"
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (80, 230, 80), 2)

        # 两个框挨着时标签会叠在一起，往上错开
        label_w = 12 * len(label)
        label_y = y - 6 if y > 24 else y + h + 18
        for _attempt in range(6):
            rect = (x, label_y - 14, x + label_w, label_y + 4)
            clash = any(not (rect[2] < other[0] or rect[0] > other[2] or
                             rect[3] < other[1] or rect[1] > other[3]) for other in occupied)
            if not clash:
                break
            label_y -= 22
        occupied.append((x, label_y - 14, x + label_w, label_y + 4))

        cv2.putText(canvas, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (80, 230, 80), 2, cv2.LINE_AA)
    return canvas


def _bucketize(records, describer, every: float):
    """按 every 秒分桶，每桶取"帧数接近中位数"的那一帧。"""
    buckets = []
    current: list[dict] = []
    bucket_end = every
    for record in records:
        timestamp = float(record.get("t", 0.0))
        if timestamp > bucket_end:
            if current:
                buckets.append(current)
            current = []
            while timestamp > bucket_end:
                bucket_end += every
        current.append(record)
    if current:
        buckets.append(current)

    picked = []
    for items in buckets:
        counts = sorted(len([u for u in item.get("units") or [] if u.get("cls") == "enemy"])
                        for item in items)
        median = counts[len(counts) // 2]
        chosen = min(items, key=lambda item: abs(
            len([u for u in item.get("units") or [] if u.get("cls") == "enemy"]) - median))
        picked.append((float(chosen.get("t", 0.0)), chosen.get("units") or []))
    return picked


def _bucket_frame_index(records, describer, every: float, number: int) -> int:
    """第 number 个桶对应的帧号（和 _bucketize 的选帧逻辑一致）。"""
    buckets = []
    current: list[dict] = []
    bucket_end = every
    for record in records:
        timestamp = float(record.get("t", 0.0))
        if timestamp > bucket_end:
            if current:
                buckets.append(current)
            current = []
            while timestamp > bucket_end:
                bucket_end += every
        current.append(record)
    if current:
        buckets.append(current)
    items = buckets[number - 1]
    counts = sorted(len([u for u in item.get("units") or [] if u.get("cls") == "enemy"])
                    for item in items)
    median = counts[len(counts) // 2]
    chosen = min(items, key=lambda item: abs(
        len([u for u in item.get("units") or [] if u.get("cls") == "enemy"]) - median))
    return int(chosen.get("frame_idx", 0))


def cmd_predict(args: argparse.Namespace) -> int:
    from .predict import predict_track

    layout = Layout.discover(args.root)
    weights = Path(args.weights) if args.weights else _default_weights(layout, args.name)
    out = Path(args.out) if args.out else layout.root / "scene.jsonl"
    stats = predict_track(
        weights=weights,
        source=Path(args.src),
        out=out,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        gallery=Path(args.gallery) if args.gallery else None,
        start=args.start,
        seconds=args.seconds,
    )
    print(f"场景图: {stats['frames']} 帧 / {stats['units']} 个目标 -> {stats['out']}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    layout = Layout.discover(args.root)
    rows: list[tuple[str, str, bool]] = []

    rows.append(("python", sys.version.split()[0], True))
    try:
        import cv2

        rows.append(("opencv", cv2.__version__, True))
    except ImportError:  # pragma: no cover
        rows.append(("opencv", "缺失", False))
    try:
        import numpy

        rows.append(("numpy", numpy.__version__, True))
    except ImportError:  # pragma: no cover
        rows.append(("numpy", "缺失", False))
    try:
        import ultralytics

        rows.append(("ultralytics", ultralytics.__version__, True))
    except ImportError:
        rows.append(("ultralytics", "未安装（build/train/predict 用不了）", False))
    try:
        import torch

        cuda = "CUDA 可用" if torch.cuda.is_available() else "仅 CPU"
        rows.append(("torch", f"{torch.__version__} / {cuda}", torch.cuda.is_available()))
    except ImportError:
        rows.append(("torch", "未安装（跟随 ultralytics）", False))
    try:
        import label_studio  # noqa: F401

        rows.append(("label-studio", "已安装", True))
    except ImportError:
        rows.append(("label-studio", "未安装（标注时才需要）", False))

    manifest = load_manifest(layout.manifest)
    clicks = load_clicks(layout.clicks) if layout.clicks.exists() else {}
    dataset_dir = layout.dataset
    rows.append(("frames", f"{len(manifest)} 帧 @ {layout.frames}", bool(manifest)))
    rows.append(("clicks", f"{sum(len(v) for v in clicks.values())} 点 / {len(clicks)} 帧",
                 bool(clicks)))
    rows.append(("dataset", "已生成" if (dataset_dir / "data.yaml").exists() else "未生成",
                 (dataset_dir / "data.yaml").exists()))
    runs = sorted(layout.runs.glob("arknights/*/weights/best.pt"))
    rows.append(("weights", str(runs[-1]) if runs else "还没有训练产物", bool(runs)))

    width = max(len(name) for name, _, _ in rows)
    for name, value, ok in rows:
        print(f"  {'OK ' if ok else '-- '} {name:<{width}}  {value}")
    return 0


def _default_weights(layout: Layout, name: str) -> Path:
    """按名字找权重；``name`` 为空或 "latest" 时取最近训练的那一版。"""
    runs_dir = layout.runs / "arknights"
    if not name or name == "latest":
        candidates = sorted(
            (path for path in runs_dir.glob("*/weights/best.pt") if path.exists()),
            key=lambda path: path.stat().st_mtime,
        )
        if not candidates:
            raise SystemExit(f"找不到任何训练产物（{runs_dir}）；先 `radar train` 或传 --weights")
        return candidates[-1]
    candidate = runs_dir / name / "weights" / "best.pt"
    if not candidate.exists():
        raise SystemExit(f"找不到权重 {candidate}；先 `radar train` 或传 --weights")
    return candidate


def _sam_weights(layout: Layout, given: str | None) -> str:
    """SAM 权重：优先用 ``weights/mobile_sam.pt``（离线放好即可），否则交给 ultralytics 自动下载。"""
    if given:
        return given
    local = layout.weights / "mobile_sam.pt"
    if local.exists():
        return str(local)
    return "mobile_sam.pt"


def _train_weights(layout: Layout, given: str | None) -> str:
    """预训练权重：``weights/yolov8s.pt`` 存在就用它，省得每次联网拉。"""
    if given:
        return given
    local = layout.weights / "yolov8s.pt"
    if local.exists():
        return str(local)
    return "yolov8s.pt"


# -- 解析器 ---------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="radar",
        description="EYESOFPRIESTESS：视频 -> 标注 -> 数据集 -> 训练/预测",
    )
    parser.add_argument("--root", help="项目根目录（默认自动向上找 pyproject.toml）")
    parser.add_argument("--catalog", help="标签目录 configs/enemies.json")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("extract", help="视频 -> 可标注帧 + manifest.jsonl")
    p.add_argument("--src", help="视频文件或目录（默认 raw/）")
    p.add_argument("--out", help="帧目录（默认 frames/）")
    p.add_argument("--max-frames", type=int, default=400, help="每个视频最多抽多少帧")
    p.add_argument("--diff-thresh", type=float, default=10.0, help="帧差阈值，越小越爱抽")
    p.add_argument("--hash-max", type=int, default=6, help="感知哈希距离阈值")
    p.add_argument("--sample-fps", type=float, default=3.0,
                   help="候选帧率（交给 ffmpeg 的 fps 滤镜；越大越慢越细，默认 3）")
    p.add_argument("--min-gap", type=float, default=0.33,
                   help="相邻标注帧最小间隔（秒），防特效密集段刷屏")
    p.add_argument("--hwaccel", action="store_true", help="让 ffmpeg 用 GPU 解码（NVDEC）")
    p.add_argument("--jobs", type=int, default=None,
                   help="并行抽几个视频（默认最多 4 个；1 = 串行）")
    p.add_argument("--jpg-quality", type=int, default=95)
    p.add_argument("--min-brightness", type=float, default=5.0,
                   help="整帧平均亮度低于此值视为黑屏/加载画面，不抽")
    p.add_argument("--min-edges", type=float, default=0.015,
                   help="边缘密度低于此值视为加载页/编队页（没有战场），不抽；0 = 关闭")
    p.add_argument("--rebuild", action="store_true", help="清空 manifest 重新抽")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("ls-serve", help="启动 Label Studio（本地文件存储，自动配好环境变量）")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--background", action="store_true",
                   help="后台启动并等它就绪（推荐：不占终端，日志写 ls_serve.log）")
    p.add_argument("--wait", type=int, default=120, help="后台模式最多等多少秒")
    p.add_argument("--public-url", default="",
                   help="公网访问地址（逗号分隔），加进 CSRF 白名单，例如 http://1.2.3.4:8000")
    p.add_argument("--email", default="", help="仅用于打印提示，不参与启动")
    p.set_defaults(func=cmd_ls_serve)

    p = sub.add_parser("ls-init", help="登录 LS 并建项目/贴界面/导任务（一次到位）")
    p.add_argument("--url", default="http://127.0.0.1:8080", help="LS 地址")
    p.add_argument("--email", required=True)
    p.add_argument("--password", required=True)
    p.add_argument("--project", default="eyes-of-priestess")
    p.add_argument("--interface", help="标注界面 XML（默认 labelstudio/interface.xml）")
    p.add_argument("--tasks", help="任务 JSON（默认 tasks/ls_tasks.json）")
    p.add_argument("--rules", help="标注规矩 md（默认 labelstudio/labeling_rules.md）")
    p.add_argument("--storage-path", help="本地存储连接的路径（默认 frames/）")
    p.add_argument("--force", action="store_true", help="项目已有任务时也再导入一遍")
    p.set_defaults(func=cmd_ls_init)

    p = sub.add_parser("ls-rename-label", help="把已有标注里的某个标签批量改名（可限定视频）")
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--email", required=True)
    p.add_argument("--password", required=True)
    p.add_argument("--project", default="RTAS model")
    p.add_argument("--src", required=True, help="原标签，例如 02")
    p.add_argument("--dst", required=True, help="新标签，例如 01")
    p.add_argument("--video", help="只改这个视频的帧（例如 v002_1-9）")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_ls_rename_label)

    p = sub.add_parser("ls-drop-labels", help="从已有标注里删掉指定标签的点（如废弃的血条）")
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--email", required=True)
    p.add_argument("--password", required=True)
    p.add_argument("--project", default="RTAS model")
    p.add_argument("--labels", help="逗号分隔，例如 hp_bar_red,hp_bar_ally")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_ls_drop_labels)

    p = sub.add_parser("ls-export", help="用 API 导出标注 JSON（等价于网页 Export）")
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--email", required=True)
    p.add_argument("--password", required=True)
    p.add_argument("--projects", default="RTAS model")
    p.add_argument("--project-id", type=int)
    p.add_argument("--out", help="输出路径（默认 ls_export.json）")
    p.set_defaults(func=cmd_ls_export)

    p = sub.add_parser("add-code", help="加一个图鉴编号到类别表（可选推送 LS）")
    p.add_argument("--code", required=True, help="图鉴里显示的编号，例如 01 / B3 / CS")
    p.add_argument("--name", default="", help="中文名（只做注释）")
    p.add_argument("--tier", default="normal", choices=["normal", "elite", "leader"])
    p.add_argument("--cls", default="enemy", help="落到哪个粗类（enemy/ally/tile）")
    p.add_argument("--hotkey", default="", help="热键（不填自动挑一个空的）")
    p.add_argument("--push", action="store_true", help="顺便推送到 Label Studio")
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--email", default="")
    p.add_argument("--password", default="")
    p.add_argument("--project", default="RTAS model")
    p.set_defaults(func=cmd_add_code)

    p = sub.add_parser("ls-sync-instruction", help="把标注规矩推到 LS 的 Instructions 面板")
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--email", required=True)
    p.add_argument("--password", required=True)
    p.add_argument("--project", default="RTAS model")
    p.add_argument("--project-id", type=int)
    p.add_argument("--rules", help="标注规矩 md（默认 labelstudio/labeling_rules.md）")
    p.set_defaults(func=cmd_ls_sync_instruction)

    p = sub.add_parser("ls-tasks", help="生成 Label Studio 导入用 tasks.json")
    p.add_argument("--manifest")
    p.add_argument("--frames", help="帧目录（默认 frames/）")
    p.add_argument("--out")
    p.add_argument("--local-prefix", help="DOCUMENT_ROOT 下到帧目录的相对路径（默认 frames）")
    p.add_argument("--absolute", action="store_true", help="写绝对路径而不是 local-files URL")
    p.set_defaults(func=cmd_ls_tasks)

    p = sub.add_parser("ls-interface", help="由 configs/enemies.json 生成标注界面 XML")
    p.add_argument("--out")
    p.set_defaults(func=cmd_ls_interface)

    p = sub.add_parser("import-ls", help="Label Studio 导出 JSON -> clicks/*.jsonl")
    p.add_argument("--export", required=True, help="LS 导出的 JSON 文件")
    p.add_argument("--manifest")
    p.add_argument("--out", help="clicks 输出目录（默认 clicks/）")
    p.add_argument("--operator", default="", help="覆盖标注人名字")
    p.set_defaults(func=cmd_import_ls)

    p = sub.add_parser("build", help="clicks -> SAM 出框 -> YOLO 数据集")
    p.add_argument("--clicks")
    p.add_argument("--frames")
    p.add_argument("--manifest")
    p.add_argument("--out", help="数据集目录（默认 dataset/）")
    p.add_argument("--sam", help="SAM 权重（默认 weights/mobile_sam.pt，没有就让 ultralytics 下）")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--backgrounds", nargs="*", default=[],
                   help="没有敌人的帧（目录或图片），空标签进训练集当负样本")
    p.add_argument("--val-videos", nargs="*",
                   help="固定这些视频当验证集（例如 v002_1-9），保证多次训练指标可比")
    p.add_argument("--min-box-area", type=int, default=20)
    p.add_argument("--region", help="只在战场区域跑 SAM: x0,y0,x1,y1（0~1 比例）")
    p.add_argument("--downscale", type=float, default=1.0, help="SAM 前降采样，0.5 提速")
    p.add_argument("--crops", help="要给哪些类裁身份图库素材（默认 enemy）")
    p.add_argument("--device", help="torch device（cuda:0 / cpu）")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("embed-train", help="训练敌人身份的 embedding（ArcFace）")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--margin", type=float, default=0.3)
    p.add_argument("--out", help="输出权重（默认 weights/embedder.pt）")
    p.add_argument("--embedder", help="训练好的 embedding 权重（默认 weights/embedder.pt）")
    p.add_argument("--reference", default="reference/enemies", help="官方渲染图目录（当类锚点）")
    p.add_argument("--validate-videos", nargs="*", help="验证集用的视频（默认最新一段）")
    p.set_defaults(func=cmd_embed_train)

    p = sub.add_parser("gallery-build", help="从 dataset/crops 建身份图库（embedding 原型）")
    p.add_argument("--crops", help="裁剪图目录（默认 dataset/crops）")
    p.add_argument("--out", help="输出 npz（默认 weights/gallery.npz）")
    p.add_argument("--arch", default="resnet18", choices=["resnet18", "convnext_tiny"])
    p.add_argument("--device")
    p.add_argument("--max-per-code", type=int, default=240)
    p.add_argument("--min-crops", type=int, default=5)
    p.add_argument("--prototypes", type=int, default=4, help="每类保留几个原型（覆盖动作/差分）")
    p.add_argument("--embedder", help="训练好的 embedding 权重（默认 weights/embedder.pt）")
    p.add_argument("--reference", default="reference/enemies",
                   help="官方渲染参考图目录（每个编号一张 png，来自 tools/render_reference.py）")
    p.add_argument("--include-all", action="store_true",
                   help="连非 enemy 类也放进图库（默认只放图鉴编号对应的敌人）")
    p.set_defaults(func=cmd_gallery_build)

    p = sub.add_parser("identity-check", help="量身份匹配准确率（检测框裁图 -> 图库匹配）")
    p.add_argument("--gallery")
    p.add_argument("--clicks")
    p.add_argument("--video", help="只评估这个视频的帧")
    p.add_argument("--embedder", help="训练好的 embedding 权重（默认 weights/embedder.pt）")
    p.add_argument("--detector", help="用哪个检测器出框（默认用标注点附近的框）")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--pad", type=float, default=0.0, help="裁剪外扩比例（要和建库时一致）")
    p.add_argument("--arch", default="resnet18", choices=["resnet18", "convnext_tiny"])
    p.add_argument("--device")
    p.set_defaults(func=cmd_identity_check)

    p = sub.add_parser("train", help="训练 YOLO（imgsz=1280 / freeze=10）")
    p.add_argument("--data", help="数据集目录（默认 dataset/）")
    p.add_argument("--weights", help="初始权重（默认 weights/yolov8s.pt，没有就让 ultralytics 下）")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--freeze", type=int, default=10)
    p.add_argument("--patience", type=int, default=60)
    p.add_argument("--project")
    p.add_argument("--name", default="latest", help="用哪一版训练产物（默认最新）")
    p.add_argument("--device")
    p.add_argument("--workers", type=int, default=2,
                   help="dataloader 进程数（Windows 上默认 2，多了容易 OOM）")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("export", help="导出 ONNX / TensorRT")
    p.add_argument("--weights")
    p.add_argument("--name", default="latest", help="用哪一版训练产物（默认最新）")
    p.add_argument("--fmt", default="onnx", choices=["onnx", "engine", "torchscript"])
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--half", action="store_true")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("val", help="在 val 上验证")
    p.add_argument("--weights")
    p.add_argument("--name", default="latest", help="用哪一版训练产物（默认最新）")
    p.add_argument("--data")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--split", default="val")
    p.add_argument("--device")
    p.set_defaults(func=cmd_val)

    p = sub.add_parser("describe", help="scene.jsonl -> VLM 可读的文字（带坐标，可选存图文对）")
    p.add_argument("--scene", default="scene.jsonl")
    p.add_argument("--src", help="原始视频/帧目录（给了就能把对应帧图片一起存出来）")
    p.add_argument("--out-dir", help="图文对输出目录（0001.jpg + 0001.txt + index.jsonl）")
    p.add_argument("--out", help="所有描述写进一个 txt（默认只打印）")
    p.add_argument("--every", type=float, default=1.0, help="每几秒一条描述")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--show", type=int, default=20)
    p.set_defaults(func=cmd_describe)

    p = sub.add_parser("predict", help="检测+跟踪 -> 场景图 JSONL")
    p.add_argument("--weights")
    p.add_argument("--name", default="latest", help="用哪一版训练产物（默认最新）")
    p.add_argument("--src", required=True, help="视频或帧目录")
    p.add_argument("--out", help="输出 JSONL（默认 scene.jsonl）")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.8,
                   help="NMS 阈值：0.7~0.8 才不会把挨着的两个敌人并成一个")
    p.add_argument("--gallery", default="weights/gallery.npz",
                   help="身份图库（给了就把图鉴编号写进 scene）")
    p.add_argument("--start", type=float, default=0.0, help="只跑视频的这一段（秒）")
    p.add_argument("--seconds", type=float, help="这段多长（秒）")
    p.add_argument("--device")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("doctor", help="环境/数据自检")
    p.set_defaults(func=cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    _utf8_console()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
