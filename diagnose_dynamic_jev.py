#!/usr/bin/env python3
"""Why is dynamic Jev not used? Run this, then give dynamic-jev-diagnosis.txt back.

  python diagnose_dynamic_jev.py --repo B:\\repos\\MyProjects\\_MyLlama\\moe-cache --bench B:\\repos\\MyProjects\\_MyLlama\\_LiveCodeBench

Options:
  --repo   the git folder you build (default: current folder)
  --build  the CMake build folder (default: REPO/build)
  --exe    the llama-server executable you start (default: found under the build folder)
  --url    the running server (default: http://127.0.0.1:8080)
  --bench  the benchmark folder (optional: shows what its client sends)
  --stats  only print GET /dynamic-jev (run it after a benchmark)

Python standard library only. It only reads files and talks to the server; it changes nothing.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

REPORT = []
PROBLEMS = []


def out(line=""):
    print(line)
    REPORT.append(line)


def result(kind, title, detail="", fix=""):
    out("[%s] %s" % (kind, title))
    if detail:
        for ln in str(detail).splitlines():
            out("       " + ln)
    if fix:
        out("       FIX: " + fix)
    if kind == "FAIL":
        PROBLEMS.append((title, fix))


def read(path, limit=4_000_000):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(limit)
    except OSError:
        return None


def run(cmd, timeout=60):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, errors="replace")
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except Exception as e:  # noqa: BLE001
        return -1, str(e)


def norm(p):
    return os.path.normcase(os.path.normpath(p.replace("\\", "/"))) if p else ""


def http(method, url, body=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode(errors="replace")
            try:
                j = json.loads(raw) if raw else None
            except ValueError:
                j = raw
            return r.status, r.headers, j
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        try:
            j = json.loads(raw)
        except ValueError:
            j = raw
        return e.code, e.headers, j
    except Exception as e:  # noqa: BLE001
        return None, None, str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.getcwd())
    ap.add_argument("--build", default=None)
    ap.add_argument("--exe", default=None)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--bench", default=None)
    ap.add_argument("--stats", action="store_true")
    a = ap.parse_args()
    url = a.url.rstrip("/")

    out("dynamic Jev diagnosis, %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    out("python %s on %s" % (sys.version.split()[0], sys.platform))

    if a.stats:
        st, _, j = http("GET", url + "/dynamic-jev")
        out("GET /dynamic-jev -> %s" % st)
        out(json.dumps(j, indent=2) if isinstance(j, (dict, list)) else str(j))
        return finish()

    repo = os.path.abspath(a.repo)
    build = os.path.abspath(a.build) if a.build else os.path.join(repo, "build")

    out("")
    out("== 1. The source folder")
    if not os.path.isdir(repo):
        result("FAIL", "repo folder not found: " + repo, fix="pass --repo with the folder you build")
        return finish()
    rc, v = run(["git", "-C", repo, "rev-parse", "--short", "HEAD"])
    out("repo      : " + repo)
    out("git HEAD  : " + (v.strip() if rc == 0 else "no git info (" + v.strip()[:80] + ")"))
    rc, v = run(["git", "-C", repo, "status", "--short"])
    out("git status: " + ("clean" if rc == 0 and not v.strip() else ("\n             ".join(v.strip().splitlines()[:12]) if rc == 0 else "unavailable")))

    feats = [
        ("the decision endpoint (/v1/decision)", "tools/server/server-context.cpp", "handle_decision"),
        ("confidence signals (patch 2)", "tools/parallel-decision/decision-engine.cpp", "logits_lse"),
        ("slot mode (patch 2)", "tools/parallel-decision/decision-engine.cpp", "on_base"),
        ("--dynamic-jev option (patch 3)", "common/arg.cpp", '"--dynamic-jev"'),
        ("chat auto-route (patch 3)", "tools/server/server-context.cpp", "dj_route"),
        ("GET /dynamic-jev route (patch 3)", "tools/server/server.cpp", "/dynamic-jev"),
    ]
    missing = []
    for name, rel, needle in feats:
        t = read(os.path.join(repo, rel.replace("/", os.sep)))
        ok = t is not None and needle in t
        out("  %-42s %s" % (name, "present" if ok else "MISSING"))
        if not ok:
            missing.append(name)
    if missing:
        result("FAIL", "the source folder lacks: " + ", ".join(missing),
               fix="apply the patches in order (jev-on-moe-cache-167742d, dynamic-core-on-jev, dynamic-jev-chat-route) or extract corrected-files.zip over this folder")
    else:
        result("PASS", "the source folder has the decision engine, signals, slot mode and the chat route")

    out("")
    out("== 2. The build folder")
    cache = read(os.path.join(build, "CMakeCache.txt"))
    if cache is None:
        result("FAIL", "no CMakeCache.txt in " + build, fix="configure first: cmake -S . -B build -DGGML_CUDA=ON")
    else:
        m = re.search(r"^CMAKE_HOME_DIRECTORY:INTERNAL=(.*)$", cache, re.M)
        home = m.group(1).strip() if m else ""
        out("CMake source folder recorded in the cache: " + home)
        if norm(home) != norm(repo):
            result("FAIL", "the build folder was configured from a DIFFERENT source folder",
                   "cache says: %s\nyou build  : %s\nEvery build in this folder compiles the OTHER tree, so none of your changes are in the binary." % (home, repo),
                   "delete the build folder and configure again: Remove-Item -Recurse -Force build ; cmake -S . -B build -G \"Visual Studio 17 2022\" -A x64 -DGGML_CUDA=ON ; cmake --build build --config Release --target llama-server -j 10")
        else:
            result("PASS", "the build folder was configured from this source folder")
        m = re.search(r"^GGML_CUDA:BOOL=(.*)$", cache, re.M)
        out("GGML_CUDA in the cache: " + (m.group(1) if m else "not set"))
        if m and m.group(1).strip().upper() not in ("ON", "TRUE", "1"):
            result("FAIL", "the build has no CUDA (CPU only)", fix="configure with -DGGML_CUDA=ON")

    out("")
    out("== 3. The executable")
    exe = a.exe
    if not exe:
        names = ["llama-server.exe", "llama-server"]
        for d in (os.path.join(build, "bin", "Release"), os.path.join(build, "bin")):
            for n in names:
                p = os.path.join(d, n)
                if os.path.isfile(p):
                    exe = p
                    break
            if exe:
                break
    if not exe or not os.path.isfile(exe):
        result("FAIL", "llama-server was not found under " + build, fix="build it: cmake --build build --config Release --target llama-server -j 10")
    else:
        mt = os.path.getmtime(exe)
        out("executable: %s" % exe)
        out("built     : %s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mt)))
        newest = (0, "")
        for rel in ["common/arg.cpp", "common/common.h", "tools/server/server-context.cpp", "tools/server/server.cpp", "tools/parallel-decision/decision-engine.cpp"]:
            p = os.path.join(repo, rel.replace("/", os.sep))
            if os.path.isfile(p) and os.path.getmtime(p) > newest[0]:
                newest = (os.path.getmtime(p), rel)
        if newest[0] > mt + 1:
            result("FAIL", "the executable is OLDER than %s (edited %s)" % (newest[1], time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(newest[0]))),
                   fix="rebuild: cmake --build build --config Release --target llama-server -j 10")
        else:
            result("PASS", "the executable is newer than the newest source file")
        rc, helptxt = run([exe, "--help"], timeout=90)
        has_dj = "--dynamic-jev" in helptxt
        has_ds = "--decision-seqs" in helptxt
        out("--help lists --decision-seqs: %s, --dynamic-jev: %s" % (has_ds, has_dj))
        if not has_dj:
            result("FAIL", "this executable does not know --dynamic-jev (it was not built from the patched source)",
                   fix="rebuild from the patched folder (section 2), then start THIS executable")
        else:
            result("PASS", "the executable knows --dynamic-jev")
        d = os.path.dirname(exe)
        dlls = sorted(x for x in os.listdir(d) if x.lower().endswith(".dll"))[:12] if os.path.isdir(d) else []
        if dlls:
            out("DLLs next to it: " + ", ".join(dlls))
            out("(an .exe must run next to the DLLs of the same build; do not copy it elsewhere)")

    out("")
    out("== 4. Which program do you actually start?")
    for name in ("llama-jev", "llama-jev.exe", "llama-server", "llama-server.exe"):
        w = shutil.which(name)
        if w:
            out("on PATH: %-18s -> %s" % (name, w))
            if exe and norm(os.path.dirname(w)) != norm(os.path.dirname(exe)):
                result("WARN", "%s on PATH is not the executable of this build" % name,
                       "if the benchmark is started with that name, it runs another build",
                       "start the server with the full path of the executable above")

    out("")
    out("== 5. Who listens on the server port")
    port = re.search(r":(\d+)\s*$", url)
    port = port.group(1) if port else "8080"
    pids = []
    if sys.platform.startswith("win"):
        rc, v = run(["netstat", "-ano", "-p", "tcp"])
        for ln in v.splitlines():
            if ":%s " % port in ln and "LISTENING" in ln.upper():
                pids.append(ln.split()[-1])
        for pid in sorted(set(pids)):
            rc, t = run(["powershell", "-NoProfile", "-Command", "(Get-Process -Id %s).Path" % pid])
            out("port %s: PID %s  %s" % (port, pid, t.strip()[:200]))
    else:
        rc, v = run(["ss", "-ltnp"])
        for ln in v.splitlines():
            if ":%s " % port in ln:
                out(ln.strip()[:200])
                pids.append(ln)
    if len(set(pids)) > 1:
        result("FAIL", "more than one program listens on port " + port + " (a stale server)", fix="close every llama-server / llama-jev process, then start one")
    elif not pids:
        out("no listener found on port %s (or the check is unavailable here)" % port)

    out("")
    out("== 6. The running server (%s)" % url)
    st, _, j = http("GET", url + "/health")
    out("GET /health -> %s %s" % (st, j if st != 200 else ""))
    server_ok = st == 200
    st, _, props = http("GET", url + "/props")
    if st == 200 and isinstance(props, dict):
        out("build_info: %s" % props.get("build_info"))
        out("model     : %s" % (props.get("model_alias") or props.get("model_path")))
        out("n_ctx     : %s" % (props.get("default_generation_settings", {}).get("n_ctx")))
    st, _, dj = http("GET", url + "/dynamic-jev")
    out("GET /dynamic-jev -> %s" % st)
    live_dj = False
    if st is None:
        result("FAIL", "the server does not answer at " + url, fix="start it, or pass --url with the right address and port")
    elif st == 404:
        result("FAIL", "this server has NO /dynamic-jev route: it is not the patched build",
               "the process answering on %s was not built from the patched source" % url,
               "stop it and start the executable of section 3 with --dynamic-jev")
    elif isinstance(dj, dict) and not dj.get("enabled"):
        result("FAIL", "the patched server is running WITHOUT --dynamic-jev", fix="add --dynamic-jev to the start command")
    elif isinstance(dj, dict):
        live_dj = True
        result("PASS", "dynamic Jev is enabled on the running server")
        out("rungs: %s" % ", ".join(r.get("name", "?") for r in dj.get("policy", {}).get("rungs", [])))
        out("override_client: %s" % dj.get("policy", {}).get("override_client"))
        out("stats: %s" % json.dumps(dj.get("stats")))

    out("")
    out("== 7. Probe requests")
    if server_ok:
        body = {"messages": [{"role": "user", "content": "Write a python function that returns the sum of a list."}], "max_tokens": 8, "temperature": 0}
        st, h, j = http("POST", url + "/v1/chat/completions", body, timeout=300)
        tag = h.get("X-Dynamic-Jev") if h is not None else None
        out("plain chat request -> HTTP %s, X-Dynamic-Jev: %s" % (st, tag))
        if live_dj and not tag:
            result("FAIL", "dynamic Jev is enabled but this chat request was not routed",
                   "see last_error below; common causes: no free KV (use -c 32768), the decision sequences, an empty user message",
                   "read GET /dynamic-jev -> stats.last_error")
            st2, _, dj2 = http("GET", url + "/dynamic-jev")
            if isinstance(dj2, dict):
                out("stats: %s" % json.dumps(dj2.get("stats")))
        elif tag:
            result("PASS", "the chat request was routed: " + tag)
        st, h, j = http("POST", url + "/v1/decision", {"schema": {"ok": {"type": "boolean", "description": "Is this a question?"}}, "contexts": ["Is the sky blue?"]}, timeout=300)
        out("POST /v1/decision -> HTTP %s" % st)
        if st == 400 and isinstance(j, dict):
            out("   " + str(j.get("error", j))[:200])
    else:
        out("skipped: the server is not up")

    if a.bench:
        out("")
        out("== 8. What the benchmark client sends (%s)" % a.bench)
        pats = [r"/chat/completions", r"/v1/decision", r"base_url", r"enable_thinking", r"chat_template_kwargs", r"reasoning", r"max_tokens", r"n_predict",
                r"temperature", r"llama-server", r"llama-jev", r"--port", r"8080", r"dynamic"]
        rx = re.compile("|".join(pats))
        hits = 0
        for root, dirs, files in os.walk(a.bench):
            dirs[:] = [d for d in dirs if d not in (".git", "node_modules", "venv", ".venv", "__pycache__", "data", "datasets")]
            for fn in files:
                if not fn.lower().endswith((".py", ".sh", ".ps1", ".cmd", ".bat", ".json", ".yaml", ".yml", ".toml", ".md", ".txt")):
                    continue
                p = os.path.join(root, fn)
                if os.path.getsize(p) > 600_000:
                    continue
                t = read(p)
                if not t:
                    continue
                for i, ln in enumerate(t.splitlines(), 1):
                    if rx.search(ln) and hits < 60:
                        out("  %s:%d: %s" % (os.path.relpath(p, a.bench), i, ln.strip()[:150]))
                        hits += 1
        if hits == 0:
            out("  nothing found (check the --bench path)")
        out("  If the client sets enable_thinking / max_tokens itself, the router's values lose (client_wins).")

    return finish()


def finish():
    out("")
    out("== Summary")
    if not PROBLEMS:
        out("no blocking problem found by these checks. If scores are still unchanged:")
        out("  1. run the benchmark, then: python diagnose_dynamic_jev.py --stats")
        out("     routed must be above 0 and failed_open near 0;")
        out("  2. the benchmark has about 32 problems, so one problem is 3.1 points: repeat each setting 3 times before comparing.")
    else:
        out("%d blocking problem(s), in the order to fix them:" % len(PROBLEMS))
        for i, (t, f) in enumerate(PROBLEMS, 1):
            out("  %d. %s" % (i, t))
            if f:
                out("     -> " + f)
    try:
        with open("dynamic-jev-diagnosis.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(REPORT) + "\n")
        print("\nreport written to dynamic-jev-diagnosis.txt")
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
