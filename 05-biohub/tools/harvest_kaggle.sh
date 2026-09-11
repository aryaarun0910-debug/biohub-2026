#!/usr/bin/env bash
# Re-harvest live competition state into evidence/raw/, then re-ingest.
# No Kaggle credentials needed: these are the frontend's own RPC endpoints,
# which answer unauthenticated for public competition data.
#   ./tools/harvest_kaggle.sh
set -euo pipefail
COMP_ID=136605
FORUM_ID=10656304
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir "$ROOT/evidence/.harvest-lock" || { echo "another harvest is active" >&2; exit 1; }
STAGE=$(mktemp -d "$ROOT/evidence/.harvest.XXXXXX")
trap 'rm -rf "$STAGE"; rmdir "$ROOT/evidence/.harvest-lock"' EXIT
OUT="$STAGE/raw"
mkdir -p "$OUT"

rpc() {  # rpc <service/Method> <json-body> <outfile>
  curl --fail --silent --show-error --max-time 60 -X POST "https://www.kaggle.com/api/i/$1" \
    -H 'content-type: application/json' -H 'x-xsrf-token: 1' \
    -H 'user-agent: Mozilla/5.0' --data "$2" -o "$3"
  python3 - "$3" <<'PYJSON'
import json, sys
with open(sys.argv[1]) as f:
    payload = json.load(f)
if not isinstance(payload, dict) or 'error' in payload:
    raise SystemExit('invalid RPC response')
PYJSON
  printf '  %-58s %8s bytes\n' "$1" "$(wc -c < "$3")"
}

echo "harvesting competition $COMP_ID ..."
rpc "competitions.CompetitionService/GetCompetition" "{\"competitionId\":$COMP_ID}"      "$OUT/comp.json"
rpc "competitions.PageService/ListPages"             "{\"competitionId\":$COMP_ID}"      "$OUT/pages.json"
rpc "competitions.LeaderboardService/GetLeaderboard" "{\"competitionId\":$COMP_ID}"      "$OUT/lb.json"
for p in 1 2 3 4 5 6; do
  rpc "discussions.DiscussionsService/GetTopicListByForumId" "{\"forumId\":$FORUM_ID,\"page\":$p}" "$OUT/topics_p$p.json"
  rpc "kernels.KernelsService/ListKernels" \
      "{\"kernelFilterCriteria\":{\"listRequest\":{\"page\":$p,\"pageSize\":40,\"competitionId\":\"$COMP_ID\",\"sortBy\":\"VOTE_COUNT\"}}}" \
      "$OUT/kernels_p$p.json"
done

# split the pages blob back out to markdown
python3 - "$OUT" <<'PY'
import json,os,sys,re
out=sys.argv[1]; d=json.load(open(os.path.join(out,"pages.json")))
os.makedirs(os.path.join(out,"pages"),exist_ok=True)
if not d.get("pages"):
    raise SystemExit('empty pages response')
for name,key in [('comp.json','deadline'),('lb.json','publicLeaderboard')]:
    if not json.load(open(os.path.join(out,name))).get(key):
        raise SystemExit(f'missing required field: {name}/{key}')
for page in range(1,7):
    for prefix,key in [('topics','topics'),('kernels','kernels')]:
        payload=json.load(open(os.path.join(out,f'{prefix}_p{page}.json')))
        if not isinstance(payload.get(key),list):
            raise SystemExit(f'missing list: {prefix}/{key}')
for p in d.get("pages",[]):
    name = re.sub(r'[^A-Za-z0-9_-]', '_', p['name'])
    fn=os.path.join(out,"pages",f"{int(p['order'])}_{name}.md")
    open(fn,"w").write(p["content"])
print(f"  wrote {len(d.get('pages',[]))} pages")
PY

# Publish only a complete validated batch; failures above leave existing evidence untouched.
python3 - "$ROOT" "$STAGE" <<'PYPUBLISH'
import datetime, hashlib, json, os, pathlib, shutil, sys
root, stage = map(pathlib.Path, sys.argv[1:])
sys.path.insert(0, str(root / 'tools'))
from evidence import verify
at = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
with (stage / 'index.jsonl').open('w') as f:
    for p in sorted((stage / 'raw').rglob('*')):
        if p.is_file():
            f.write(json.dumps({'file':str(p.relative_to(stage)), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                                'method':'rpc', 'fetched_at':at}) + '\n')
verify(stage)
evidence = root / 'evidence'
backup = stage / 'previous'
backup.mkdir()
installed = []
saved = []
try:
    for name in ('raw', 'index.jsonl'):
        if (evidence / name).exists():
            os.replace(evidence / name, backup / name)
            saved.append(name)
        os.replace(stage / name, evidence / name)
        installed.append(name)
except BaseException:
    for name in reversed(installed):
        path = evidence / name
        shutil.rmtree(path) if path.is_dir() else path.unlink()
    for name in saved:
        os.replace(backup / name, evidence / name)
    raise
print('Published verified evidence acquired at', at)
PYPUBLISH
python3 "$ROOT/tools/ingest_kaggle.py"
