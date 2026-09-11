#!/usr/bin/env bash
# Re-harvest live competition state into evidence/raw/, then re-ingest.
# No Kaggle credentials needed: these are the frontend's own RPC endpoints,
# which answer unauthenticated for public competition data.
#   ./tools/harvest_kaggle.sh
set -euo pipefail
COMP_ID=136605
FORUM_ID=10656304
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/evidence/raw"
mkdir -p "$OUT"

rpc() {  # rpc <service/Method> <json-body> <outfile>
  curl -s --max-time 60 -X POST "https://www.kaggle.com/api/i/$1" \
    -H 'content-type: application/json' -H 'x-xsrf-token: 1' \
    -H 'user-agent: Mozilla/5.0' --data "$2" -o "$3"
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
import json,os,sys
out=sys.argv[1]; d=json.load(open(os.path.join(out,"pages.json")))
os.makedirs(os.path.join(out,"pages"),exist_ok=True)
for p in d.get("pages",[]):
    fn=os.path.join(out,"pages",f"{p['order']}_{p['name'].replace(' ','_')}.md")
    open(fn,"w").write(p["content"])
print(f"  wrote {len(d.get('pages',[]))} pages")
PY

# re-freeze by sha256
cd "$ROOT/evidence"; : > index.jsonl
for f in raw/*.json raw/pages/*.md; do
  [ -f "$f" ] || continue
  printf '{"file":"%s","sha256":"%s","method":"rpc","fetched_at":"%s"}\n' \
    "$f" "$(sha256sum "$f" | cut -d' ' -f1)" "$(date -u +%FT%TZ)" >> index.jsonl
done
echo "  froze $(wc -l < index.jsonl) artifacts"
python3 "$ROOT/tools/ingest_kaggle.py"
