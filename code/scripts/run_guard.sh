#!/usr/bin/env bash
# 无人值守跑实验的看门狗：GPU 掉卡自动重启恢复，训练进程死了自动续跑。
#
#   setsid bash scripts/run_guard.sh < /dev/null >> logs/guard.log 2>&1 &
#
# 行为：
#   1) 每轮先验 GPU：nvidia-smi 拿不到卡名 -> 视为掉卡
#        a. nvidia-smi -r 尝试重置
#        b. 仍不行 -> 记录 NEED_REBOOT 并 reboot（6 小时内最多 3 次，防死循环）
#           机器起来后由 crontab 的 @reboot 项把 guard 拉起来，从断点继续
#   2) GPU 正常 -> 跑 run_one_shot.sh（内部幂等：已完成的 run 跳过、有 last.pt 就 --resume）
#   3) run_one_shot 返回 2（冒烟判定一致性项没被优化）-> 停止，等人看
#   4) 所有 run 都产出 eval_eval.json -> 正常退出
set -uo pipefail
export PYTHONUNBUFFERED=1   # 重定向到日志时 python 默认块缓冲，进度会延迟几分钟才可见
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs
CAMP="${CAMP:-camp_v4}"
MANIFEST="${MANIFEST:-campaign/$CAMP/manifest.tsv}"
# 这台卡在 450 W / 320 W 都 3 分钟内掉总线（Xid 79）；实测 280 W + 锁频 1600 MHz
# 能连续跑满一个 epoch。重启后 nvidia-smi 设置会复位，所以每轮都要重设。
POWER_W="${POWER_W:-280}"
LOCK_CLK="${LOCK_CLK:-}"        # 默认不锁频（用户要求放开频率）；要压瞬态就传 LOCK_CLK=1600
PY="${PY:-python3}"
GUARD_LOG="logs/guard.log"
SELF="$PWD/scripts/run_guard.sh"

stamp() { date '+%F %T'; }
say() { echo "[$(stamp)] [guard] $*" | tee -a "$GUARD_LOG"; }

gpu_name() { nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1; }
gpu_ok()  { local n; n=$(gpu_name); [ -n "$n" ] && case "$n" in *Error*|*Unable*) return 1;; *) return 0;; esac; }

tune_gpu() {
  if [ "$POWER_W" != "0" ]; then
    nvidia-smi -pl "$POWER_W" >/dev/null 2>&1 && say "功耗限制 ${POWER_W} W" || say "[warn] 限功耗失败"
  fi
  if [ -n "$LOCK_CLK" ]; then
    nvidia-smi -lgc "$LOCK_CLK" >/dev/null 2>&1 && say "锁频 ${LOCK_CLK} MHz（抑制电流瞬态尖峰）" \
      || say "[warn] 锁频失败（无权限或不支持）"
  fi
}

expected_runs() { tail -n +2 "$MANIFEST" 2>/dev/null | grep -v '^#' | grep -c . ; }
finished_runs() { find outputs -mindepth 2 -maxdepth 2 -name 'eval_eval.json' 2>/dev/null | wc -l; }

reboot_if_allowed() {
  mkdir -p logs
  local hist=logs/guard_reboots.txt
  local now rc=0
  now=$(date +%s)
  # 只保留 6 小时内的重启记录
  if [ -f "$hist" ]; then awk -v t="$((now-21600))" '$1>=t' "$hist" > "$hist.tmp" && mv "$hist.tmp" "$hist"; fi
  local n; n=$(wc -l < "$hist" 2>/dev/null || echo 0)
  if [ "$n" -ge 3 ]; then
    say "6 小时内已重启 $n 次仍不见卡 —— 这是硬件问题，停止自动重启，等人换机"
    return 1
  fi
  echo "$now" >> "$hist"
  say "GPU 掉卡且重置无效，第 $((n+1)) 次重启本机（起机后 @reboot 会自动接回）"
  sync
  (sleep 2; reboot) >/dev/null 2>&1 &
  return 0
}

install_reboot_hook() {
  local line="@reboot cd $PWD && nohup setsid bash $SELF < /dev/null >> $PWD/logs/guard.log 2>&1 &"
  command -v crontab >/dev/null 2>&1 || { say "没有 crontab，跳过 @reboot 钩子"; return 0; }
  crontab -l 2>/dev/null | grep -q 'scripts/run_guard.sh' && return 0
  ( crontab -l 2>/dev/null; echo "$line" ) | crontab - && say "@reboot 钩子已写入 crontab"
}

install_reboot_hook
say "看门狗启动：pid=$$ campaign=$CAMP 期望 run=$(expected_runs) 已完成=$(finished_runs)"

while :; do
  if ! gpu_ok; then
    say "GPU 不可用（$(gpu_name || echo 'no devices')）；先尝试 nvidia-smi -r"
    timeout 30 nvidia-smi -r >/dev/null 2>&1 || true
    sleep 5
    if ! gpu_ok; then
      reboot_if_allowed || { say "等待人工处理：请到租用平台控制台强制重启或换一台机器"; sleep 600; continue; }
      sleep 60
      continue
    fi
    say "GPU 重置后恢复：$(gpu_name)"
  fi
  tune_gpu

  local_exp=$(expected_runs); local_fin=$(finished_runs)
  if [ "$local_fin" -ge "$local_exp" ] && [ "$local_exp" -gt 0 ]; then
    say "全部 $local_exp 个 run 都有 eval_eval.json，看门狗正常退出"
    exit 0
  fi

  say "开跑流水线（已完成 $local_fin / $local_exp）；日志 logs/one_shot.log"
  bash scripts/run_one_shot.sh > logs/one_shot.log 2>&1
  rc=$?
  case $rc in
    0) say "run_one_shot 正常结束（rc=0）";;
    2) say "冒烟判定 ABORT：一致性项没被优化，先修目标函数，别继续烧机时"; exit 2;;
    *) say "run_one_shot 异常退出 rc=$rc，60 s 后重试（已完成 $local_fin / $local_exp）"; sleep 60;;
  esac
  sleep 30
done
