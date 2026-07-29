#!/usr/bin/env bash
# Само-пробуждающий «молоти»-цикл. Раз в N минут печатает ОДНУ строку-событие в stdout →
# harness-Monitor превращает её в notification → будит сессию, даже если та «остановилась».
# N читается из state-файла КАЖДЫЙ тик → /moloti 10 меняет период на лету, без перезапуска.
# Singleton по flock: второй запуск при живом первом мгновенно выходит (не плодим будильники).
UID_=$(id -u)
LOCK="/tmp/moloti-${UID_}.lock"
STATE="/tmp/moloti-${UID_}.min"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo '{"moloti":"already-running"}'   # инкумбент жив — этот экземпляр самоустраняется
  exit 0
fi

while true; do
  m=$(cat "$STATE" 2>/dev/null)
  case "$m" in ''|*[!0-9]*) m=20 ;; esac      # дефолт/санитизация → 20 мин
  sleep $(( m * 60 ))
  echo "{\"moloti_nudge\":true,\"every_min\":${m},\"at\":\"$(date +%H:%M)\"}"
done
