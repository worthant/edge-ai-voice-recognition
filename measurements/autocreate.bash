#!/usr/bin/env bash
# Создаёт пустые CSV-файлы с заголовками под замеры новых моделей.
# Существующие файлы не трогает.
#
# Запуск из корня проекта (там где находится measurements/):
#   bash create_measurement_stubs.sh

set -euo pipefail

BASE="best"

# Объединённый список моделей для замеров (17 шт).
# Энергию мерим у всех. Профиль и stats — тоже у всех
# (послойная выборка — подмножество, но дешевле сразу всё снять).
MODELS=(
  "f24_b3_ptq"
  "f24_b5_qat"
  "f32_b4_ptq"
  "f32_b6_ptq"
  "f40_b2_ptq"
  "f48_b3_ptq"
  "f48_b4_qat"
  "f48_b5_ptq"
  "f64_b2_ptq"
  "f64_b3_qat"
  "f64_b7_ptq"
  "f72_b3_ptq"
  "f72_b4_qat"
  "f80_b5_qat"
  "f88_b3_ptq"
  "f88_b6_ptq"
  "f104_b5_ptq"
  "f176_b7_qat"
)

mkdir -p "$BASE/energy" "$BASE/profile" "$BASE/stats"

ENERGY_HEADER="timestamp,bus_voltage,bus_current,state"
PROFILE_HEADER="run_id,op_index,op_tag,ticks_us"
STATS_HEADER="op_index,op_tag,mean_ms,std_ms,min_ms,max_ms,median_ms,pct_time"

created=0
skipped=0

create_if_absent() {
  local path="$1"
  local header="$2"
  if [[ -e "$path" ]]; then
    echo "  skip  $path"
    skipped=$((skipped + 1))
  else
    echo "$header" > "$path"
    echo "  new   $path"
    created=$((created + 1))
  fi
}

for m in "${MODELS[@]}"; do
  create_if_absent "$BASE/energy/energy_${m}.csv"   "$ENERGY_HEADER"
  create_if_absent "$BASE/profile/profile_${m}.csv" "$PROFILE_HEADER"
  create_if_absent "$BASE/stats/stats_${m}.csv"     "$STATS_HEADER"
done

echo
echo "Создано: $created, пропущено (уже было): $skipped"
