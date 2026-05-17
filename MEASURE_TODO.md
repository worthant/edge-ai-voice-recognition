# Чеклист замеров: 19 моделей

Для каждой модели:

1. Прошить с `PLATFORMIO_MODEL_SLUG=<slug>`
2. Запустить `kws_benchmark()`, скачать `/spiffs/profile.csv`
3. Переименовать → положить в `measurements/profile/`
4. Запустить отдельный прогон с INA228 (cascade с GPIO-маркерами FSM)
5. CSV от ESP32-C3 → переименовать → положить в `measurements/energy/`

**Папки:**

- `measurements/profile/` — послойный тайминг (от kws_benchmark на ESP32-S3)
- `measurements/energy/` — энергия (от INA228 + ESP32-C3 logger)

---

## Группа A — Pareto по filters (b=6, 10 моделей)

- [x] **f64_b6_qat**
  - [x] `measurements/profile/profile_f64_b6_qat.csv`
  - [x] `measurements/energy/energy_f64_b6_qat.csv`
  - [x] `measurements/stats/stats_f64_b6_qat.csv`

- [x] **f96_b6_qat**
  - [x] `measurements/profile/profile_f96_b6_qat.csv`
  - [x] `measurements/energy/energy_f96_b6_qat.csv`
  - [x] `measurements/stats/stats_f96_b6_qat.csv`

- [ ] **f128_b6_qat**
  - [ ] `measurements/profile/profile_f128_b6_qat.csv`
  - [ ] `measurements/energy/energy_f128_b6_qat.csv`
  - [ ] `measurements/stats/stats_f128_b6_qat.csv`

- [ ] **f160_b6_qat**
  - [ ] `measurements/profile/profile_f160_b6_qat.csv`
  - [ ] `measurements/energy/energy_f160_b6_qat.csv`
  - [ ] `measurements/stats/stats_f160_b6_qat.csv`

- [ ] **f168_b6_qat**
  - [ ] `measurements/profile/profile_f168_b6_qat.csv`
  - [ ] `measurements/energy/energy_f168_b6_qat.csv`
  - [ ] `measurements/stats/stats_f168_b6_qat.csv`

- [ ] **f172_b6_qat** ← baseline, НЕ aligned
  - [ ] `measurements/profile/profile_f172_b6_qat.csv`
  - [ ] `measurements/energy/energy_f172_b6_qat.csv`
  - [ ] `measurements/stats/stats_f172_b6_qat.csv`

- [ ] **f176_b6_qat**
  - [ ] `measurements/profile/profile_f176_b6_qat.csv`
  - [ ] `measurements/energy/energy_f176_b6_qat.csv`
  - [ ] `measurements/stats/stats_f176_b6_qat.csv`

- [ ] **f184_b6_qat**
  - [ ] `measurements/profile/profile_f184_b6_qat.csv`
  - [ ] `measurements/energy/energy_f184_b6_qat.csv`
  - [ ] `measurements/stats/stats_f184_b6_qat.csv`

- [ ] **f192_b6_qat**
  - [ ] `measurements/profile/profile_f192_b6_qat.csv`
  - [ ] `measurements/energy/energy_f192_b6_qat.csv`
  - [ ] `measurements/stats/stats_f192_b6_qat.csv`

- [ ] **f224_b6_qat**
  - [ ] `measurements/profile/profile_f224_b6_qat.csv`
  - [ ] `measurements/energy/energy_f224_b6_qat.csv`
  - [ ] `measurements/stats/stats_f224_b6_qat.csv`

---

## Группа B — PTQ vs QAT (4 модели)

- [x] **f96_b6_ptq**
  - [x] `measurements/profile/profile_f96_b6_ptq.csv`
  - [x] `measurements/energy/energy_f96_b6_ptq.csv`
  - [x] `measurements/stats/stats_f96_b6_ptq.csv`

- [ ] **f172_b6_ptq**
  - [ ] `measurements/profile/profile_f172_b6_ptq.csv`
  - [ ] `measurements/energy/energy_f172_b6_ptq.csv`
  - [ ] `measurements/stats/stats_f172_b6_ptq.csv`

- [ ] **f176_b6_ptq**
  - [ ] `measurements/profile/profile_f176_b6_ptq.csv`
  - [ ] `measurements/energy/energy_f176_b6_ptq.csv`
  - [ ] `measurements/stats/stats_f176_b6_ptq.csv`

- [ ] **f192_b6_ptq**
  - [ ] `measurements/profile/profile_f192_b6_ptq.csv`
  - [ ] `measurements/energy/energy_f192_b6_ptq.csv`
  - [ ] `measurements/stats/stats_f192_b6_ptq.csv`

---

## Группа C — глубина сети (f=176, 5 моделей)

- [ ] **f176_b2_qat**
  - [ ] `measurements/profile/profile_f176_b2_qat.csv`
  - [ ] `measurements/energy/energy_f176_b2_qat.csv`
  - [ ] `measurements/stats/stats_f176_b2_qat.csv`

- [ ] **f176_b4_qat**
  - [ ] `measurements/profile/profile_f176_b4_qat.csv`
  - [ ] `measurements/energy/energy_f176_b4_qat.csv`
  - [ ] `measurements/stats/stats_f176_b4_qat.csv`

- [ ] **f176_b5_qat**
  - [ ] `measurements/profile/profile_f176_b5_qat.csv`
  - [ ] `measurements/energy/energy_f176_b5_qat.csv`
  - [ ] `measurements/stats/stats_f176_b5_qat.csv`

- [ ] **f176_b7_qat**
  - [ ] `measurements/profile/profile_f176_b7_qat.csv`
  - [ ] `measurements/energy/energy_f176_b7_qat.csv`
  - [ ] `measurements/stats/stats_f176_b7_qat.csv`

- [ ] **f176_b8_qat**
  - [ ] `measurements/profile/profile_f176_b8_qat.csv`
  - [ ] `measurements/energy/energy_f176_b8_qat.csv`
  - [ ] `measurements/stats/stats_f176_b8_qat.csv`

---

**Итого: 19 моделей × 3 файла = 57 csv**
