# Werkbank: Architektur & technische Details

Notizen für alle, die weiterbauen. Für die Nutzersicht siehe [README.md](README.md).

## Warum BOPTEST anders behandelt wird als vorher CityLearn

CityLearn ist eine Python-Bibliothek: `pip install citylearn`, Simulation läuft im selben
Prozess. **BOPTEST läuft als eigener Dienst** (Docker, mehrere Container: `web`, `worker`,
`provision`, dazu Redis + MinIO als Abhängigkeiten), der über eine REST-API angesprochen
wird (`http://127.0.0.1:8000`). Das hat Konsequenzen für die ganze Architektur:

- **Kein `pip install boptest`.** `logic/boptest_gym_env.py` ist eine bereinigte,
  eigenständige Kopie von [BOPTEST-Gym](https://github.com/ibpsa/project1-boptest-gym)
  (kein PyPI-Release dieses Projekts, siehe [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
  für Lizenz/Attribution und die Liste der Änderungen).
- **BOPTEST muss vor der Oberfläche laufen** — `scripts/start_boptest.ps1` (siehe README),
  nicht automatisch aus `app.py` heraus gestartet, weil der erste Build >10 Minuten dauert.
- **Jeder Simulationsschritt ist ein HTTP-Request**, kein reiner In-Memory-Aufruf. Das macht
  Training spürbar langsamer als bei CityLearn — deshalb sind `total_timesteps`/`buffer_size`
  in `logic/live_control.py::RECOMMENDED` und `configs/*.yaml` bewusst viel kleiner als vorher
  (20.000 statt 100.000+ Schritte).
- **Ein Testfall, eine Zone** statt mehrerer Gebäude wie bei CityLearn: `bestest_hydronic_heat_pump`
  hat eine Zone, eine Aktion (`oveHeaPumY_u`, Wärmepumpen-Modulationssignal 0–1). Deshalb
  entfällt die frühere Gebäudeauswahl in der Oberfläche komplett.
- **Kein browsbarer Roh-Datensatz aus CSV-Dateien.** CityLearns Dataset-Explorer las lokale
  CSVs je Gebäude; BOPTEST liefert seine Randbedingungen (Wetter, Preis, interne Lasten) nur
  live über die REST-API (`/forecast`, siehe unten). Die "Datensatz"-Seite gibt es weiterhin
  (`logic/dataset.py`, `gui/dataset_view.py`), holt ihre Daten aber bei jedem Neuladen frisch
  vom laufenden BOPTEST-Dienst statt aus Dateien.

### Zwei Infrastruktur-Bugs gefunden und gepatcht (beim Erststart reproduziert)

Betreffen nur den BOPTEST-Checkout selbst (per `scripts/start_boptest.ps1` automatisch
gepatcht), nicht dieses Projekt-Repo:

1. **MinIO-Images entfernt.** BOPTESTs `docker-compose.yml` referenziert `minio/minio` und
   `minio/mc` von Docker Hub. MinIO hat diese Images 2025 im Zuge einer Lizenzumstellung von
   Docker Hub entfernt (404 „repository does not exist"). Gepatcht auf `quay.io/minio/minio`
   bzw. `quay.io/minio/mc`.
2. **`mc`-Kommandozeile veraltet.** Die dadurch neu gezogene, aktuelle `mc`-Version hat den
   Befehl `mc config host add` durch `mc alias set` ersetzt. Ohne Patch schlägt die
   Bucket-Einrichtung im `mc`-Container fehl (`InvalidAccessKeyId`), `web` kann keine
   Testfälle laden und stürzt ab. Gepatcht: `mc config host add` → `mc alias set`.

### Mehrere Worker nötig (kein Bug, aber leicht zu übersehen)

BOPTEST startet standardmäßig mit **einem** Worker — reicht für einen einzelnen REST-Client,
aber `logic/training.py::train()` baut *zwei* Umgebungen gleichzeitig auf (Training +
Auswertung/`eval_env`). Mit nur einem Worker bleibt der zweite `PUT /initialize`-Aufruf
hängen bzw. schlägt mit `KeyError: 'payload'` fehl (leere/fehlerhafte Antwort, weil kein
Worker mehr frei ist). Reproduziert und behoben, indem `scripts/start_boptest.ps1`
standardmäßig mit `--scale worker=6` startet (`-Workers`-Parameter zum Anpassen) — 6, damit
auch SAC+PPO+TD3 gleichzeitig (je 2 Worker) laufen können, wie es die Oberfläche erlaubt.

## Belohnung: Kosten + Komfort + Batterie

`logic/envs.py::make_env()` baut standardmäßig einen Wrapper-Stapel:
`RewardWrapper(SolarEnv(BatteryEnv(BoptestGymEnv(...))))`. `BatteryEnv` und `SolarEnv` rechnen
nur Physik und Kosten und legen sie ins `info`-Dict; die Belohnung entsteht an genau einer
Stelle, in `logic/reward.py::RewardWrapper` (auch mit `battery=False, solar=False` — dann
dient BOPTESTs `cost_tot` als Kostenquelle). Je Schritt, in BOPTESTs KPI-Einheiten (Kosten
in €/m², Komfort in K·h), damit das Komfortgewicht dieselbe Bedeutung hat wie in BOPTEST-Gym:

```
r = -scale * [ (Kosten + Verschleiß - Restwert) / Wohnfläche + w_comfort * ΔKomfort-Defizit ]
Verschleiß  = |Batterieleistung| * Δt * Anschaffungspreis / (2 * Zyklen * Entladetiefe)
Restwert    = w_terminal * Δentnehmbare Batterieenergie * mittlerer Preis   (nur letzter Schritt)
```

Jeder Parameter ist in der Oberfläche einstellbar (Formular „Belohnungsfunktion“,
`gui/reward_form.py`, mit Voreinstellungen und „Beste theoretische Werte“-Knopf) und in
`configs/*.yaml` (Schlüssel `reward`). Standardwerte und Quellen (`logic/reward.py::REWARD_DEFAULTS`/`PARAM_INFO`):

| Parameter | Standard | Begründung |
|---|---|---|
| `w_comfort` (€/m² je K·h) | 1.0 | BOPTEST-Gym-Referenz `cost_tot + 1·tdis_tot`; dort auch 0.1 (kostenbetont) und 10 (komfortbetont) — als Voreinstellungen wählbar |
| `floor_area_m2` | 192 | Grundriss 12 m × 16 m laut BOPTEST-Doku von `bestest_hydronic_heat_pump` |
| `battery_invest_eur_per_kwh` | 500 | Richtwert LFP-Heimspeicher 2024/25 (grob 400–800 €/kWh) |
| `battery_cycle_life` | 6000 | übliche Herstellerangabe für LFP-Heimspeicher → ≈ 4.6 ct je kWh Durchsatz |
| `w_terminal` | 1.0 | Restwert voll anrechnen: sonst verbraucht der Agent die Startladung gratis |
| `scale` | 1.0 | ändert nur die Größenordnung, nicht das Optimum |

Zur Einordnung: w_comfort = 1 heißt für dieses Gebäude 1 K·h Komfortverletzung ≙ 192 €
Stromkosten — Komfort ist damit fast eine harte Grenze, gespart wird über Zeitverschiebung
und Batterie innerhalb des Komfortbands. Die Einzelanteile stehen im `info`-Dict
(`reward_cost`, `reward_comfort`, `reward_battery`, `reward_terminal`, zusammen = r) und in
den Aufzeichnungen von `logic/watch.py::record()`.

## Batterie & PV-Anlage — komplett in Python, unabhängig von BOPTEST

`bestest_hydronic_heat_pump` hat im FMU-Modell weder Batterie noch PV-Anlage. Beides kommt
rein aus zwei gestapelten `gymnasium.Wrapper`n obendrauf, die BOPTEST selbst nicht kennt:

**`logic/battery_env.py::BatteryEnv`** (wickelt `BoptestGymEnv` ein):
1. **Batterie-Zustand** (`battery_soc`, 0–1): eigener Python-Zustand, Schritt für Schritt
   fortgeschrieben, physikalisch beschränkt (Lade-/Entladewirkungsgrad, `MIN_SOC`/`MAX_SOC`
   als Schongrenzen, `MAX_POWER_KW` als Leistungsgrenze — alles Konstanten am Dateianfang).
2. **Erweiterter Aktionsraum**: `Box([0, -3], [1, 3])` — Dimension 0 ist weiterhin die
   Wärmepumpen-Modulation (geht an BOPTEST), Dimension 1 ist `battery_power_kw` (positiv =
   laden, negativ = entladen; geht **nie** an BOPTEST, bleibt rein lokal).
3. **Erweiterte Beobachtung**: `battery_soc` wird an die BOPTEST-Beobachtung angehängt.
4. **Kosten**: Netzbezug = `max(0, Wärmepumpenleistung + Batterieladeleistung)` — Laden
   erhöht den Netzbezug, Entladen senkt ihn (bei ausreichender Entladung bis auf 0, kein
   Export/keine Einspeisevergütung in diesem einfachen Modell). Die Wärmepumpenleistung kommt
   aus `reaPHeaPum_y`, das `BoptestGymEnv.step()` jetzt zusätzlich im `info['res']`-Dict
   durchreicht (kleine, bewusste Erweiterung der vendorten Datei, siehe dort). Der Strompreis
   wird — wie schon in `logic/watch.py::record()` — einmal pro Episode komplett vorab per
   `/forecast` geladen (deterministisches Szenario, kein Grund für einen API-Aufruf pro
   Schritt).

**`logic/solar_env.py::SolarEnv`** (wickelt `BatteryEnv` ein, gleiches Prinzip): keine eigene
Aktion — Solarerzeugung ist nicht steuerbar —, berechnet stattdessen je Schritt

```
P_solar = HDirNor * A_panel * eta_panel / 1000        [kW]
```

aus BOPTESTs *gemessener* (nicht vorhergesagter) direkter Solarstrahlung
`weaSta_reaWeaHDirNor_y` (aus demselben `info['res']`). `A_panel` (20 m²) und `eta_panel`
(19 %) sind **eigene Annahmen dieses Projekts**, keine BOPTEST-Vorgabe — als Konstanten am
Dateianfang klar gekennzeichnet und leicht änderbar. Solarstrom deckt zuerst den Netzbezug,
den `BatteryEnv` sonst berechnet hätte (kostenlos, wie bereits entladener Batteriestrom);
überschüssige Erzeugung wird in diesem einfachen Modell verworfen statt eingespeist.

Beide Wrapper hängen ihre jeweilige Zustandsgröße (`battery_soc`, `solar_power_kw`) zusätzlich
ins `info`-Dict jedes `step()`-Aufrufs (neben `grid_power_kw`, `heat_pump_power_kw`,
`step_cost`, `price`) — praktisch für Auswertung/Logging, ohne bei jedem Zugriff durch die
Wrapper-Kette zu müssen. `logic/watch.py::record()` liest das für die "Beobachten"-Ansicht.

**Bekannte Einschränkung:** Die feste Testperiode (`TEST_START`, 1.–3. Februar, siehe unten)
ist in diesem Wetterjahr komplett bewölkt (`HDirNor` = 0 durchgehend, verifiziert über
`logic/dataset.py::load_year()`) — die PV-Anlage produziert auf der Testperiode also nichts,
nur die Batterie ist dort sichtbar wirksam. Für einen Test mit sichtbarer PV-Nutzung eignet
sich z. B. ein sommerlicher Zeitraum (`TEST_START` entsprechend anpassen).

`logic/baselines.py::RuleBasedController` bleibt unverändert (setzt weiterhin nur
`env.unwrapped.actions = []`) — beide Wrapper erkennen die dabei leere Aktion und lassen die
Batterie einfach untätig, statt einen Fehler auszulösen; die RBC-Baseline ist dadurch effektiv
"Wärmepumpe vom eingebauten Regler, Batterie/PV ungenutzt".

## Rule-Based-Controller-Vergleich

Jeder BOPTEST-Testfall bringt bereits eine Referenzregelung mit (Teil des FMU-Modells). Sie
wird aktiv, sobald keine Aktion überschrieben wird. `logic/baselines.py::RuleBasedController`
macht daher nichts anderes, als `env.actions = []` zu setzen — der eingebaute Regler
übernimmt automatisch. Kein eigener PID-/Hysterese-Regler nötig, und der Vergleich ist fair,
weil es exakt die im Testfall hinterlegte Referenz ist (dieselbe, die auch BOPTESTs eigene
Publikationen als Baseline nutzen).

## Train/Test-Split

Wie zuvor bei CityLearn (zusammenhängende Trainings-/Testperiode, keine zufällige Mischung):

- **Training**: zufälliger Wochen-Startzeitpunkt übers Jahr (`random_start_time=True`),
  die Testperiode ist per `excluding_periods` ausgeschlossen.
- **Test**: feste Periode — die ersten drei Februartage, mit drei Tagen Einschwingzeit davor
  (`TEST_START`/`TEST_LENGTH`/`TEST_WARMUP` in `logic/envs.py`). Das ist die im
  BOPTEST-Gym-Ökosystem gängige Testperiode für diesen Testfall.

## Seite "Agent": Training als Hintergrundprozess

Unverändert gegenüber der vorherigen CityLearn-Version (siehe Git-Historie/ältere
Konversation für Details): Jedes Training läuft als eigener `python -m logic.training`-
Prozess, gesteuert über Dateien in `runs/live/` (Steuerdatei für Pause/Stopp, CSV-Lernkurve,
Statusdatei) — siehe `logic/live_control.py::paths_for()`. Das gilt unverändert für BOPTEST:
`logic/training.py::train()` kennt CityLearn vs. BOPTEST gar nicht, es ruft nur
`logic.envs.make_env()` auf.

**Stoppen, Abbrechen, Schließen** — ein Training läuft nie unbemerkt im Hintergrund weiter:

- **⏹ Stopp** (sauber): Steuerdatei `stop=true`; das Training beendet den aktuellen Schritt —
  bzw. bricht eine laufende Auswertungs-Episode sofort ab (`StopRequested`) —, sichert das
  Modell und beendet sich. Anzeige währenddessen „🟠 wird gestoppt …“; dauert Sekunden.
- **⛔ Sofort erzwingen**: nach Rückfrage wird der ganze Prozessbaum beendet (`psutil`; das
  venv-Python unter Windows ist nur ein Starter mit dem Training als Kindprozess). Die
  BOPTEST-Tests gibt dann die Oberfläche frei (`PUT /stop/{testid}`), die Test-IDs stehen
  dafür in der Statusdatei.
- **Fenster schließen** bei laufendem Training: Nachfrage „Stoppen & sichern“ (wartet bis
  90 s, dann hart) / „Sofort abbrechen“ / „Fenster offen lassen“.
- **Absturz der Oberfläche**: die Statusdatei enthält die Prozess-ID; beim nächsten Start
  findet `AgentView._reattach_running()` noch laufende Trainings wieder (geprüft über die
  Kommandozeile des Prozesses), Stopp/Pause funktionieren dann wieder.
- **BOPTEST-Freigabe**: `train()` gibt seine beiden BOPTEST-Tests in einem `finally`-Block
  immer frei (auch bei Stopp/Fehler). Vorher blieben sie belegt, und spätere Läufe scheiterten
  bei knappen Workern mit `KeyError: 'payload'`.

## Komfortgewicht-Studie & Reiter "Vergleich": Kosten-Komfort-Kurve statt Einzelergebnis

Ein einzelnes trainiertes Modell zeigt nur einen Punkt der Abwägung Kosten ↔ Komfort. Die
Studie trainiert mehrere Varianten, die sich **nur im Komfortgewicht w** unterscheiden, und
stellt sie auf derselben Testperiode dem RBC gegenüber:
„Bei w=1 kaum Ersparnis, bei w=0.1 sparen wir X %, aber Y K·h Komfortverletzung.“

1. **Belohnungsformular → „Komfortgewicht-Studie“** aktivieren, w-Werte eingeben (Standard
   `1.0, 0.3, 0.1`; Dezimalkomma geht auch: `1,0; 0,3; 0,1`). „▶ Start“ legt je Algorithmus
   und w einen Lauf an: `runs/live/sac_seed0_w0_3.*`, Modell `models/sac_seed0_w0_3.zip`
   (`logic/live_control.py::run_tag()`). Neben jedem Modell legt `logic/training.py` eine
   `models/<name>.json` mit den Belohnungsparametern ab — daher weiß der Vergleich, mit
   welchem w ein Modell trainiert wurde.
2. **Warteschlange:** jedes Training belegt zwei BOPTEST-Worker (Training + Auswertung), bei
   6 Workern laufen also höchstens 3 gleichzeitig (`MAX_PARALLEL_TRAININGS`). Weitere Läufe
   stehen auf „⏳ wartet auf freie BOPTEST-Worker“ und starten automatisch; „⏹ Stopp“
   verwirft wartende Läufe.
3. **Reiter „Vergleich“** (`gui/compare_view.py`, Logik in `logic/evaluation.py::compare()`):
   RBC + alle Modelle (ohne `*_final`) je eine Testepisode, im Hintergrund-Thread. Kennzahlen:
   Stromkosten **inkl. Batterie/PV** (Summe `info['step_cost']` — BOPTESTs `cost_tot` kennt
   beide nicht), Ersparnis ggü. RBC in %, Komfortverletzung `tdis_tot` (K·h), Netzbezug (kWh).
   Anzeige als Kosten-über-Komfort-Diagramm (je Algorithmus eine Kurve über w, RBC als Stern
   mit gestrichelter Kostenlinie, Details beim Überfahren mit der Maus), Tabelle und fertigen
   Sätzen zum Kopieren (`summary_sentences()`). Ergebnisse werden in `results/compare.csv`
   zwischengespeichert; nur neue/neu trainierte Modelle (Dateizeit) werden neu simuliert.
   Der Vergleich braucht selbst einen Worker — sind alle durch Trainings belegt, wird er
   nicht gestartet.

## Seite "Datensatz": jeder BOPTEST-Testfall live erkundbar, nicht nur einer

Bewusst **nicht** fest auf `bestest_hydronic_heat_pump` zugeschnitten: `logic/dataset.py`
nimmt `testcase` als Parameter überall entgegen, und `gui/dataset_view.py` hat oben ein
Auswahlfeld, das alle bei diesem BOPTEST-Dienst bereitgestellten Testfälle auflistet
(`logic/dataset.py::list_testcases()`, `GET /testcases`) — Stand dieses Projekts z. B.
`bestest_air`, `bestest_hydronic`, `bestest_hydronic_heat_pump`,
`multizone_office_complex_air`, `multizone_office_simple_air/_hydronic`,
`multizone_residential_hydronic`, `singlezone_commercial_hydronic`, `testcase1/2/3`,
`twozone_apartment_hydronic`. Das Auswahlfeld ist zusätzlich **editierbar**: ein Testfall, den
man selbst zusätzlich per BOPTESTs `provision`-Dienst hochgeladen hat (eigene FMU, siehe
[BOPTEST-Dokumentation](https://ibpsa.github.io/project1-boptest/)), lässt sich per Namen
eintippen und genauso laden.

**Wichtig zur Einordnung „anderen Datensatz hochladen":** BOPTEST-Testfälle sind keine
CSV-Dateien, sondern kompilierte Gebäude-Simulationsmodelle (FMU) mit fest hinterlegten
Randbedingungen. Ein neuer Testfall lässt sich deshalb nicht einfach per Datei-Upload in
dieser Oberfläche hinzufügen — das würde eine neue FMU + `provision`-Lauf im BOPTEST-Dienst
selbst voraussetzen. Was die Oberfläche bietet, ist das praktikable Äquivalent: verlustfrei
zwischen *allen* im laufenden BOPTEST-Dienst bereits bereitgestellten Testfällen wechseln,
inklusive eines, den man selbst nachträglich provisioniert hat.

Keiner der Punktnamen ist hartcodiert — verschiedene Testfälle haben grundverschiedene
Merkmale (z. B. hat `multizone_office_simple_air` 186 Merkmale über mehrere Zonen und keinen
dynamischen Strompreis, `testcase1` dagegen zusätzlich Gas-/Fernwärme-/Biomasse-Preise und
nur 25 Merkmale insgesamt). `logic/dataset.py::load_year()` lädt deshalb standardmäßig **alle**
Vorhersagegrößen des gewählten Testfalls für ein Jahr auf einmal
(`PUT /forecast/{testid}` mit `horizon` ≈ ein Jahr), und `gui/dataset_view.py` befüllt jedes
Auswahlfeld (Zeitreihen, Verteilung) dynamisch aus den tatsächlich geladenen Spalten. Der
Trick dahinter: BOPTESTs „Vorhersage" ist bei diesen Randbedingungen keine unsichere Prognose,
sondern eine deterministische, im Testfall fest hinterlegte Zeitreihe (Wetterjahr,
Preisszenario) — eine Abfrage ab Jahresbeginn liefert deshalb exakt dieselben Werte, die
später während Training/Test auch tatsächlich auftreten. Jede Auswahl belegt kurzzeitig einen
BOPTEST-Worker und gibt ihn danach sofort wieder frei (`PUT /stop/{testid}`).

Läuft BOPTEST nicht oder gibt es den eingetippten Testfall nicht, fängt `reload()` die
`requests.exceptions.RequestException` bzw. sonstige Fehler ab und zeigt eine Meldung statt
abzustürzen — derselbe Schutz wie in `gui/watch_view.py`.

### Alle Merkmale + Datenqualität

Der Tab „🔬 Alle Merkmale" zeigt **wirklich alle** Größen des gewählten Testfalls, nicht nur
eine Auswahl — `logic/dataset.py::list_all_points()` fragt `/measurements`,
`/forecast_points` und `/inputs` komplett ab und zeigt sie mit BOPTESTs eigenen Metadaten
(Einheit, Grenzen, Beschreibung) in drei sortierbaren Tabellen.

`logic/dataset.py::quality_report()` prüft die geladene Jahres-Zeitreihe auf fehlende Werte,
Zeitlücken (Abstand zwischen Zeitstempeln größer als das erwartete Intervall) und konstante
Spalten (z. B. ist ein `constant`-Preisszenario per Definition konstant — kein Fehler,
deshalb als Hinweis statt als Warnung markiert). Da die Daten direkt aus BOPTESTs
deterministischer Randbedingung kommen (kein Sensorrauschen wie bei einem realen Datenlogger),
prüft „Vollständigkeit" hier vor allem, ob die Übertragung selbst vollständig war.
`logic/dataset.py::summary()` und `::is_temperature()` erkennen Temperatur-/Preis-/
Emissionsspalten anhand von Namensmustern (z. B. „Price…" für alle Energieträger-Preise, ein
`TEMPERATURE_HINTS`-Tupel für die Kelvin→°C-Umrechnung) statt fester Spaltennamen, damit
dieselbe Logik für jeden Testfall funktioniert.

## Erweiterungsideen

- Anderer Testfall (z. B. `bestest_air` für reine Kühlung statt Heizung): `TESTCASE` in
  `logic/envs.py` ändern, `ACTIONS`/`OBSERVATIONS` an dessen Ein-/Ausgänge anpassen (Punktnamen
  über `GET /inputs/{testid}`, `/measurements/{testid}`, `/forecast_points/{testid}` abrufbar).
- Preis-Szenario `highly_dynamic` statt `dynamic` (stündlich vs. 15-minütlich variierender
  Preis, stärkere Preisspitzen) — `scenario` in `logic.envs.make_env()`.
- Mehrere Seeds parallel: `logic/live_control.py::paths_for()` ist schon auf `(algo, seed)`
  ausgelegt.
