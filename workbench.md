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

## Belohnung: Kosten + Komfort

`logic/boptest_gym_env.py::HVACReward` (Unterklasse von `BoptestGymEnv`) berechnet die
Belohnung als negierten Anstieg von

```
Zielfunktion = cost_tot + w_comfort * tdis_tot
```

`cost_tot` und `tdis_tot` sind BOPTESTs eigene, offiziell definierte KPIs (Betriebskosten
zum dynamischen Strompreis bzw. integriertes Komfort-Defizit in Kelvin-Stunden) — nicht
selbst nachgerechnet, sondern direkt von der BOPTEST-API abgefragt (`GET /kpi/{testid}`).
Das stellt sicher, dass die Belohnung exakt das misst, was auch die offizielle Auswertung
(`python -m logic.evaluation`) ausweist. `w_comfort` ist in der Oberfläche einstellbar
(Standard 1.0, siehe `logic/live_control.py::DEFAULT_REWARD`).

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
