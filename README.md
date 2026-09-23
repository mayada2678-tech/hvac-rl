# HVAC-RL — BOPTEST

Ein Reinforcement-Learning-Agent (SAC, alternativ PPO/TD3) lernt, eine Wärmepumpe **und**
einen Batteriespeicher so zu steuern, dass er bei **dynamischem Strompreis** Kosten spart,
ohne den **thermischen Komfort** zu verletzen — auf [BOPTEST](https://github.com/ibpsa/project1-boptest)
(Testfall `bestest_hydronic_heat_pump`: Einzonen-Wohngebäude mit Wärmepumpe und
Fußbodenheizung). Batterie und eine PV-Anlage gibt es im BOPTEST-Testfall selbst nicht —
beide kommen als eigenständige Python-Schicht obendrauf (`logic/battery_env.py`,
`logic/solar_env.py`), die den Netzbezug und damit die Kosten senken, wenn sie genutzt
werden — Details in [workbench.md](workbench.md). Vergleich am Ende: der trainierte Agent
gegen BOPTESTs eingebauten Rule-Based Controller (RBC), auf derselben, nie im Training
gesehenen Testperiode.

## Voraussetzungen

- Python 3.11 (siehe `.venv311`), `pip install -r requirements.txt`
- [Docker Desktop](https://docs.docker.com/get-docker/) — BOPTEST läuft als eigener
  Docker-Dienst (REST-API), nicht als Python-Bibliothek wie zuvor CityLearn.

## BOPTEST starten

```powershell
scripts\start_boptest.ps1
```

Klont BOPTEST (falls noch nicht geschehen) in einen Geschwisterordner (`..\boptest`) und
baut/startet den Dienst per Docker Compose mit **6 parallelen Arbeitsprozessen**
(`-Workers 6`, siehe unten warum). **Der erste Start dauert deutlich über 10 Minuten** (eine
Conda-Umgebung mit der FMU-Simulationsbibliothek `pyfmi` wird gebaut). Spätere Starts sind
schnell. Läuft danach unter `http://127.0.0.1:8000`.

Stoppen: `scripts\stop_boptest.ps1`

**Wichtig — warum mehrere Worker:** Jedes laufende Training braucht *zwei* BOPTEST-Umgebungen
gleichzeitig (Training + Auswertung). Mit nur einem Worker (BOPTESTs Standard) bleibt das
Training beim Aufbau der zweiten Umgebung hängen. Trainieren Sie nur einen Algorithmus auf
einmal, reicht `scripts\start_boptest.ps1 -Workers 2`; für SAC+PPO+TD3 gleichzeitig (wie es
die Oberfläche erlaubt) braucht es 6.

## Oberfläche starten

```bash
python app.py
```

Startet BOPTEST **nicht** automatisch mit (das dauert beim ersten Mal zu lange, um es bei
jedem Programmstart zu prüfen) — vorher `scripts\start_boptest.ps1` laufen lassen. Die
Oberfläche (PySide6-Fenster, kein Browser) bietet:

- **🤖 Agent**: SAC, PPO und TD3 einzeln, zu zweit oder alle drei gleichzeitig trainieren
  (Start / Pause / Weiter / Stopp, Netzwerk & Hyperparameter je Algorithmus, live Lernkurve).
  Den Agenten auf der Testperiode beobachten (Stunde für Stunde: Zonentemperatur vs.
  Komfortband, Wärmepumpen-Modulation, Strompreis) und gegen den RBC vergleichen.
- **📊 Datensatz**: oben ein Auswahlfeld für **jeden bei BOPTEST bereitgestellten Testfall**
  (`bestest_air`, `bestest_hydronic`, `bestest_hydronic_heat_pump`, mehrere Mehrzonen- und
  Wohn-/Bürogebäude-Varianten, …) — auch ein selbst eintippbarer, eigener Testfall. Die ganze
  Analyse passt sich dynamisch an, was der gewählte Testfall tatsächlich anbietet, nichts ist
  fest auf ein Gebäude zugeschnitten.
  - **🧾 Übersicht & Datenqualität**: alle Kennzahlen auf einen Blick, dazu eine echte
    Datenqualitätsprüfung (Vollständigkeit, Zeitlücken, auffällige Spalten) der geladenen
    Jahresdaten.
  - **🔬 Alle Merkmale**: wirklich *alle* Mess-, Vorhersage- und Eingangsgrößen des gewählten
    Testfalls (z. B. 78 bei `bestest_hydronic_heat_pump`, 186 bei `multizone_office_simple_air`)
    — mit BOPTESTs eigenen Metadaten (Einheit, Grenzen, Beschreibung), nicht nur eine Auswahl.
  - **💶 Preisszenarien**, **📈 Zeitreihen**, **📊 Verteilung & Tagesprofil**: alle Preis-
    Randbedingungen im Vergleich (Strom, ggf. Gas/Fernwärme/Biomasse), jede verfügbare Größe
    als Zeitreihe/Verteilung/Tagesprofil wählbar, übers ganze simulierte Jahr.

  Kommt live von BOPTEST, keine lokalen CSV-Dateien wie zuvor bei CityLearn.

Details zur internen Funktionsweise: [workbench.md](workbench.md).

## Ohne Oberfläche (Kommandozeile)

```bash
python -m logic.training --algo SAC --config configs/sac.yaml --seed 0
python -m logic.evaluation      # RBC + alle Modelle in models/, BOPTESTs offizielle KPIs
tensorboard --logdir runs
```

## Projektstruktur

```
app.py                    Desktop-Einstiegspunkt (PySide6)
scripts/                   start_boptest.ps1 / stop_boptest.ps1
logic/
├── boptest_gym_env.py         Gymnasium-Umgebung für BOPTEST (REST-Client), Drittanbieter-
│                              Ursprung siehe THIRD_PARTY_NOTICES.md
├── battery_env.py               Batterie als eigener Wrapper (Ladezustand, 2. Aktion, Kosten)
├── solar_env.py                   PV-Anlage als eigener Wrapper (Erzeugung, Kosten)
├── envs.py                          Testfall-Auswahl, Beobachtungen/Aktionen, train/test-Split
├── baselines.py                       RBC-Baseline = BOPTESTs eingebauter Regler
├── watch.py                       Eine Testepisode aufzeichnen (für Live-Beobachtung)
├── training.py                    Trainings-Kernlogik (CLI + Oberfläche, eine train()-Funktion)
├── live_control.py                 Hyperparameter-Empfehlungen, Dateisteuerung (Pause/Stopp)
└── evaluation.py                    BOPTESTs KPIs für RBC + trainierte Modelle
logic/dataset.py               Randbedingungen (Wetter/Preis/Lasten) fürs ganze Jahr laden
gui/
├── agent_view.py               Training + Beobachten
├── watch_view.py                 UI-Baustein "Beobachten"
├── dataset_view.py                 Seite "Datensatz": Randbedingungen erkunden
├── main_window.py                   Hauptfenster
└── theme.py                          Dark+-Oberflächen-Theme
configs/{sac,ppo,td3}.yaml    Hyperparameter je Algorithmus
models/, runs/, results/       Gespeicherte Modelle, TensorBoard-/Live-Logs, KPI-Tabellen
```
