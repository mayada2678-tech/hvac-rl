"""Randbedingungen eines BOPTEST-Testfalls laden und zusammenfassen — Wetter, Strompreis,
interne Lasten, Komfort-Sollwerte, CO2-Intensität. Grundlage der "Datensatz"-Seite.

Bewusst *nicht* auf einen festen Satz Merkmale oder den einen Testfall aus logic/envs.py
zugeschnitten: jede Funktion nimmt `testcase` als Parameter, und welche Größen es gibt, wird
bei jedem Aufruf live von BOPTEST erfragt (verschiedene Testfälle haben unterschiedliche
Merkmale — z. B. hat `multizone_office_simple_air` keinen dynamischen Strompreis, dafür 62
Vorhersagegrößen, `testcase1` dagegen Gas-/Fernwärme-/Biomasse-Preise). Die "Datensatz"-Seite
kann dadurch zwischen allen bei BOPTEST verfügbaren Testfällen wechseln, ohne Code-Änderung.

Anders als bei CityLearn steckt hier keine lokale CSV-Sammlung dahinter: BOPTEST liefert
diese Größen als "Vorhersage" über seine REST-API — in Wahrheit sind es deterministische,
bekannte Randbedingungen (Wetterjahr, Preisszenario), keine unsichere Prognose. Ein Aufruf
mit `horizon` über ein ganzes Jahr liefert deshalb exakt die Werte, die später während
Training/Test auch tatsächlich auftreten.
"""
import requests
import pandas as pd

from logic.envs import TESTCASE, URL

# Bekannte Punktnamen -> hübscher Anzeigename samt Einheit, rein kosmetisch. Nicht
# abschließend: jeder Testfall kann zusätzliche, hier unbekannte Punkte mitbringen, die dann
# einfach mit ihrem Rohnamen angezeigt werden (siehe label() unten).
KNOWN_LABELS = {
    'TDryBul': 'Außentemperatur (K)', 'HDirNor': 'Direkte Solarstrahlung (W/m²)',
    'HGloHor': 'Globalstrahlung horizontal (W/m²)', 'HDifHor': 'Diffuse Solarstrahlung (W/m²)',
    'relHum': 'Relative Luftfeuchte (-)', 'winSpe': 'Windgeschwindigkeit (m/s)',
    'PriceElectricPowerConstant': 'Strompreis, konstant',
    'PriceElectricPowerDynamic': 'Strompreis, dynamisch (Tag-voraus, stündlich)',
    'PriceElectricPowerHighlyDynamic': 'Strompreis, stark dynamisch (15-Minuten-Raster)',
    'PriceGasPower': 'Gaspreis', 'PriceDistrictHeatingPower': 'Fernwärmepreis',
    'PriceBiomassPower': 'Biomassepreis', 'PriceSolarThermalPower': 'Solarthermie-Preis',
    'EmissionsElectricPower': 'CO2-Intensität Strom (kg/kWh)',
    'EmissionsGasPower': 'CO2-Intensität Gas (kg/kWh)',
    'EmissionsDistrictHeatingPower': 'CO2-Intensität Fernwärme (kg/kWh)',
}
# Für die Preisszenario-Übersicht: alle Punkte, die mit "Price" beginnen, gehören dazu — wird
# dynamisch je Testfall bestimmt (siehe price_points()), diese Farbtabelle deckt die
# gebräuchlichsten ab, alles andere bekommt automatisch eine Reservefarbe (siehe gui/dataset_view.py).
PRICE_COLORS = {'PriceElectricPowerConstant': '#9d9d9d', 'PriceElectricPowerDynamic': '#4FC1FF',
                'PriceElectricPowerHighlyDynamic': '#F48771', 'PriceGasPower': '#DCDCAA',
                'PriceDistrictHeatingPower': '#C586C0', 'PriceBiomassPower': '#B5CEA8'}

TEMPERATURE_HINTS = ('TDryBul', 'TDewPoi', 'TBlaSky', 'TWetBul', 'LowerSetp', 'UpperSetp', 'TZon', 'TRoo')
SECONDS_PER_YEAR = 365 * 24 * 3600


def label(point: str) -> str:
    """Anzeigename für einen Punkt — bekannter hübscher Name, sonst der Rohname selbst."""
    return KNOWN_LABELS.get(point, point)


def is_temperature(point: str) -> bool:
    """Grobe Heuristik, ob ein Punkt in Kelvin gemessen wird (für die °C-Umrechnung in
    load_year) — nicht abschließend, aber deckt die gebräuchlichen Namensmuster ab."""
    return any(hint in point for hint in TEMPERATURE_HINTS)


def list_testcases(url: str = URL) -> list[str]:
    """Alle bei diesem BOPTEST-Dienst bereitgestellten Testfälle (per `provision` hochgeladen,
    siehe workbench.md). Das ist die "Datensatz wechseln"-Auswahl in der Oberfläche."""
    payload = requests.get(f'{url}/testcases', timeout=10).json()
    return sorted(tc['testcaseid'] for tc in payload)


def select_testcase(testcase: str = TESTCASE, url: str = URL) -> str:
    """Testfall auswählen, liefert eine testid für nachfolgende Aufrufe. Muss mit
    stop_testcase() wieder freigegeben werden."""
    return requests.post(f'{url}/testcases/{testcase}/select').json()['testid']


def stop_testcase(testid: str, url: str = URL):
    requests.put(f'{url}/stop/{testid}')


def list_all_points(testcase: str = TESTCASE, url: str = URL) -> tuple[dict, dict, dict]:
    """Alle Mess-, Vorhersage- und Eingangsgrößen dieses Testfalls, mit vollständigen
    Metadaten (Einheit, Beschreibung, ggf. Grenzen) — vollständig, nicht kuratiert."""
    testid = select_testcase(testcase, url)
    try:
        measurements = requests.get(f'{url}/measurements/{testid}').json()['payload']
        forecasts = requests.get(f'{url}/forecast_points/{testid}').json()['payload']
        inputs = requests.get(f'{url}/inputs/{testid}').json()['payload']
    finally:
        stop_testcase(testid, url)
    return measurements, forecasts, inputs


def price_points(forecasts: dict) -> list[str]:
    """Alle Preis-Randbedingungen eines Testfalls (können je nach Testfall mehrere Energieträger
    sein: Strom, Gas, Fernwärme, Biomasse, ...) — dynamisch bestimmt, nicht fest verdrahtet."""
    return sorted(p for p in forecasts if p.startswith('Price'))


def load_year(testcase: str = TESTCASE, url: str = URL, interval: int = 3600,
             points: list[str] | None = None) -> pd.DataFrame:
    """Randbedingungen eines Testfalls für ein ganzes Jahr, stündlich (Standard). Ohne
    `points`: **alle** verfügbaren Vorhersagegrößen dieses Testfalls (nicht nur eine
    Auswahl) — macht die Funktion für jeden BOPTEST-Testfall gleichermaßen nutzbar. Belegt
    kurzzeitig einen BOPTEST-Arbeitsprozess (worker) — wird danach wieder freigegeben.
    """
    testid = select_testcase(testcase, url)
    try:
        forecasts = requests.get(f'{url}/forecast_points/{testid}').json()['payload']
        available = points if points is not None else list(forecasts)
        available = [p for p in available if p in forecasts]
        if not available:
            raise ValueError(f'Testfall "{testcase}" bietet keine der angefragten Randbedingungen an.')
        horizon = SECONDS_PER_YEAR - interval
        payload = requests.put(f'{url}/forecast/{testid}',
                               json={'point_names': available, 'horizon': int(horizon),
                                    'interval': int(interval)}).json()['payload']
    finally:
        stop_testcase(testid, url)

    df = pd.DataFrame(payload)
    for col in df.columns:
        if col != 'time' and is_temperature(col):
            df[col] = df[col] - 273.15
    df['day'] = (df['time'] // 86400).astype(int)
    df['hour'] = (df['time'] // 3600 % 24).astype(int)
    return df


def summary(df: pd.DataFrame) -> dict:
    """Kennzahlen über das ganze Jahr — für eine Übersichtskarte. Sucht sich passende
    Temperatur-/Preis-/Emissionsspalten dynamisch aus den tatsächlich geladenen Spalten."""
    n_hours = len(df)
    out = {'Stunden': n_hours, 'Tage': round(n_hours * (df['time'].diff().median() or 3600) / 86400, 1)}
    temp_cols = [c for c in df.columns if c not in ('time', 'day', 'hour') and is_temperature(c)
                and 'Setp' not in c]
    if temp_cols:
        # Außenlufttemperatur (Trockenkugel) bevorzugen, falls vorhanden — aussagekräftiger
        # als z. B. Taupunkt als Headline-Kennzahl.
        c = 'TDryBul' if 'TDryBul' in temp_cols else temp_cols[0]
        out[f'{label(c)}-Spanne'] = (df[c].min(), df[c].max())
    for c in price_points(dict.fromkeys(df.columns)):
        if c in df.columns:
            out[f'{label(c)}-Spanne'] = (df[c].min(), df[c].max())
    for c in [x for x in df.columns if x.startswith('Emissions')]:
        out[f'{label(c)}-Spanne'] = (df[c].min(), df[c].max())
    return out


def daily_profile(df: pd.DataFrame, column: str) -> pd.Series:
    """Mittlerer Tagesverlauf (0-23 Uhr) einer Randbedingung übers ganze Jahr gemittelt."""
    return df.groupby('hour')[column].mean()


def quality_report(df: pd.DataFrame) -> dict:
    """Datenqualität der geladenen Jahres-Randbedingungen: fehlende Werte, Zeitlücken,
    konstante (degenerierte, also inhaltlich uninteressante) Spalten. Da diese Daten direkt
    von BOPTESTs eigener, deterministischer Randbedingung kommen (kein Mess-/Sensorrauschen
    wie bei einem realen Datenlogger), ist "Vollständigkeit" hier vor allem eine Prüfung, ob
    die Übertragung selbst vollständig war."""
    value_cols = [c for c in df.columns if c not in ('time', 'day', 'hour')
                 and pd.api.types.is_numeric_dtype(df[c])]
    n_cells = len(df) * len(value_cols)
    n_missing = int(df[value_cols].isna().sum().sum()) if value_cols else 0

    dt = df['time'].diff().dropna()
    expected_interval = dt.median() if len(dt) else 3600
    gaps = int((dt > expected_interval * 1.5).sum())

    constant_cols = [c for c in value_cols if df[c].nunique(dropna=True) <= 1]

    return {
        'Spalten geprüft': len(value_cols),
        'Zellen gesamt': n_cells,
        'davon fehlend': n_missing,
        'Vollständigkeit (%)': 100 * (1 - n_missing / n_cells) if n_cells else 100.0,
        'Zeitlücken': gaps,
        'Erwartetes Intervall (s)': int(expected_interval),
        'Konstante Spalten': constant_cols,
    }


if __name__ == '__main__':
    print('Verfügbare Testfälle:', list_testcases())
