"""Gymnasium-Umgebung für BOPTEST (Building Optimization Testing Framework).

Übernommen und für dieses Projekt bereinigt aus BOPTEST-Gym
(https://github.com/ibpsa/project1-boptest-gym, Javier Arroyo et al., BSD-3-Clause-Lizenz,
siehe THIRD_PARTY_NOTICES.md). Nicht per pip installierbar (kein PyPI-Release), deshalb hier
als eigenständige Datei übernommen statt als Git-Abhängigkeit — macht das Projekt
reproduzierbar ohne Netzwerkzugriff auf GitHub bei jeder Installation.

Änderungen gegenüber dem Original:
  - `examples.test_and_plot`-Import entfernt (gehörte zu Demo-Skripten, die dieses Projekt
    nicht verwendet — Live-Training/-Beobachtung übernehmen logic/training.py und
    gui/watch_view.py).
  - `render()`/`SaveAndTestCallback` entfernt (ungenutzt, gleicher Grund).
  - Bug im Original behoben: `self.testid` wurde im ersten `try`-Block referenziert, bevor es
    je zugewiesen wurde (führte beim allerersten Aufruf zu einem stillschweigend
    abgefangenen `AttributeError` — harmlos, aber unsauber).
  - `HVACReward`-Mixin ergänzt: konfigurierbares Komfortgewicht `w_comfort` statt fest
    einprogrammierter Subklassen je Gewichtung (passend zu diesem Projekt: dieselbe
    `w_comfort`-Konvention wie zuvor bei der CityLearn-Belohnung).

BOPTEST läuft als eigener REST-Dienst (Docker, siehe workbench.md) — diese Klasse redet
ausschließlich über HTTP mit ihm, keine lokale Gebäudesimulation im Python-Prozess.
"""
import random

import gymnasium as gym
import numpy as np
import requests
from gymnasium import spaces
from scipy import interpolate


class BoptestGymEnv(gym.Env):
    """BOPTEST-Umgebung im Gymnasium-Interface. Kommuniziert über die REST-API eines
    laufenden BOPTEST-Dienstes (siehe logic/envs.py für die Testfall-Auswahl)."""

    metadata = {'render.modes': ['console']}

    def __init__(self, url='http://127.0.0.1:8000', testcase='bestest_hydronic_heat_pump',
                actions=('oveHeaPumY_u',), observations=None, max_episode_length=3 * 3600,
                random_start_time=False, excluding_periods=None, regressive_period=None,
                predictive_period=None, start_time=0, warmup_period=0,
                scenario=None, step_period=3600):
        super().__init__()
        observations = observations or {'reaTZon_y': (280., 310.)}
        scenario = scenario or {'electricity_price': 'constant'}

        self.url = url
        self.testcase = testcase
        self.actions = list(actions)
        self.max_episode_length = max_episode_length
        self.random_start_time = random_start_time
        self.excluding_periods = excluding_periods
        self.start_time = start_time
        self.warmup_period = warmup_period
        self.predictive_period = predictive_period
        self.regressive_period = regressive_period
        self.step_period = step_period
        self.scenario = scenario
        self.testid = None

        self.bgn_year_margin = regressive_period if regressive_period is not None else 0
        self.end_year_margin = max_episode_length

        # Laufenden Test stoppen, falls schon einer ausgewählt war, dann neu auswählen.
        if self.testid is not None:
            try:
                requests.put(f'{url}/stop/{self.testid}')
            except requests.RequestException:
                pass
        self.testid = requests.post(f'{url}/testcases/{testcase}/select').json()['testid']
        self.name = requests.get(f'{url}/name/{self.testid}').json()['payload']
        self.all_measurement_vars = requests.get(f'{url}/measurements/{self.testid}').json()['payload']
        self.all_predictive_vars = requests.get(f'{url}/forecast_points/{self.testid}').json()['payload']
        self.all_input_vars = requests.get(f'{url}/inputs/{self.testid}').json()['payload']
        self.step_def = requests.get(f'{url}/step/{self.testid}').json()['payload']
        self.scenario_def = requests.get(f'{url}/scenario/{self.testid}').json()['payload']

        # ---------- Beobachtungsraum ----------
        for obs, bounds in observations.items():
            if len(bounds) != 2:
                raise ValueError(f'"{obs}": Werte müssen ein (min, max)-Tupel sein.')
        for obs in observations:
            if not (obs == 'time' or obs in self.all_measurement_vars or obs in self.all_predictive_vars):
                raise ReferenceError(f'"{obs}" ist weder Messgröße noch Vorhersagegröße dieses Testfalls.\n'
                                     f'Messgrößen: {list(self.all_measurement_vars)}\n'
                                     f'Vorhersagegrößen: {list(self.all_predictive_vars)}')

        self.measurement_vars = [o for o in observations if o in self.all_measurement_vars]
        self.observations, self.lower_obs_bounds, self.upper_obs_bounds = [], [], []

        if 'time' in observations:
            self.observations.append('time')
            self.lower_obs_bounds.append(observations['time'][0])
            self.upper_obs_bounds.append(observations['time'][1])

        self.observations.extend(self.measurement_vars)
        self.lower_obs_bounds.extend(observations[o][0] for o in self.measurement_vars)
        self.upper_obs_bounds.extend(observations[o][1] for o in self.measurement_vars)

        self.is_regressive = self.regressive_period is not None
        if self.is_regressive:
            if self.regressive_period <= 0:
                raise ValueError('regressive_period muss > 0 sein.')
            self.regressive_vars = self.measurement_vars
            self.regr_n = int(self.regressive_period / self.step_period)
            for obs in self.regressive_vars:
                obs_list = [f'{obs}_regr_{int(i * self.step_period)}' for i in range(1, self.regr_n + 1)]
                self.observations.extend(obs_list)
                self.lower_obs_bounds.extend([observations[obs][0]] * len(obs_list))
                self.upper_obs_bounds.extend([observations[obs][1]] * len(obs_list))

        self.is_predictive = any(o in self.all_predictive_vars for o in observations)
        self.predictive_vars = []
        if self.is_predictive:
            if self.predictive_period is None or self.predictive_period < 0:
                raise ValueError('predictive_period muss >= 0 sein, wenn Vorhersagegrößen genutzt werden.')
            self.predictive_vars = [o for o in observations if o in self.all_predictive_vars and o != 'time']
            self.pred_n = int(self.predictive_period / self.step_period) + 1
            for obs in self.predictive_vars:
                obs_list = [f'{obs}_pred_{int(i * self.step_period)}' for i in range(self.pred_n)]
                self.observations.extend(obs_list)
                self.lower_obs_bounds.extend([observations[obs][0]] * len(obs_list))
                self.upper_obs_bounds.extend([observations[obs][1]] * len(obs_list))
            self.end_year_margin = self.max_episode_length + self.predictive_period

        self.observation_space = spaces.Box(low=np.array(self.lower_obs_bounds, dtype=np.float32),
                                            high=np.array(self.upper_obs_bounds, dtype=np.float32),
                                            dtype=np.float32)

        # ---------- Aktionsraum ----------
        for act in self.actions:
            if act not in self.all_input_vars:
                raise ReferenceError(f'"{act}" ist keine Eingangsgröße dieses Testfalls.\n'
                                     f'Verfügbare Eingänge: {list(self.all_input_vars)}')
        self.lower_act_bounds = [self.all_input_vars[a]['Minimum'] for a in self.actions]
        self.upper_act_bounds = [self.all_input_vars[a]['Maximum'] for a in self.actions]
        self.action_space = spaces.Box(low=np.array(self.lower_act_bounds, dtype=np.float32),
                                       high=np.array(self.upper_act_bounds, dtype=np.float32),
                                       dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Setzt das Gebäudemodell zurück und lässt es `warmup_period` Sekunden mit dem
        eingebauten Regler einschwingen, bevor die Episode beginnt."""

        def find_start_time():
            start = random.randint(self.bgn_year_margin, int(3.1536e7 - self.end_year_margin))
            episode = (start, start + self.max_episode_length)
            if self.excluding_periods:
                for lo, hi in self.excluding_periods:
                    if episode[0] < hi and lo < episode[1]:
                        return find_start_time()
            return start

        if self.random_start_time:
            self.start_time = find_start_time()

        res = requests.put(f'{self.url}/initialize/{self.testid}',
                           json={'start_time': int(self.start_time),
                                'warmup_period': int(self.warmup_period)}).json()['payload']
        requests.put(f'{self.url}/step/{self.testid}', json={'step': int(self.step_period)})
        requests.put(f'{self.url}/scenario/{self.testid}', json=self.scenario)

        self.objective_integrand = 0.
        self.episode_rewards = []
        return self._get_observations(res), {'res': res}

    def stop(self):
        requests.put(f'{self.url}/stop/{self.testid}')

    def step(self, action):
        u = {}
        for i, act in enumerate(self.actions):
            u[act] = float(action[i])
            u[act.replace('_u', '_activate')] = 1.0

        res = requests.post(f'{self.url}/advance/{self.testid}', json=u).json()['payload']
        reward = self.get_reward()
        self.episode_rewards.append(reward)
        terminated = self._compute_terminated()
        truncated = res['time'] >= self.start_time + self.max_episode_length
        # `res` roh mit durchreichen (z. B. reaPHeaPum_y, die elektrische Wärmepumpenleistung) —
        # Wrapper wie logic/battery_env.py brauchen Messgrößen, die nicht Teil des gepackten
        # Beobachtungsvektors sind.
        return self._get_observations(res), reward, terminated, truncated, {'res': res}

    def close(self):
        pass

    def get_reward(self):
        """Standard-Belohnung: negierter Anstieg der Zielfunktion (Betriebskosten +
        gewichtetes Komfort-Defizit) — dieselbe Größe, mit der BOPTEST seine offiziellen
        KPIs berechnet. Überschrieben von HVACReward unten mit konfigurierbarem Gewicht."""
        kpis = requests.get(f'{self.url}/kpi/{self.testid}').json()['payload']
        objective_integrand = kpis['cost_tot'] + kpis['tdis_tot']
        reward = -(objective_integrand - self.objective_integrand)
        self.objective_integrand = objective_integrand
        return reward

    def _compute_terminated(self):
        return False

    def _get_observations(self, res):
        observations = []
        if 'time' in self.observations:
            observations.append(res['time'] % self.upper_obs_bounds[0])
        for obs in self.measurement_vars:
            observations.append(res[obs])

        if self.is_regressive:
            regr_index = res['time'] - self.step_period * np.arange(1, self.regr_n + 1)
            for var in self.regressive_vars:
                res_var = requests.put(f'{self.url}/results/{self.testid}',
                                       json={'point_names': [var], 'start_time': int(regr_index[-1]),
                                            'final_time': int(regr_index[0])}).json()['payload']
                f = interpolate.interp1d(res_var['time'], res_var[var], kind='linear', fill_value='extrapolate')
                observations.extend(f(regr_index))

        if self.is_predictive:
            predictions = requests.put(f'{self.url}/forecast/{self.testid}',
                                       json={'point_names': self.predictive_vars,
                                            'horizon': int(self.predictive_period),
                                            'interval': int(self.step_period)}).json()['payload']
            for var in self.predictive_vars:
                observations.extend(predictions[var][i] for i in range(self.pred_n))

        return np.array(observations, dtype=np.float32)

    def get_kpis(self):
        """BOPTESTs offizielle Kennzahlen (Kosten, Komfort-Defizit, Emissionen, ...)."""
        return requests.get(f'{self.url}/kpi/{self.testid}').json()['payload']

    def get_results(self, point_names, start_time=None, final_time=3.1536e7):
        """Zeitreihe beliebiger Mess-/Eingangsgrößen über den Verlauf einer Episode —
        Grundlage für logic/watch.py (Live-Beobachtung, Tab "Beobachten")."""
        start = self.start_time + 1 if start_time is None else start_time
        return requests.put(f'{self.url}/results/{self.testid}',
                            json={'point_names': list(point_names), 'start_time': start,
                                 'final_time': final_time}).json()['payload']


class HVACReward(BoptestGymEnv):
    """BOPTEST-Umgebung mit einstellbarem Komfortgewicht (statt BOPTESTs fest verdrahteter
    Beispiel-Subklassen je Gewichtung) — dieselbe `w_comfort`-Idee wie zuvor bei der
    CityLearn-Version dieses Projekts: höheres Gewicht = strengere Bestrafung von
    Komfortabweichungen gegenüber den Betriebskosten."""

    def __init__(self, w_comfort=1.0, **kwargs):
        super().__init__(**kwargs)
        self.w_comfort = w_comfort

    def get_reward(self):
        kpis = requests.get(f'{self.url}/kpi/{self.testid}').json()['payload']
        objective_integrand = kpis['cost_tot'] + self.w_comfort * kpis['tdis_tot']
        reward = -(objective_integrand - self.objective_integrand)
        self.objective_integrand = objective_integrand
        return reward


class NormalizedObservationWrapper(gym.ObservationWrapper):
    """Normalisiert Beobachtungen auf [-1, 1] — hilft SB3-Algorithmen beim Konvergieren."""

    def observation(self, observation):
        return 2 * (observation - self.observation_space.low) / \
            (self.observation_space.high - self.observation_space.low) - 1


class NormalizedActionWrapper(gym.ActionWrapper):
    """Normalisiert den Aktionsraum auf [-1, 1] (SAC/TD3 gehen von symmetrischen,
    beschränkten Aktionsräumen aus)."""

    def __init__(self, env):
        super().__init__(env)
        self.low = self.unwrapped.action_space.low
        self.high = self.unwrapped.action_space.high
        self.action_space = spaces.Box(low=-1, high=1, shape=self.unwrapped.action_space.shape, dtype=np.float32)

    def action(self, action_wrapper):
        return self.low + (0.5 * (action_wrapper + 1.0) * (self.high - self.low))
