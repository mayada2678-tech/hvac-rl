"""Baseline zum Vergleich mit dem RL-Agenten: BOPTESTs eingebauter Regler (Rule-Based
Controller). Jeder BOPTEST-Testfall bringt bereits eine Referenzregelung mit — sie wird
aktiv, sobald keine Aktion überschrieben wird. Ein fairer RBC-Vergleich braucht deshalb keine
eigene Regel-Implementierung, nur eine leere Aktionsliste.
"""
import numpy as np


class RuleBasedController:
    """Überschreibt keine Aktion — lässt BOPTESTs eingebauten Regler laufen. So vergleicht
    man den RL-Agenten fair gegen die im Testfall hinterlegte Referenzregelung."""

    def __init__(self, env):
        self.env = env
        self.env.unwrapped.actions = []   # keine Überschreibung -> eingebauter Regler steuert

    def predict(self, obs=None, deterministic=True):
        return np.array([]), obs


def build(name: str, env):
    """'rbc' = BOPTESTs eingebauter Regler (kein 'none'-Fall wie bei CityLearn: BOPTEST
    braucht immer irgendeine Regelgröße für Ventile/Pumpen — der eingebaute Regler *ist*
    hier die Referenz ohne RL)."""
    return {'rbc': RuleBasedController}[name](env)
