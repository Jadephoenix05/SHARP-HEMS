"""Synthetic human attention and override behaviour for SHARP.

SHARP stands for Shielded Human-override Adaptive Reward Personalisation. The
override signal is the project's core evidence: an override is the user telling
the controller it was wrong, in a specific context. Without overrides there is
nothing for a learned preference reward to fit.

EVERYTHING IN THIS MODULE IS SYNTHETIC. No SHARP source dataset records a user
overriding a demand-response action, because no such deployment has happened
yet. What this module does is generate override evidence from a stated,
inspectable behavioural rule so the preference-learning pipeline has structured
data to develop against. It must never be described as observed user behaviour.

The behavioural rule, stated plainly:

  * A user can only override what they notice. Attention comes from the TUS
    adult-location proxy: an adult reported at home. That proxy is one adult,
    not the whole household, so attention is itself a documented approximation.

  * A user overrides when the controller denied something they wanted AND they
    were present. Pressure to override rises with:
      - thermal discomfort, for an air conditioner denied while the room is
        above the comfort band;
      - unmet service, for any device denied during a preferred service slot,
        rising as the remaining daily service budget goes unserved.

  * Overrides are deterministic given the episode, step and device, so the
    dataset is reproducible. The randomness is a seeded draw, not a simulation
    of individual psychology.

  * An override REQUESTS an action. The shield still decides whether to honour
    it, because a safety shield that can be overridden by a user is not a
    safety shield. Refused overrides are preserved as evidence.
"""
from dataclasses import dataclass
import hashlib
import math

# Behavioural parameters. Declared assumptions, not fitted to any source.
ATTENTION_HOME_THRESHOLD = 0.5      # adult-home proxy above this means present
BASE_OVERRIDE_PROBABILITY = 0.05    # denied and present, but otherwise content
DISCOMFORT_SENSITIVITY = 0.15       # added probability per degree outside band
UNMET_SERVICE_SENSITIVITY = 0.10    # added probability per remaining hour
MAXIMUM_OVERRIDE_PROBABILITY = 0.85  # people do not always act, even when annoyed
RESPONSE_WINDOW_STEPS = 1           # a user gets one interval to react


@dataclass(frozen=True)
class OverrideEvent:
    device_id: str
    step_id: int
    requested_on: int
    probability: float
    pressure_source: str
    degrees_outside_band: float
    remaining_service_hours: float
    attention_available: bool
    latency_steps: int
    is_synthetic: bool = True


def _draw(episode_id, step_id, device_id):
    """Deterministic uniform draw in [0, 1) for one device at one step."""
    key = f'{episode_id}|{step_id}|{device_id}'.encode()
    return int(hashlib.sha256(key).hexdigest()[:8], 16) / 0x100000000


def attention_available(home_fraction):
    """Whether a user was present enough to notice an intervention."""
    value = float(home_fraction)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Adult-home fraction must be a finite value in [0, 1]')
    return value > ATTENTION_HOME_THRESHOLD


def override_probability(*, denied, present, degrees_outside_band,
                         remaining_service_hours, is_air_conditioner):
    """Probability that a present user overrides a denied action."""
    if not denied or not present:
        return 0.0
    for name, value in [('degrees_outside_band', degrees_outside_band),
                        ('remaining_service_hours', remaining_service_hours)]:
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    probability = BASE_OVERRIDE_PROBABILITY
    if is_air_conditioner:
        probability += DISCOMFORT_SENSITIVITY * float(degrees_outside_band)
    else:
        probability += UNMET_SERVICE_SENSITIVITY * float(remaining_service_hours)
    return min(MAXIMUM_OVERRIDE_PROBABILITY, probability)


def decide_overrides(*, episode_id, step_id, device_ids, wanted, executed,
                     home_fraction, degrees_outside_band, remaining_service_hours,
                     is_air_conditioner, protected):
    """Return {device_id: 1} override requests plus the evidence behind them.

    An override is only generated where the user wanted the device ON and the
    controller left it OFF. Protected devices are never shed, so they never
    produce an override.
    """
    present = attention_available(home_fraction)
    requests, events = {}, []
    for index, device_id in enumerate(device_ids):
        if protected[index]:
            continue
        denied = bool(wanted[index]) and not bool(executed[index])
        probability = override_probability(
            denied=denied, present=present,
            degrees_outside_band=degrees_outside_band if is_air_conditioner[index] else 0.0,
            remaining_service_hours=remaining_service_hours[index],
            is_air_conditioner=bool(is_air_conditioner[index]))
        if probability <= 0:
            continue
        if _draw(episode_id, step_id, device_id) < probability:
            requests[device_id] = 1
            events.append(OverrideEvent(
                device_id=device_id, step_id=step_id, requested_on=1,
                probability=probability,
                pressure_source=('THERMAL_DISCOMFORT' if is_air_conditioner[index]
                                 else 'UNMET_SERVICE'),
                degrees_outside_band=float(degrees_outside_band),
                remaining_service_hours=float(remaining_service_hours[index]),
                attention_available=present, latency_steps=0))
    return requests, events, present


def self_test():
    """Behavioural invariants. These are properties of the rule, not of users."""
    # No attention means no override, however uncomfortable.
    requests, events, present = decide_overrides(
        episode_id='t', step_id=0, device_ids=['a'], wanted=[1], executed=[0],
        home_fraction=0.0, degrees_outside_band=10.0,
        remaining_service_hours=[5.0], is_air_conditioner=[True], protected=[False])
    assert not requests and not events and not present

    # A protected device is never shed, so it never generates an override.
    requests, _, _ = decide_overrides(
        episode_id='t', step_id=0, device_ids=['a'], wanted=[1], executed=[0],
        home_fraction=1.0, degrees_outside_band=10.0,
        remaining_service_hours=[5.0], is_air_conditioner=[False], protected=[True])
    assert not requests

    # Getting what you wanted produces no override.
    requests, _, _ = decide_overrides(
        episode_id='t', step_id=0, device_ids=['a'], wanted=[1], executed=[1],
        home_fraction=1.0, degrees_outside_band=10.0,
        remaining_service_hours=[5.0], is_air_conditioner=[True], protected=[False])
    assert not requests

    # Pressure is monotonic in discomfort and bounded.
    low = override_probability(denied=True, present=True, degrees_outside_band=0.5,
                               remaining_service_hours=0, is_air_conditioner=True)
    high = override_probability(denied=True, present=True, degrees_outside_band=4.0,
                                remaining_service_hours=0, is_air_conditioner=True)
    assert 0 < low < high <= MAXIMUM_OVERRIDE_PROBABILITY

    # Determinism: the same context always gives the same decision.
    first = decide_overrides(
        episode_id='e', step_id=7, device_ids=['d1', 'd2'], wanted=[1, 1],
        executed=[0, 0], home_fraction=1.0, degrees_outside_band=3.0,
        remaining_service_hours=[2.0, 2.0], is_air_conditioner=[True, False],
        protected=[False, False])[0]
    for _ in range(20):
        assert decide_overrides(
            episode_id='e', step_id=7, device_ids=['d1', 'd2'], wanted=[1, 1],
            executed=[0, 0], home_fraction=1.0, degrees_outside_band=3.0,
            remaining_service_hours=[2.0, 2.0], is_air_conditioner=[True, False],
            protected=[False, False])[0] == first

    # Over many contexts the override rate sits strictly between never and always.
    fired = sum(bool(decide_overrides(
        episode_id=f'e{i}', step_id=i % 96, device_ids=['d'], wanted=[1],
        executed=[0], home_fraction=1.0, degrees_outside_band=2.0,
        remaining_service_hours=[1.0], is_air_conditioner=[True],
        protected=[False])[0]) for i in range(2000))
    assert 0 < fired < 2000, f'Degenerate override rate: {fired}/2000'
    print('SHARP HUMAN MODEL SELF-TEST PASSED')
    print(f'  override rate at 2 C outside band, present: {fired / 2000:.3f}')
    print('  all override evidence is SYNTHETIC and must be labelled as such')


if __name__ == '__main__':
    self_test()
