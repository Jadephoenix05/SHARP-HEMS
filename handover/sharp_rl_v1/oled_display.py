"""OLED sequence and buzzer for the SHARP rig. SSD1306 128x64 over I2C.

A display that never changes stops being read after two minutes. This runs a
small state machine instead, so the panel always shows the thing that is
currently true, and the buzzer marks the one moment that matters.

    BOOT ........ welcome, then the self-test result
    NORMAL ...... rotates: live load -> supply -> appliance grid
    PEAK_ALERT .. buzzer, full-screen banner, three seconds
    PEAK_ACTIVE . what is protected and what is shed, held while the peak lasts
    RECOVER ..... "peak over", two seconds, back to NORMAL

The buzzer sounds on the TRANSITION into a peak, not while the peak lasts. A
buzzer that runs for forty minutes gets disconnected, and then it is not there
for the one event it existed to announce.

Hardware: SSD1306 on BCM 2/3 (I2C), buzzer on BCM 18.
    pip install adafruit-circuitpython-ssd1306 pillow
"""
import time

try:
    import board, busio, digitalio
    import adafruit_ssd1306
    from PIL import Image, ImageDraw, ImageFont
    HARDWARE = True
except ImportError:          # so the sequence can be tested on a laptop
    HARDWARE = False

WIDTH, HEIGHT = 128, 64
LINE = 8                     # default font is 6x8, so eight lines of 21 chars
BUZZER_PIN_NAME = 'D18'

ROTATE_SECONDS = 4.0         # how long each NORMAL screen is held
ALERT_SECONDS = 3.0
RECOVER_SECONDS = 2.0
PEAK_THRESHOLD = 0.5


class Panel:
    """Draws text lines. Falls back to stdout when there is no display attached."""

    def __init__(self):
        self.display = None
        self.buzzer = None
        if HARDWARE:
            i2c = busio.I2C(board.SCL, board.SDA)
            self.display = adafruit_ssd1306.SSD1306_I2C(WIDTH, HEIGHT, i2c)
            self.font = ImageFont.load_default()
            self.buzzer = digitalio.DigitalInOut(getattr(board, BUZZER_PIN_NAME))
            self.buzzer.direction = digitalio.Direction.OUTPUT
            self.buzzer.value = False

    def show(self, lines, invert=False):
        lines = [str(line)[:21] for line in lines[:8]]
        if not HARDWARE:
            print('+' + '-' * 21 + '+')
            for line in lines:
                print('|' + line.ljust(21) + '|')
            print('+' + '-' * 21 + '+')
            return
        image = Image.new('1', (WIDTH, HEIGHT), 1 if invert else 0)
        draw = ImageDraw.Draw(image)
        for index, line in enumerate(lines):
            draw.text((0, index * LINE), line, font=self.font,
                      fill=0 if invert else 1)
        self.display.image(image)
        self.display.show()

    def beep(self, times=3, on=0.12, off=0.10):
        """Short pattern on the TRANSITION into a peak. Never a continuous tone."""
        if not HARDWARE:
            print(f'    *** BUZZER x{times} ***')
            return
        for _ in range(times):
            self.buzzer.value = True
            time.sleep(on)
            self.buzzer.value = False
            time.sleep(off)


def screen_welcome():
    return ['', '      S H A R P', '', ' Shielded Human-',
            ' override Adaptive', ' Reward Personal.', '', '   starting up...']


def screen_selftest(passed, features, branches):
    return ['SELF TEST', '-' * 21,
            f'golden vector  {"OK" if passed else "FAIL"}',
            f'features       {features}',
            f'branches       {branches}', '',
            'ready' if passed else 'DO NOT ACTUATE']


def screen_load(state):
    bar = int(min(1.0, state['aggregate_kw'] / max(0.01, state['sanctioned_kw'])) * 20)
    return ['LIVE LOAD', '-' * 21,
            f"{state['aggregate_kw']:.3f} kW",
            '[' + '#' * bar + '.' * (20 - bar) + ']',
            f"limit   {state['sanctioned_kw']:.2f} kW",
            f"today   {state['today_kwh']:.2f} kWh",
            f"tariff  Rs {state['tariff']:.2f}/kWh"]


def screen_supply(state):
    return ['SUPPLY', '-' * 21,
            f"voltage   {state['voltage_v']:.0f} V",
            f"current   {state['current_a']:.2f} A",
            f"grid      {'ON' if not state['grid_absent'] else 'OUTAGE'}",
            f"severity  {state['severity']:.2f}",
            f"mode      {state['mode']}"]


def screen_appliances(appliances):
    """Two columns, protected marked. Pages if there are more than ten."""
    rows = ['APPLIANCES', '-' * 21]
    items = [(name, info) for name, info in appliances.items()]
    for left, right in zip(items[0::2], items[1::2] + [(None, None)]):
        cells = []
        for name, info in (left, right):
            if name is None:
                cells.append(' ' * 10)
                continue
            mark = '*' if info['protected'] else ' '
            cells.append(f"{name[:5]:5s}{info['state'][:3]:>4s}{mark}")
        rows.append(''.join(cells)[:21])
    return rows[:8]


def screen_peak_alert(severity):
    return ['', '   ! GRID PEAK !', '', f'   severity {severity:.2f}', '',
            '  luxury loads will', '   pause shortly']


def screen_peak_active(appliances, relieved):
    protected = [n for n, i in appliances.items() if i['protected']]
    shed = [n for n, i in appliances.items() if i['state'] == 'SHED']
    return ['PEAK ACTIVE', '-' * 21,
            'ON  ' + ' '.join(protected)[:17],
            '    ' + ' '.join(protected[3:])[:17] if len(protected) > 3 else '',
            'OFF ' + ' '.join(shed)[:17],
            '-' * 21,
            f'relieved {relieved:.1f}/100']


def screen_recover():
    return ['', '   peak over', '', '  luxury loads', '  available again']


class DisplaySequence:
    """Drives the panel from whatever the controller last published."""

    def __init__(self, panel):
        self.panel = panel
        self.mode = 'BOOT'
        self.since = time.monotonic()
        self.rotation = 0
        self.was_peak = False

    def _enter(self, mode):
        self.mode = mode
        self.since = time.monotonic()

    def tick(self, state, appliances, relieved):
        """Call about once a second with the latest state."""
        now = time.monotonic()
        elapsed = now - self.since
        is_peak = state['severity'] >= PEAK_THRESHOLD

        # A peak starting is the one event worth interrupting for.
        if is_peak and not self.was_peak:
            self._enter('PEAK_ALERT')
            self.panel.beep()
        elif not is_peak and self.was_peak:
            self._enter('RECOVER')
        self.was_peak = is_peak

        if self.mode == 'BOOT':
            if elapsed < 3:
                self.panel.show(screen_welcome())
            else:
                self._enter('NORMAL')
            return
        if self.mode == 'PEAK_ALERT':
            self.panel.show(screen_peak_alert(state['severity']), invert=True)
            if elapsed >= ALERT_SECONDS:
                self._enter('PEAK_ACTIVE')
            return
        if self.mode == 'PEAK_ACTIVE':
            self.panel.show(screen_peak_active(appliances, relieved))
            return
        if self.mode == 'RECOVER':
            self.panel.show(screen_recover())
            if elapsed >= RECOVER_SECONDS:
                self._enter('NORMAL')
            return

        # NORMAL: rotate through the three informational screens.
        screens = [screen_load(state), screen_supply(state),
                   screen_appliances(appliances)]
        self.rotation = int(elapsed // ROTATE_SECONDS) % len(screens)
        self.panel.show(screens[self.rotation])


if __name__ == '__main__':
    # Rehearsal with no hardware: prints the panel to the terminal so the
    # sequence and the timing can be checked before anything is wired.
    panel = Panel()
    sequence = DisplaySequence(panel)
    appliances = {
        'fan1': {'state': 'ON', 'protected': True},
        'fan2': {'state': 'ON', 'protected': True},
        'lamp1': {'state': 'ON', 'protected': True},
        'fridg': {'state': 'ON', 'protected': True},
        'tv': {'state': 'ON', 'protected': False},
        'mixer': {'state': 'ON', 'protected': False},
        'pump': {'state': 'ON', 'protected': False},
        'ev': {'state': 'WAIT', 'protected': False},
    }
    state = {'aggregate_kw': 0.42, 'sanctioned_kw': 2.0, 'today_kwh': 3.1,
             'tariff': 4.50, 'voltage_v': 231, 'current_a': 1.8,
             'grid_absent': False, 'severity': 0.0, 'mode': 'grid_import'}

    print('=== boot, then normal rotation ===')
    for second in range(14):
        sequence.tick(state, appliances, 0.0)
        time.sleep(0.25)

    print('\n=== grid controller declares a peak ===')
    state['severity'] = 0.67
    for name in ('tv', 'mixer', 'pump'):
        appliances[name]['state'] = 'SHED'
    for second in range(6):
        sequence.tick(state, appliances, 4.7)
        time.sleep(0.25)

    print('\n=== peak clears ===')
    state['severity'] = 0.0
    for name in ('tv', 'mixer', 'pump'):
        appliances[name]['state'] = 'ON'
    for second in range(4):
        sequence.tick(state, appliances, 0.0)
        time.sleep(0.25)
