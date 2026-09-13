import sys
from pathlib import Path
import unittest
from decimal import Decimal as D
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from sharp_apcpdcl_tariff import Tariff, validate_examples

class TariffTests(unittest.TestCase):
    def setUp(self):
        self.t = Tariff.load()

    def test_hand_calculations(self):
        self.assertEqual(validate_examples(self.t)['status'], 'PASS')

    def test_customer_jump(self):
        r = self.t.incremental_components(30, '0.1')
        self.assertEqual(r['energy_inr'], D('0.3'))
        self.assertEqual(r['customer_inr'], D('5'))

    def test_zero_step(self):
        self.assertEqual(self.t.incremental_components(75, 0)['tariff_increment_inr'], 0)

    def test_fractional_power_units(self):
        # 1000 W for 15 minutes = 0.25 kWh; first slab = Rs 0.475.
        self.assertEqual(self.t.incremental_components(0, '0.25')['energy_inr'], D('0.475'))

    def test_no_fake_tod_savings(self):
        # Changing chronological order of identical import amounts within a billing period
        # cannot change the total under the verified non-ToD tariff.
        def total(sequence):
            q = D(0); cost = D(0)
            for amount in sequence:
                r = self.t.incremental_components(q, amount)
                q = r['month_kwh_after']; cost += r['tariff_increment_inr']
            return cost
        self.assertEqual(total([10, 90, 300]), total([300, 90, 10]))

    def test_reject_invalid(self):
        for value in [-1, 'NaN', 'Infinity', True, None]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.t.energy_charge(value)
        with self.assertRaises(ValueError):
            self.t.fixed_charge(0)

    def test_highest_slab(self):
        self.assertEqual(self.t.incremental_components(500, 1)['energy_inr'], D('9.75'))

if __name__ == '__main__':
    unittest.main()
