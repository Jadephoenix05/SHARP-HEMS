'use client';

import React from 'react';
import {
  TrendingDown,
  Gauge,
  Scale,
  AlertCircle,
  Award,
} from 'lucide-react';

export function MetricsPanel() {
  return (
    <div className="space-y-6">

      {/* =====================================================
          PROJECT HEADLINE
      ===================================================== */}
      <div className="rounded-2xl border border-blue-100 bg-gradient-to-r from-blue-50 via-white to-sky-50 p-5 shadow-sm">

        <div className="flex items-start gap-3.5">

          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-blue-100 bg-white text-blue-600 shadow-sm">
            <Award className="h-5 w-5" />
          </div>

          <div className="space-y-1">

            <h4 className="text-sm font-bold uppercase tracking-wide text-blue-800">
              The Project Headline: Equity, Not Mere Bill Arbitrage
            </h4>

            <blockquote className="my-1 border-l-2 border-blue-300 pl-3 text-xs italic leading-relaxed text-slate-600">
              &ldquo;We do not choose which village goes dark.
              We take the television from everyone so that
              nobody goes dark.&rdquo;
            </blockquote>

            <p className="text-[11px] leading-normal text-slate-500">
              Evaluated on 40 held-out household days. SHARP
              cuts household peak demand, delivering the
              equivalent relief of keeping{' '}
              <strong className="text-[#172b4d]">
                4.7 homes in 100
              </strong>{' '}
              from being blacked out—without disconnecting any
              household.
            </p>

          </div>
        </div>
      </div>

      {/* =====================================================
          PRIMARY METRICS
      ===================================================== */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">

        {/* Peak Avoidance */}
        <div className="rounded-2xl border border-blue-200 bg-white p-5 shadow-sm">

          <div className="flex items-center justify-between">

            <div className="flex items-center gap-2">

              <div className="flex h-8 w-8 items-center justify-center rounded-lg border border-blue-100 bg-blue-50 text-blue-600">
                <TrendingDown className="h-4 w-4" />
              </div>

              <span className="font-mono text-xs font-bold uppercase tracking-wider text-[#172b4d]">
                PEAK AVOIDANCE
              </span>
            </div>

            <span className="rounded-full border border-blue-100 bg-blue-50 px-2.5 py-0.5 font-mono text-xs font-semibold text-blue-700">
              Primary KPI
            </span>
          </div>

          <div className="mt-4 flex items-baseline gap-2">
            <span className="font-mono text-4xl font-extrabold text-blue-600">
              18.4%
            </span>

            <span className="text-xs text-slate-500">
              midday window reduction (4.65% daily aggregate)
            </span>
          </div>

          <p className="mt-2 text-xs leading-relaxed text-slate-500">
            Feeder-level stress relief achieved by coordinated
            discretionary load pausing. Baseline peak{' '}
            <strong className="text-[#172b4d]">
              0.408 kW
            </strong>{' '}
            reduced to{' '}
            <strong className="text-[#172b4d]">
              0.379 kW
            </strong>
            .
          </p>

          <div className="mt-4 rounded-lg border border-blue-100 bg-blue-50 p-2.5 text-xs">
            <div className="flex items-center justify-between text-slate-600">
              <span>Blackout Relief Equivalence:</span>

              <strong className="font-mono text-blue-700">
                4.7 homes / 100
              </strong>
            </div>
          </div>
        </div>

        {/* PAR */}
        <div className="rounded-2xl border border-blue-200 bg-white p-5 shadow-sm">

          <div className="flex items-center justify-between">

            <div className="flex items-center gap-2">

              <div className="flex h-8 w-8 items-center justify-center rounded-lg border border-blue-100 bg-blue-50 text-blue-600">
                <Gauge className="h-4 w-4" />
              </div>

              <span className="font-mono text-xs font-bold uppercase tracking-wider text-[#172b4d]">
                PEAK-TO-AVERAGE RATIO (PAR)
              </span>
            </div>

            <span className="rounded-full border border-blue-100 bg-blue-50 px-2.5 py-0.5 font-mono text-xs font-semibold text-blue-700">
              Grid Stability
            </span>
          </div>

          <div className="mt-4 flex items-baseline gap-2">
            <span className="font-mono text-4xl font-extrabold text-blue-600">
              2.558
            </span>

            <span className="font-mono text-xs text-blue-700">
              -8.7% vs rule-based (2.803)
            </span>
          </div>

          <p className="mt-2 text-xs leading-relaxed text-slate-500">
            Measures load curve smoothness. Lower PAR prevents
            distribution transformer overload. Rule-based
            shedding creates rebound spikes, whereas SHARP
            smooths recovery.
          </p>

          <div className="mt-4 rounded-lg border border-blue-100 bg-blue-50/50 p-2.5 text-xs">
            <div className="flex items-center justify-between text-slate-600">
              <span>Uncontrolled Baseline PAR:</span>

              <span className="font-mono text-slate-700">
                2.593
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* =====================================================
          BENCHMARK TABLE
      ===================================================== */}
      <div className="rounded-2xl border border-blue-100 bg-white p-5 shadow-sm">

        <div className="mb-3 flex items-center justify-between">

          <div className="flex items-center gap-2">
            <Scale className="h-4 w-4 text-blue-600" />

            <h4 className="font-mono text-xs font-bold uppercase tracking-wider text-[#172b4d]">
              Policy Benchmark (40 Held-Out Household-Days)
            </h4>
          </div>

          <span className="text-[11px] font-mono text-slate-400">
            APCPDCL Tariff LT-I
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">

            <thead>
              <tr className="border-b border-blue-100 text-slate-500">
                <th className="px-3 py-2 font-semibold">
                  Evaluation Metric
                </th>

                <th className="px-3 py-2 font-semibold">
                  No Control
                </th>

                <th className="px-3 py-2 font-semibold">
                  Rule-Based
                </th>

                <th className="rounded-t bg-blue-50 px-3 py-2 font-semibold text-blue-700">
                  SHARP (Ours)
                </th>
              </tr>
            </thead>

            <tbody className="divide-y divide-blue-50 text-slate-600">

              <tr>
                <td className="px-3 py-2.5 font-sans font-medium text-[#172b4d]">
                  Peak Demand (kW)
                </td>

                <td className="px-3 py-2.5">
                  0.408
                </td>

                <td className="px-3 py-2.5 text-blue-500">
                  0.409
                </td>

                <td className="bg-blue-50/70 px-3 py-2.5 font-bold text-blue-700">
                  0.379{' '}
                  <span className="text-[10px] text-blue-600">
                    (-7.1%)
                  </span>
                </td>
              </tr>

              <tr>
                <td className="px-3 py-2.5 font-sans font-medium text-[#172b4d]">
                  Peak-to-Average Ratio
                </td>

                <td className="px-3 py-2.5">
                  2.593
                </td>

                <td className="px-3 py-2.5 text-blue-500">
                  2.803 (Worse)
                </td>

                <td className="bg-blue-50/70 px-3 py-2.5 font-bold text-blue-700">
                  2.558{' '}
                  <span className="text-[10px] text-blue-600">
                    (Best)
                  </span>
                </td>
              </tr>

              <tr>
                <td className="px-3 py-2.5 font-sans font-medium text-[#172b4d]">
                  Daily Cost (₹/day)
                </td>

                <td className="px-3 py-2.5">
                  ₹19.18
                </td>

                <td className="px-3 py-2.5 font-bold text-blue-600">
                  ₹18.06
                </td>

                <td className="bg-blue-50/70 px-3 py-2.5 font-semibold text-[#172b4d]">
                  ₹19.00
                </td>
              </tr>

            </tbody>
          </table>
        </div>

        {/* Tariff Honesty */}
        <div className="mt-3 flex items-start gap-2 rounded-lg border border-blue-100 bg-blue-50/60 p-3 text-[11px] text-slate-600">

          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-blue-600" />

          <p>
            <strong className="text-[#172b4d]">
              Tariff Honesty:
            </strong>{' '}
            APCPDCL LT-I domestic tariff is telescopic based
            on total monthly unit slabs with{' '}
            <em>no time-of-day (ToD) tariff</em>. Time-shifting
            load does not reduce the monthly bill significantly.
            The rule-based controller wins on pure cost by
            blindly cutting loads, but causes severe rebound
            peaks. SHARP preserves citizen comfort while
            minimizing peak strain.
          </p>
        </div>
      </div>
    </div>
  );
}