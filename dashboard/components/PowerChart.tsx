'use client';

import React from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
} from 'recharts';

import { LoadProfilePoint } from '@/lib/mockState';

interface PowerChartProps {
  data: LoadProfilePoint[];
  sanctionedLoadKw?: number;
}

export function PowerChart({
  data,
  sanctionedLoadKw = 1.0,
}: PowerChartProps) {
  return (
    <div className="rounded-2xl border border-blue-100 bg-white p-5 shadow-sm">

      {/* Header */}
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="font-mono text-sm font-bold tracking-wide text-[#172b4d]">
            24-HOUR HOUSEHOLD LOAD &amp; GRID PROFILE (96 STEPS)
          </h3>

          <p className="text-xs text-slate-500">
            15-minute simulated intervals. Midday grid peak
            (08:00–17:00) with SHARP flexible load reduction.
          </p>
        </div>

        {/* Legend */}
        <div className="flex flex-wrap items-center gap-4 text-xs font-mono">
          <div className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-blue-600" />
            <span className="text-slate-600">
              SHARP Managed kW
            </span>
          </div>

          <div className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-blue-300" />
            <span className="text-slate-500">
              Uncontrolled kW
            </span>
          </div>

          <div className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-blue-800" />
            <span className="text-slate-500">
              Grid Peak Severity
            </span>
          </div>
        </div>
      </div>

      {/* Chart */}
      <div className="h-72 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            data={data}
            margin={{
              top: 10,
              right: 10,
              left: -20,
              bottom: 0,
            }}
          >
            <defs>
              <linearGradient
                id="sharpGradient"
                x1="0"
                y1="0"
                x2="0"
                y2="1"
              >
                <stop
                  offset="5%"
                  stopColor="#2563eb"
                  stopOpacity={0.18}
                />
                <stop
                  offset="95%"
                  stopColor="#2563eb"
                  stopOpacity={0}
                />
              </linearGradient>

              <linearGradient
                id="peakGradient"
                x1="0"
                y1="0"
                x2="0"
                y2="1"
              >
                <stop
                  offset="5%"
                  stopColor="#1d4ed8"
                  stopOpacity={0.14}
                />
                <stop
                  offset="95%"
                  stopColor="#1d4ed8"
                  stopOpacity={0}
                />
              </linearGradient>
            </defs>

            <XAxis
              dataKey="time"
              stroke="#94a3b8"
              tick={{
                fill: '#64748b',
                fontSize: 11,
                fontFamily: 'monospace',
              }}
              interval={11}
            />

            <YAxis
              yAxisId="power"
              stroke="#94a3b8"
              tick={{
                fill: '#64748b',
                fontSize: 11,
                fontFamily: 'monospace',
              }}
              domain={[0, 1.2]}
              unit=" kW"
            />

            <YAxis
              yAxisId="severity"
              orientation="right"
              stroke="#1d4ed8"
              tick={{
                fill: '#1e40af',
                fontSize: 10,
                fontFamily: 'monospace',
              }}
              domain={[0, 1.0]}
              hide={true}
            />

            <Tooltip
              content={({ active, payload, label }) => {
                if (active && payload && payload.length) {
                  return (
                    <div className="rounded-xl border border-blue-100 bg-white p-3 shadow-lg text-xs font-mono">
                      <p className="mb-1.5 font-bold text-[#172b4d]">
                        {label} IST
                      </p>

                      <div className="space-y-1">
                        <p className="text-blue-600">
                          SHARP Load:{' '}
                          <strong>
                            {payload[0]?.value} kW
                          </strong>
                        </p>

                        <p className="text-blue-400">
                          Baseline Load:{' '}
                          <strong>
                            {payload[1]?.value} kW
                          </strong>
                        </p>

                        <p className="text-blue-800">
                          Grid Severity:{' '}
                          <strong>
                            {payload[2]?.value}
                          </strong>
                        </p>
                      </div>
                    </div>
                  );
                }

                return null;
              }}
            />

            {/* Sanctioned Load */}
            <ReferenceLine
              yAxisId="power"
              y={sanctionedLoadKw}
              stroke="#1d4ed8"
              strokeDasharray="4 4"
              label={{
                value: `Sanctioned Cap (${sanctionedLoadKw} kW)`,
                fill: '#1e40af',
                fontSize: 10,
                position: 'insideTopRight',
              }}
            />

            {/* Background Load */}
            <ReferenceLine
              yAxisId="power"
              y={0.084}
              stroke="#94a3b8"
              strokeDasharray="2 2"
              label={{
                value: 'Uncontrollable Base (0.084 kW)',
                fill: '#64748b',
                fontSize: 9,
                position: 'insideBottomLeft',
              }}
            />

            {/* Grid Peak Severity */}
            <Area
              yAxisId="severity"
              type="monotone"
              dataKey="gridPeakSeverity"
              fill="url(#peakGradient)"
              stroke="#1d4ed8"
              strokeWidth={1.5}
            />

            {/* Baseline */}
            <Line
              yAxisId="power"
              type="monotone"
              dataKey="baselineKw"
              stroke="#93c5fd"
              strokeWidth={1.5}
              strokeDasharray="3 3"
              dot={false}
            />

            {/* SHARP Controlled Load */}
            <Area
              yAxisId="power"
              type="monotone"
              dataKey="sharpKw"
              fill="url(#sharpGradient)"
              stroke="#2563eb"
              strokeWidth={2}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}