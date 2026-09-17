'use client';

import React from 'react';
import {
  Zap,
  Activity,
  Thermometer,
  CloudSun,
  Calendar,
  IndianRupee,
} from 'lucide-react';

import {
  HomeState,
  PeakEvent,
  ActionLevel,
  CommandAck,
} from '@/lib/contracts';

import { StatCard } from './StatCard';
import { PeakEventCard } from './PeakEventCard';
import { PowerChart } from './PowerChart';
import { Panel4Appliances } from './Panel4Appliances';
import { MetricsPanel } from './MetricsPanel';
import { LoadProfilePoint } from '@/lib/mockState';

interface ResidentViewProps {
  homeState: HomeState;
  peakEvent: PeakEvent;
  chartData: LoadProfilePoint[];
  onOverride: (
    applianceId: string,
    level: ActionLevel
  ) => Promise<CommandAck>;
  lastAck: CommandAck | null;
  onClearAck: () => void;
}

export function ResidentView({
  homeState,
  peakEvent,
  chartData,
  onOverride,
  lastAck,
  onClearAck,
}: ResidentViewProps) {
  const isPeakActive =
    homeState.grid_peak_severity >= 0.4 ||
    peakEvent.is_active;

  const loadPercentage =
    (homeState.aggregate_power_kw /
      homeState.sanctioned_load_kw) *
    100;

  return (
    <div className="space-y-6">

      {/* =========================================================
          1. TOP SUMMARY METRICS
      ========================================================= */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">

        <StatCard
          title="Current Power"
          value={homeState.aggregate_power_kw.toFixed(3)}
          unit="kW"
          subtext={`Cap: ${homeState.sanctioned_load_kw.toFixed(1)} kW`}
          icon={Zap}
          variant="cyan"
          badge={{
            text: `${loadPercentage.toFixed(0)}% LOAD`,
            type: 'neutral',
          }}
        />

        <StatCard
          title="Grid Severity"
          value={homeState.grid_peak_severity.toFixed(2)}
          subtext={
            isPeakActive
              ? 'Midday Peak Window'
              : 'Normal Grid Baseline'
          }
          icon={Activity}
          variant="cyan"
          badge={{
            text: isPeakActive ? 'PEAK ACTIVE' : 'RELAXED',
            type: 'neutral',
          }}
        />

        <StatCard
          title="Indoor Temp"
          value={homeState.indoor_temperature_c.toFixed(1)}
          unit="°C"
          subtext="Comfort band: 24–28 °C"
          icon={Thermometer}
          variant="default"
        />

        <StatCard
          title="Outdoor Temp"
          value={homeState.outdoor_temperature_c.toFixed(1)}
          unit="°C"
          subtext="Guntur Weather Stn"
          icon={CloudSun}
          variant="default"
        />

        <StatCard
          title="Month Energy"
          value={homeState.month_to_date_kwh.toFixed(1)}
          unit="kWh"
          subtext="Today: ~8.42 kWh"
          icon={Calendar}
          variant="default"
        />

        <StatCard
          title="Marginal Tariff"
          value={`₹${homeState.marginal_tariff_inr_kwh.toFixed(2)}`}
          unit="/kWh"
          subtext="APCPDCL Telescopic"
          icon={IndianRupee}
          variant="default"
          badge={{
            text: 'SLAB 2',
            type: 'neutral',
          }}
        />
      </div>

      {/* =========================================================
          2. PEAK EVENT
      ========================================================= */}
      <PeakEventCard peakEvent={peakEvent} />

      {/* =========================================================
          3. POWER PROFILE
      ========================================================= */}
      <PowerChart
        data={chartData}
        sanctionedLoadKw={homeState.sanctioned_load_kw}
      />

      {/* =========================================================
          4. APPLIANCES
      ========================================================= */}
      <Panel4Appliances
        appliances={homeState.appliances}
        onOverride={onOverride}
        lastAck={lastAck}
        onClearAck={onClearAck}
      />

      {/* =========================================================
          5. PERFORMANCE METRICS
      ========================================================= */}
      <MetricsPanel />
    </div>
  );
}