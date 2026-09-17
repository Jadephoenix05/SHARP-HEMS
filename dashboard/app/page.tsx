
'use client';

import React, { useState, useMemo } from 'react';
import { useHomeState } from '@/lib/useHomeState';
import { generateDailyLoadProfile } from '@/lib/mockState';
import { DashboardHeader } from '@/components/DashboardHeader';
import { SystemStatus } from '@/components/SystemStatus';
import { ResidentView } from '@/components/ResidentView';
import { GridController } from '@/components/GridController';

export default function DashboardPage() {
  const [activeTab, setActiveTab] = useState<'resident' | 'discom'>('resident');

  const {
    homeState,
    peakEvent,
    systemStatus,
    activeScenario,
    setScenario,
    triggerOverride,
    declarePeakEvent,
    cancelPeakEvent,
    lastAck,
    clearLastAck,
  } = useHomeState();

  const chartData = useMemo(() => generateDailyLoadProfile(), []);

  return (
    <div className="min-h-screen bg-[#f5f9ff] bg-grid-pattern text-[#172b4d] selection:bg-blue-100 selection:text-blue-900">
      {/* 1. TOP HEADER & NAVIGATION */}
      <DashboardHeader
        activeTab={activeTab}
        onTabChange={setActiveTab}
        activeScenario={activeScenario}
        onScenarioChange={setScenario}
        systemStatus={systemStatus}
        isPeakActive={peakEvent.is_active}
      />

      {/* 2. MAIN CONTAINER */}
      <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8 space-y-6">
        {/* System Status Banner */}
        <SystemStatus
          status={systemStatus}
          houseId={homeState.house_id}
          gridAbsent={homeState.grid_absent}
          operatingMode={homeState.operating_mode}
        />

        {/* Dynamic View: Resident vs DISCOM Controller */}
        {activeTab === 'resident' ? (
          <ResidentView
            homeState={homeState}
            peakEvent={peakEvent}
            chartData={chartData}
            onOverride={triggerOverride}
            lastAck={lastAck}
            onClearAck={clearLastAck}
          />
        ) : (
          <GridController
            peakEvent={peakEvent}
            homeState={homeState}
            onDeclarePeak={declarePeakEvent}
            onCancelPeak={cancelPeakEvent}
          />
        )}
      </main>

      {/* 3. FOOTER */}
      <footer className="mt-12 border-t border-blue-100 bg-white/90 py-6 text-center text-xs text-slate-500 font-mono">
        <div className="mx-auto max-w-7xl px-4 flex flex-wrap items-center justify-between gap-2">
          <span>SHARP-HEMS • Cyber-Physical Energy Management System</span>
          <span>APCPDCL Guntur Region • 431 Enrolled Smart Households</span>
          <span>Safety Shield: Inviolable Necessity Mask</span>
        </div>
      </footer>
    </div>
  );
}

