'use client';

import React from 'react';
import { Zap, Shield, Radio, Eye } from 'lucide-react';
import { SystemConnectionStatus } from '@/lib/contracts';

interface DashboardHeaderProps {
  activeTab: 'resident' | 'discom';
  onTabChange: (tab: 'resident' | 'discom') => void;
  activeScenario: 'normal' | 'peak' | 'outage';
  onScenarioChange: (scenario: 'normal' | 'peak' | 'outage') => void;
  systemStatus: SystemConnectionStatus;
  isPeakActive: boolean;
}

export function DashboardHeader({
  activeTab,
  onTabChange,
  activeScenario,
  onScenarioChange,
  systemStatus,
  isPeakActive,
}: DashboardHeaderProps) {
  const isConnected = systemStatus.status === 'connected';

  return (
    <header className="sticky top-0 z-50 border-b border-blue-100 bg-white/95 backdrop-blur-xl shadow-sm">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex min-h-16 items-center justify-between gap-4 py-2">

          {/* Brand */}
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-blue-200 bg-blue-50">
              <Zap className="h-5 w-5 text-blue-600" />
            </div>

            <div>
              <div className="flex items-center gap-2">
                <span className="font-mono text-lg font-black tracking-wider text-[#17365D]">
                  SHARP
                </span>

                <span className="hidden sm:inline-flex items-center rounded-full border border-blue-200 bg-blue-50 px-2 py-0.5 font-mono text-[10px] font-semibold text-blue-700">
                  HEMS v2.1
                </span>
              </div>

              <p className="hidden md:block text-[11px] text-blue-700/70 font-medium">
                Smart Home Appliance Response Platform
              </p>
            </div>
          </div>

          {/* Navigation */}
          <div className="flex items-center rounded-xl border border-blue-100 bg-blue-50 p-1 font-mono text-xs">

            <button
              type="button"
              onClick={() => onTabChange('resident')}
              className={`flex items-center gap-1.5 rounded-lg px-3.5 py-1.5 font-semibold transition-all cursor-pointer ${
                activeTab === 'resident'
                  ? 'border border-blue-200 bg-blue-600 text-white shadow-sm'
                  : 'text-blue-700 hover:bg-blue-100'
              }`}
            >
              <Shield className="h-3.5 w-3.5" />
              Resident View
            </button>

            <button
              type="button"
              onClick={() => onTabChange('discom')}
              className={`relative flex items-center gap-1.5 rounded-lg px-3.5 py-1.5 font-semibold transition-all cursor-pointer ${
                activeTab === 'discom'
                  ? 'border border-blue-200 bg-blue-600 text-white shadow-sm'
                  : 'text-blue-700 hover:bg-blue-100'
              }`}
            >
              <Radio className="h-3.5 w-3.5" />
              DISCOM / Grid

              {isPeakActive && (
                <span className="relative flex h-2 w-2">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-blue-300 opacity-70" />
                  <span className="relative inline-flex h-2 w-2 rounded-full bg-blue-500" />
                </span>
              )}
            </button>
          </div>

          {/* Right Controls */}
          <div className="flex items-center gap-3">

            {/* Demo Scenario */}
            <div className="hidden lg:flex items-center gap-1 rounded-lg border border-blue-100 bg-blue-50 p-1 text-[11px] font-mono">

              <span className="flex items-center gap-1 px-1.5 text-[10px] font-bold uppercase text-blue-600">
                <Eye className="h-3 w-3" />
                Demo:
              </span>

              <button
                type="button"
                onClick={() => onScenarioChange('normal')}
                className={`rounded px-2 py-0.5 transition-all cursor-pointer ${
                  activeScenario === 'normal'
                    ? 'bg-blue-600 text-white font-bold shadow-sm'
                    : 'text-blue-600 hover:bg-blue-100'
                }`}
              >
                Normal
              </button>

              <button
                type="button"
                onClick={() => onScenarioChange('peak')}
                className={`rounded px-2 py-0.5 transition-all cursor-pointer ${
                  activeScenario === 'peak'
                    ? 'bg-blue-500 text-white font-bold shadow-sm'
                    : 'text-blue-600 hover:bg-blue-100'
                }`}
              >
                Midday Peak (14:05)
              </button>

              <button
                type="button"
                onClick={() => onScenarioChange('outage')}
                className={`rounded px-2 py-0.5 transition-all cursor-pointer ${
                  activeScenario === 'outage'
                    ? 'bg-blue-300 text-blue-900 font-bold'
                    : 'text-blue-600 hover:bg-blue-100'
                }`}
              >
                Outage
              </button>
            </div>

            {/* MQTT Status */}
            <div
              className={`flex items-center gap-1.5 rounded-full border px-2.5 py-1 font-mono text-[11px] font-semibold ${
                isConnected
                  ? 'border-blue-200 bg-blue-50 text-blue-700'
                  : 'border-blue-200 bg-blue-100 text-blue-800'
              }`}
            >
              <span
                className={`h-1.5 w-1.5 rounded-full ${
                  isConnected
                    ? 'bg-blue-500 animate-pulse'
                    : 'bg-blue-700'
                }`}
              />

              <span className="hidden sm:inline">MQTT</span>
              {isConnected ? 'LIVE' : 'OFFLINE'}
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}