'use client';

import React from 'react';
import {
  Wifi,
  WifiOff,
  RefreshCw,
  ShieldAlert,
  Cpu,
} from 'lucide-react';
import { SystemConnectionStatus } from '@/lib/contracts';

interface SystemStatusProps {
  status: SystemConnectionStatus;
  houseId: string;
  gridAbsent: boolean;
  operatingMode: string;
}

export function SystemStatus({
  status,
  houseId,
  gridAbsent,
  operatingMode,
}: SystemStatusProps) {
  const isConnected = status.status === 'connected';

  return (
    <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-blue-100 bg-white px-4 py-2.5 shadow-sm text-xs">

      <div className="flex flex-wrap items-center gap-4">

        {/* Connection */}
        <div className="flex items-center gap-2">
          <span className="relative flex h-2.5 w-2.5">
            {isConnected ? (
              <>
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-blue-300 opacity-50" />
                <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-blue-500" />
              </>
            ) : (
              <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-blue-700" />
            )}
          </span>

          <span className="flex items-center gap-1.5 font-mono font-medium text-blue-800">
            {isConnected ? (
              <>
                <Wifi className="h-3.5 w-3.5 text-blue-600" />
                MQTT CONNECTED
              </>
            ) : (
              <>
                <WifiOff className="h-3.5 w-3.5 text-blue-700" />
                MQTT DISCONNECTED (Retained State)
              </>
            )}
          </span>
        </div>

        {/* Data Age */}
        <div className="flex items-center gap-1 font-mono text-blue-600">
          <RefreshCw className="h-3 w-3 text-blue-400" />

          <span>Age:</span>

          <span
            className={`font-semibold ${
              status.data_age_seconds > 30
                ? 'text-blue-800'
                : 'text-blue-700'
            }`}
          >
            {status.data_age_seconds}s
          </span>
        </div>

        {/* House Node */}
        <div className="hidden sm:flex items-center gap-1 font-mono text-blue-600">
          <Cpu className="h-3.5 w-3.5 text-blue-500" />

          <span>Node:</span>

          <span className="font-semibold text-blue-800">
            {houseId}
          </span>
        </div>

        {/* Operating Mode */}
        <div className="hidden md:flex items-center gap-1.5 font-mono text-blue-600">
          <span>Mode:</span>

          <span
            className={`rounded px-1.5 py-0.5 text-[11px] font-semibold uppercase ${
              gridAbsent
                ? 'border border-blue-300 bg-blue-100 text-blue-800'
                : 'border border-blue-100 bg-blue-50 text-blue-700'
            }`}
          >
            {operatingMode.replace('_', ' ')}
          </span>
        </div>
      </div>

      {/* Right Status */}
      <div className="flex items-center gap-3">

        {gridAbsent && (
          <div className="flex items-center gap-1.5 rounded-md border border-blue-300 bg-blue-100 px-2 py-0.5 font-mono text-[11px] font-semibold text-blue-800 animate-pulse">
            <ShieldAlert className="h-3 w-3" />
            GRID ABSENT: ISLANDED BATTERY
          </div>
        )}

        <span className="hidden lg:inline text-[11px] text-blue-400 font-mono">
          WSS:443 (TLS Encrypted)
        </span>
      </div>
    </div>
  );
}