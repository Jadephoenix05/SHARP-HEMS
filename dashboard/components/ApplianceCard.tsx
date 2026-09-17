'use client';

import React from 'react';
import {
  Shield,
  CheckCircle2,
  Clock,
  RotateCcw,
  Power,
} from 'lucide-react';
import { ApplianceState, ActionLevel } from '@/lib/contracts';

interface ApplianceCardProps {
  appliance: ApplianceState;
  onOverride?: (applianceId: string, level: ActionLevel) => void;
  isLoading?: boolean;
}

export function ApplianceCard({
  appliance,
  onOverride,
  isLoading,
}: ApplianceCardProps) {
  const isNecessity = appliance.is_necessity;
  const isRunning = appliance.level === 1;
  const isDeferred =
    appliance.level === 0 &&
    Boolean(appliance.deferred_until);
  const isShed =
    appliance.level === 0 &&
    !appliance.deferred_until;

  return (
    <div
      className={`relative overflow-hidden rounded-xl border p-4 transition-all duration-200 bg-white shadow-sm ${
        isNecessity
          ? 'border-blue-300 bg-blue-50/60'
          : isRunning
          ? 'border-blue-200 bg-white'
          : isDeferred
          ? 'border-blue-200 bg-blue-50/40'
          : 'border-blue-100 bg-blue-50/20 opacity-95'
      }`}
    >
      {/* Top row: Name, Type & Class Badge */}
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <h4 className="font-semibold text-sm text-[#17365D]">
              {appliance.display_name ||
                appliance.appliance_id}
            </h4>

            {isNecessity && (
              <span className="inline-flex items-center gap-1 rounded bg-blue-100 px-1.5 py-0.5 font-mono text-[10px] font-bold text-blue-800 border border-blue-200">
                <Shield className="h-2.5 w-2.5" />
                PROTECTED
              </span>
            )}
          </div>

          <p className="text-[11px] text-blue-500 capitalize mt-0.5">
            {appliance.service_class} • ID:{' '}
            <code className="font-mono text-blue-700">
              {appliance.appliance_id}
            </code>
          </p>
        </div>

        {/* State Pill */}
        <div>
          {isNecessity ? (
            <span className="inline-flex items-center gap-1 rounded-full border border-blue-300 bg-blue-100 px-2.5 py-1 font-mono text-xs font-bold text-blue-800">
              <span className="h-1.5 w-1.5 rounded-full bg-blue-600 animate-pulse" />
              ON
            </span>
          ) : isRunning ? (
            <span className="inline-flex items-center gap-1 rounded-full border border-blue-200 bg-blue-50 px-2.5 py-1 font-mono text-xs font-bold text-blue-700">
              <span className="h-1.5 w-1.5 rounded-full bg-blue-500" />
              RUNNING
            </span>
          ) : isDeferred ? (
            <span className="inline-flex items-center gap-1 rounded-full border border-blue-200 bg-blue-100 px-2.5 py-1 font-mono text-xs font-bold text-blue-800">
              <Clock className="h-3 w-3 text-blue-600" />
              DEFERRED
            </span>
          ) : (
            <span className="inline-flex items-center gap-1 rounded-full border border-blue-200 bg-blue-50 px-2.5 py-1 font-mono text-xs font-bold text-blue-600">
              <span className="h-1.5 w-1.5 rounded-full bg-blue-400" />
              SHED
            </span>
          )}
        </div>
      </div>

      {/* Middle row: Power Metrics & Verification */}
      <div className="mt-3.5 grid grid-cols-2 gap-2 rounded-lg border border-blue-100 bg-blue-50/50 p-2.5 text-xs">
        <div>
          <span className="block text-[10px] font-medium uppercase tracking-wider text-blue-500">
            SIMULATED POWER
          </span>

          <div className="mt-0.5 flex items-baseline gap-1">
            <span className="font-mono text-sm font-bold text-[#17365D]">
              {isRunning
                ? appliance.power_15min_mean_w.toFixed(1)
                : '0.0'}
            </span>

            <span className="font-mono text-[10px] text-blue-500">
              W
            </span>
          </div>

          <span className="block text-[9px] text-blue-400">
            {appliance.measured_w === null
              ? 'No meter fitted'
              : `${appliance.measured_w} W (meter)`}
          </span>
        </div>

        <div>
          <span className="block text-[10px] font-medium uppercase tracking-wider text-blue-500">
            ACTUATION VERIFIED
          </span>

          <div className="mt-0.5 flex items-center gap-1.5">
            {appliance.actuation_verified ? (
              <>
                <CheckCircle2 className="h-3.5 w-3.5 text-blue-600 shrink-0" />
                <span className="font-mono text-xs font-medium text-blue-700">
                  GPIO Readback MATCH
                </span>
              </>
            ) : (
              <span className="font-mono text-xs text-blue-600">
                PENDING VERIFY
              </span>
            )}
          </div>

          <span className="block text-[9px] text-blue-400">
            Pin state:{' '}
            {appliance.gpio_state === 1
              ? 'HIGH (1)'
              : 'LOW (0)'}
          </span>
        </div>
      </div>

      {/* Reason line for shed or deferred */}
      {!isNecessity && !isRunning && (
        <div className="mt-2.5 flex items-center justify-between rounded border border-blue-100 bg-white px-2.5 py-1.5 text-xs">
          <div className="flex items-center gap-1.5 text-blue-600">
            <span className="text-[11px] font-semibold text-blue-500">
              Reason:
            </span>

            <span className="font-mono font-bold text-blue-700 text-[11px]">
              {appliance.shed_reason || 'GRID_PEAK'}
            </span>
          </div>

          {appliance.deferred_until && (
            <span className="font-mono text-[10px] text-blue-600">
              Resumes: {appliance.deferred_until}
            </span>
          )}
        </div>
      )}

      {/* Action / Control Section:
          CRITICAL INVARIANT:
          Protected appliances must NOT have OFF/SHED controls rendered!
      */}
      <div className="mt-3.5 pt-2 border-t border-blue-100 flex items-center justify-between">
        {isNecessity ? (
          <div className="flex items-center gap-1.5 text-[11px] text-blue-700 font-mono">
            <Shield className="h-3 w-3 text-blue-600" />
            <span>Shield Guarantee: Always Protected</span>
          </div>
        ) : (
          <>
            <span className="text-[11px] text-blue-500">
              Resident Control:
            </span>

            {isShed || isDeferred ? (
              <button
                type="button"
                onClick={() =>
                  onOverride &&
                  onOverride(appliance.appliance_id, 1)
                }
                disabled={isLoading}
                className="inline-flex items-center gap-1.5 rounded-lg border border-blue-200 bg-blue-50 px-3 py-1 font-mono text-xs font-semibold text-blue-700 hover:bg-blue-100 hover:border-blue-300 active:scale-95 transition-all disabled:opacity-50 cursor-pointer"
              >
                <RotateCcw className="h-3 w-3" />
                RESTORE LOAD
              </button>
            ) : (
              <button
                type="button"
                onClick={() =>
                  onOverride &&
                  onOverride(appliance.appliance_id, 0)
                }
                disabled={isLoading}
                className="inline-flex items-center gap-1.5 rounded-lg border border-blue-200 bg-white px-3 py-1 font-mono text-xs font-semibold text-blue-700 hover:bg-blue-50 hover:border-blue-300 active:scale-95 transition-all disabled:opacity-50 cursor-pointer"
              >
                <Power className="h-3 w-3 text-blue-500" />
                SHED
              </button>
            )}
          </>
        )}
      </div>
    </div>
  );
}