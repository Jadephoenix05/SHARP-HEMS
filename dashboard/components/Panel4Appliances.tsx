'use client';

import React, { useState } from 'react';
import {
  ShieldCheck,
  Sliders,
  CheckCircle2,
  XCircle,
} from 'lucide-react';

import {
  ApplianceState,
  ActionLevel,
  CommandAck,
} from '@/lib/contracts';

import { ApplianceCard } from './ApplianceCard';

interface Panel4AppliancesProps {
  appliances: ApplianceState[];
  onOverride: (
    applianceId: string,
    level: ActionLevel
  ) => Promise<CommandAck>;
  lastAck: CommandAck | null;
  onClearAck: () => void;
}

export function Panel4Appliances({
  appliances,
  onOverride,
  lastAck,
  onClearAck,
}: Panel4AppliancesProps) {
  const [loadingApplianceId, setLoadingApplianceId] =
    useState<string | null>(null);

  const protectedAppliances = appliances.filter(
    (a) => a.is_necessity
  );

  const flexibleAppliances = appliances.filter(
    (a) => !a.is_necessity
  );

  const handleOverride = async (
    applianceId: string,
    level: ActionLevel
  ) => {
    try {
      setLoadingApplianceId(applianceId);
      await onOverride(applianceId, level);
    } finally {
      setLoadingApplianceId(null);
    }
  };

  return (
    <div className="space-y-6">

      {/* =====================================================
          ACKNOWLEDGEMENT
      ===================================================== */}
      {lastAck && (
        <div
          className={`flex items-start justify-between gap-3 rounded-xl border p-4 shadow-sm ${
            lastAck.accepted
              ? 'border-blue-200 bg-blue-50 text-blue-800'
              : 'border-blue-300 bg-blue-100 text-blue-900'
          }`}
        >
          <div className="flex items-start gap-3">

            {lastAck.accepted ? (
              <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0 text-blue-600" />
            ) : (
              <XCircle className="mt-0.5 h-5 w-5 shrink-0 text-blue-700" />
            )}

            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <span className="font-mono text-xs font-bold uppercase tracking-wider">
                  {lastAck.accepted
                    ? 'COMMAND ACCEPTED'
                    : 'COMMAND REJECTED'}
                </span>

                <span className="font-mono text-[11px] text-slate-500">
                  {lastAck.appliance_id} →{' '}
                  {lastAck.applied_level === 1
                    ? 'ON'
                    : 'SHED'}
                </span>
              </div>

              <p className="text-xs text-slate-600">
                {lastAck.accepted ? (
                  <>
                    Actuation verified via GPIO readback (
                    {lastAck.verification}). Round-trip latency:{' '}
                    <strong className="font-mono text-blue-700">
                      {lastAck.latency_ms} ms
                    </strong>
                    .
                  </>
                ) : (
                  <>
                    Refused by Safety Shield:{' '}
                    <strong className="font-mono text-blue-800">
                      {lastAck.rejected_reason}
                    </strong>
                    . Critical loads cannot be disconnected.
                  </>
                )}
              </p>
            </div>
          </div>

          <button
            type="button"
            onClick={onClearAck}
            className="rounded border border-slate-200 bg-white px-2 py-0.5 text-xs text-slate-500 hover:border-blue-200 hover:text-blue-700 cursor-pointer"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* =====================================================
          PROTECTED APPLIANCES
      ===================================================== */}
      <section className="rounded-2xl border border-blue-200 bg-white p-5 shadow-sm">

        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">

          <div className="flex items-center gap-2.5">

            <div className="flex h-8 w-8 items-center justify-center rounded-lg border border-blue-200 bg-blue-50 text-blue-700">
              <ShieldCheck className="h-4 w-4" />
            </div>

            <div>
              <h3 className="font-mono text-sm font-bold tracking-wide text-blue-800">
                PROTECTED APPLIANCES (CRITICAL NECESSITY)
              </h3>

              <p className="text-xs text-slate-500">
                Guaranteed uninterrupted power. Control buttons
                are omitted by design—these loads can never be
                shed.
              </p>
            </div>
          </div>

          <span className="inline-flex items-center gap-1.5 rounded-full border border-blue-200 bg-blue-50 px-2.5 py-1 font-mono text-xs font-semibold text-blue-700">
            <span className="h-1.5 w-1.5 rounded-full bg-blue-600 animate-pulse" />
            {protectedAppliances.length} OF{' '}
            {protectedAppliances.length} ACTIVE
          </span>
        </div>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
          {protectedAppliances.map((app) => (
            <ApplianceCard
              key={app.appliance_id}
              appliance={app}
              isLoading={
                loadingApplianceId === app.appliance_id
              }
            />
          ))}
        </div>
      </section>

      {/* =====================================================
          FLEXIBLE APPLIANCES
      ===================================================== */}
      <section className="rounded-2xl border border-blue-100 bg-white p-5 shadow-sm">

        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">

          <div className="flex items-center gap-2.5">

            <div className="flex h-8 w-8 items-center justify-center rounded-lg border border-blue-100 bg-blue-50 text-blue-600">
              <Sliders className="h-4 w-4" />
            </div>

            <div>
              <h3 className="font-mono text-sm font-bold tracking-wide text-[#172b4d]">
                FLEXIBLE APPLIANCES (DEMAND RESPONSE READY)
              </h3>

              <p className="text-xs text-slate-500">
                Controlled by SHARP policy during feeder peaks.
                Residents can submit authenticated override
                requests.
              </p>
            </div>
          </div>

          <span className="rounded-md border border-blue-100 bg-blue-50 px-2.5 py-1 font-mono text-xs text-blue-700">
            {
              flexibleAppliances.filter(
                (a) => a.level === 1
              ).length
            }{' '}
            RUNNING •{' '}
            {
              flexibleAppliances.filter(
                (a) => a.level === 0
              ).length
            }{' '}
            SHED/DEFERRED
          </span>
        </div>

        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {flexibleAppliances.map((app) => (
            <ApplianceCard
              key={app.appliance_id}
              appliance={app}
              onOverride={handleOverride}
              isLoading={
                loadingApplianceId === app.appliance_id
              }
            />
          ))}
        </div>
      </section>
    </div>
  );
}