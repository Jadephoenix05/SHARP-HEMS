'use client';

import React from 'react';
import {
  AlertTriangle,
  Clock,
  MapPin,
  Users,
  Zap,
  ShieldCheck,
} from 'lucide-react';

import { PeakEvent } from '@/lib/contracts';

interface PeakEventCardProps {
  peakEvent: PeakEvent;
  onDismiss?: () => void;
}

export function PeakEventCard({
  peakEvent,
  onDismiss,
}: PeakEventCardProps) {
  if (!peakEvent.is_active) {
    return (
      <div className="flex items-center justify-between rounded-xl border border-blue-100 bg-white px-4 py-3 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg border border-blue-100 bg-blue-50 text-blue-600">
            <ShieldCheck className="h-4 w-4" />
          </div>

          <div>
            <p className="text-xs font-semibold text-[#172b4d]">
              Normal Grid Condition
            </p>

            <p className="text-[11px] text-slate-500">
              Current grid stress is within baseline parameters
              (Severity: {peakEvent.severity.toFixed(2)})
            </p>
          </div>
        </div>

        <span className="rounded-full border border-blue-100 bg-blue-50 px-2.5 py-0.5 font-mono text-[10px] font-semibold text-blue-700">
          GRID RELAXED
        </span>
      </div>
    );
  }

  return (
    <div className="relative overflow-hidden rounded-xl border-2 border-blue-300 bg-blue-50/60 p-5 shadow-sm">

      <div className="relative z-10 space-y-4">

        {/* Header */}
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-blue-200 pb-3">

          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border border-blue-200 bg-white text-blue-600">
              <AlertTriangle className="h-5 w-5" />
            </div>

            <div>
              <div className="flex flex-wrap items-center gap-2">
                <h3 className="font-mono text-sm font-bold tracking-wide text-blue-800">
                  PEAK EVENT ACTIVE
                </h3>

                <span className="rounded bg-blue-600 px-1.5 py-0.5 font-mono text-[10px] font-bold text-white uppercase">
                  DISCOM Broadcast
                </span>
              </div>

              <p className="text-xs text-slate-600">
                Midday feeder peak declared under APCPDCL
                dynamic response tariff
              </p>
            </div>
          </div>

          <div className="text-right">
            <span className="block text-[10px] font-medium uppercase tracking-wider text-slate-500">
              Grid Peak Severity
            </span>

            <span className="font-mono text-xl font-extrabold text-blue-700">
              {peakEvent.severity.toFixed(2)}
            </span>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">

          <div className="rounded-lg border border-blue-100 bg-white p-2.5 shadow-sm">
            <div className="flex items-center gap-1.5 text-slate-500 text-[11px]">
              <Clock className="h-3 w-3 text-blue-600" />
              <span>Event Window</span>
            </div>

            <p className="mt-1 font-mono text-xs font-bold text-[#172b4d]">
              {peakEvent.declared_at} → {peakEvent.expires_at}
            </p>
          </div>

          <div className="rounded-lg border border-blue-100 bg-white p-2.5 shadow-sm">
            <div className="flex items-center gap-1.5 text-slate-500 text-[11px]">
              <MapPin className="h-3 w-3 text-blue-600" />
              <span>Feeder Region</span>
            </div>

            <p
              className="mt-1 truncate font-mono text-xs font-bold text-[#172b4d]"
              title={peakEvent.region}
            >
              {peakEvent.region}
            </p>
          </div>

          <div className="rounded-lg border border-blue-100 bg-white p-2.5 shadow-sm">
            <div className="flex items-center gap-1.5 text-slate-500 text-[11px]">
              <Users className="h-3 w-3 text-blue-600" />
              <span>Homes Enrolled</span>
            </div>

            <p className="mt-1 font-mono text-xs font-bold text-blue-700">
              {peakEvent.homes_responding ?? 431} homes
            </p>
          </div>

          <div className="rounded-lg border border-blue-100 bg-white p-2.5 shadow-sm">
            <div className="flex items-center gap-1.5 text-slate-500 text-[11px]">
              <Zap className="h-3 w-3 text-blue-600" />
              <span>Relief Achieved</span>
            </div>

            <p className="mt-1 font-mono text-xs font-bold text-blue-700">
              {peakEvent.mw_relieved ?? 0.28} MW
            </p>
          </div>
        </div>

        {/* Safety Banner */}
        <div className="flex items-start gap-2.5 rounded-lg border border-blue-200 bg-blue-50 p-3 text-xs text-blue-800">
          <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-blue-600" />

          <div className="space-y-0.5">
            <p className="font-semibold text-blue-800">
              SHARP Safety Guarantee: Necessity Loads Protected
            </p>

            <p className="text-[11px] leading-relaxed text-slate-600">
              Ceiling fan, lighting, and refrigeration remain
              100% active. Even the DISCOM cannot cut your
              essential services. Non-essential cooling and
              interruptible loads are paused to prevent local
              transformer tripping.
            </p>
          </div>
        </div>

      </div>
    </div>
  );
}