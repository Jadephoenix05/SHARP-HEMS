'use client';

import React from 'react';
import { LucideIcon } from 'lucide-react';

interface StatCardProps {
  title: string;
  value: string | number;
  unit?: string;
  subtext?: string;
  icon: LucideIcon;
  variant?: 'cyan' | 'emerald' | 'amber' | 'rose' | 'default';
  badge?: {
    text: string;
    type: 'success' | 'warning' | 'alert' | 'neutral';
  };
}

export function StatCard({
  title,
  value,
  unit,
  subtext,
  icon: Icon,
  variant = 'default',
  badge,
}: StatCardProps) {
  const variantStyles = {
    cyan:
      'border-blue-200 bg-blue-50/50 hover:border-blue-300',
    emerald:
      'border-blue-200 bg-blue-50/30 hover:border-blue-300',
    amber:
      'border-blue-300 bg-blue-50/70 hover:border-blue-400',
    rose:
      'border-blue-200 bg-blue-50/40 hover:border-blue-300',
    default:
      'border-blue-100 bg-white hover:border-blue-200',
  }[variant];

  const iconColor = {
    cyan:
      'text-blue-600 bg-blue-100 border-blue-200',
    emerald:
      'text-blue-600 bg-blue-50 border-blue-200',
    amber:
      'text-blue-700 bg-blue-100 border-blue-300',
    rose:
      'text-blue-700 bg-blue-50 border-blue-200',
    default:
      'text-blue-600 bg-blue-50 border-blue-100',
  }[variant];

  const valueColor = 'text-[#17365D]';

  const badgeStyles = badge
    ? {
        success:
          'bg-blue-50 text-blue-700 border-blue-200',
        warning:
          'bg-blue-100 text-blue-800 border-blue-300',
        alert:
          'bg-blue-200 text-blue-900 border-blue-300',
        neutral:
          'bg-blue-50 text-blue-600 border-blue-100',
      }[badge.type]
    : '';

  return (
    <div
      className={`group relative overflow-hidden rounded-xl border p-4 shadow-sm transition-all duration-200 hover:shadow-md ${variantStyles}`}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="space-y-1 min-w-0">
          <p className="text-xs font-medium uppercase tracking-wider text-blue-600">
            {title}
          </p>

          <div className="flex items-baseline gap-1.5 min-w-0">
            <span
              className={`font-mono text-2xl font-bold tracking-tight ${valueColor}`}
            >
              {value}
            </span>

            {unit && (
              <span className="font-mono text-xs font-semibold text-blue-500">
                {unit}
              </span>
            )}
          </div>
        </div>

        <div
          className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border ${iconColor}`}
        >
          <Icon className="h-5 w-5" />
        </div>
      </div>

      <div className="mt-3 flex items-center justify-between gap-2 border-t border-blue-100 pt-2.5 text-xs">
        {subtext && (
          <span className="truncate text-blue-500">
            {subtext}
          </span>
        )}

        {badge && (
          <span
            className={`inline-flex shrink-0 items-center rounded-full border px-2 py-0.5 font-mono text-[10px] font-medium tracking-wide ${badgeStyles}`}
          >
            {badge.text}
          </span>
        )}
      </div>
    </div>
  );
}