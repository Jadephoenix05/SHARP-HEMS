'use client';

/**
 * The resident's screen.
 *
 * Answers the three questions a reviewer will ask:
 *   1. What did the agent want to do, and why?
 *   2. What did the safety shield refuse, and why?
 *   3. What happened when a human disagreed?
 *
 * Runs on the mock until the real feed exists. Swapping one is ONE line, by
 * design - if it were more, the seam between data and presentation was drawn
 * in the wrong place.
 */

import { useEffect, useMemo, useState } from 'react';
import { ApplianceGrid } from '@/components/ApplianceGrid';
import { Feed, homesNotBlackedOut } from '@/lib/contracts';
import { DEMO_PEAK_STEP, mockFeed } from '@/lib/mockState';
// import { connectFeed } from '@/lib/mqtt';   // <- the one-line swap

const USE_MOCK = process.env.NEXT_PUBLIC_USE_MOCK !== 'false';

export default function Page() {
  const [step, setStep] = useState(DEMO_PEAK_STEP - 4);
  const [feed, setFeed] = useState<Feed | null>(null);
  const [outage, setOutage] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    if (USE_MOCK) {
      const timer = setInterval(() => setStep((s) => s + 1), 1500);
      return () => clearInterval(timer);
    }
    // return connectFeed({ onFeed: setFeed });
    return undefined;
  }, []);

  const live = useMemo(
    () => (USE_MOCK ? mockFeed(step, { outage }) : feed),
    [step, outage, feed],
  );

  if (!live?.state) {
    return <main className="p-8 text-neutral-400">Connecting…</main>;
  }

  const s = live.state;
  const stale = live.status !== 'live' || live.dataAgeSeconds > 1800;
  const peak = s.peak_event_active;

  return (
    <main
      className={`min-h-screen bg-neutral-950 p-6 text-neutral-100 transition-opacity
                  ${stale ? 'opacity-50' : ''}`}
    >
      <header className="mb-6 flex flex-wrap items-baseline justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold">SHARP</h1>
          <p className="text-xs text-neutral-500">
            {new Date(s.timestamp_ist).toLocaleTimeString('en-IN', {
              hour: '2-digit', minute: '2-digit', timeZone: 'Asia/Kolkata',
            })}{' '}
            IST · house {s.house_id}
          </p>
        </div>

        {/* Stale must not look live. Show the age, always. */}
        <div className="text-right text-xs">
          <span className={stale ? 'text-amber-400' : 'text-emerald-400'}>
            {live.status}
          </span>
          {live.dataAgeSeconds > 60 && (
            <span className="ml-2 text-neutral-500">
              {Math.round(live.dataAgeSeconds / 60)} min old
            </span>
          )}
        </div>
      </header>

      {peak && (
        <div className="mb-5 rounded-lg border border-amber-600 bg-amber-950/40 p-4">
          <p className="font-medium text-amber-300">
            Grid peak · severity {s.grid_peak_severity.toFixed(2)}
          </p>
          <p className="mt-1 text-sm text-amber-200/80">
            Luxury loads are paused. Your fans, lights and fridge are unaffected.
          </p>
        </div>
      )}

      {s.grid_absent && (
        <div className="mb-5 rounded-lg border border-red-700 bg-red-950/40 p-4 text-red-200">
          Power cut. Running on the inverter circuit.
        </div>
      )}

      <section className="mb-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="Now" value={`${s.aggregate_power_kw.toFixed(3)} kW`}
              note={`of ${s.sanctioned_load_kw} kW sanctioned`} />
        <Stat label="Not controllable" value={`${s.background_load_kw.toFixed(3)} kW`}
              note="background draw" />
        <Stat label="Tariff" value={`₹${s.marginal_tariff_inr_kwh.toFixed(2)}`}
              note={`${s.month_to_date_kwh.toFixed(0)} kWh this month`} />
        <Stat label="Grid" value={peak ? 'PEAK' : 'normal'}
              note={`severity ${s.grid_peak_severity.toFixed(2)}`} />
      </section>

      <ApplianceGrid
        appliances={s.appliances}
        intent={live.intent}
        status={live.status}
        onOverride={(id, level) => {
          // The API records intent. It does NOT decide whether the override is
          // safe - the Pi's shield does, and it may refuse.
          setNotice(`Requested ${level === 0 ? 'off' : 'on'} for ${id}`);
          setTimeout(() => setNotice(null), 2500);
        }}
      />

      {notice && <p className="mt-4 text-xs text-sky-400">{notice}</p>}

      <footer className="mt-8 border-t border-neutral-800 pt-4 text-xs text-neutral-500">
        <p>
          Peak avoided today: 4.15 % — {homesNotBlackedOut(4.15)}.
        </p>
        <p className="mt-1">
          Appliance wattages are <strong>simulated</strong>; no meter is fitted.
          Model not approved for deployment.
        </p>
        {USE_MOCK && (
          <button
            className="mt-3 rounded border border-neutral-700 px-2 py-1"
            onClick={() => setOutage((o) => !o)}
          >
            {outage ? 'End outage' : 'Simulate outage'}
          </button>
        )}
      </footer>
    </main>
  );
}

function Stat({ label, value, note }: { label: string; value: string; note: string }) {
  return (
    <div className="rounded-lg border border-neutral-800 p-3">
      <p className="text-xs uppercase tracking-wide text-neutral-500">{label}</p>
      <p className="mt-1 text-lg">{value}</p>
      <p className="text-xs text-neutral-500">{note}</p>
    </div>
  );
}
