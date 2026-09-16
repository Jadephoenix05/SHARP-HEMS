'use client';

/**
 * Panel 4 — the panel that carries the argument.
 *
 * Three rules, all easy to get wrong:
 *
 *  1. A critical appliance in use renders with NO off control. Not greyed, not
 *     disabled - ABSENT. The resident must never see a button that would cut
 *     their fan, because no such action exists anywhere in the system.
 *
 *  2. Two states, not three. Nothing dims in this build.
 *
 *  3. The PROTECTED row must visibly not move when a peak starts. A reviewer
 *     should be able to watch an event begin and see it unchanged. That is the
 *     claim, rendered.
 */

import {
  ApplianceState,
  ConnectionStatus,
  DisplayGroup,
  Intent,
  REJECTION_TEXT,
  displayGroup,
  overrideAvailable,
} from '@/lib/contracts';

const GROUP_ORDER: DisplayGroup[] = ['PROTECTED', 'RUNNING', 'SHED', 'DEFERRED'];

const GROUP_LABEL: Record<DisplayGroup, string> = {
  PROTECTED: 'Protected — always served',
  RUNNING: 'Running',
  SHED: 'Paused by SHARP',
  DEFERRED: 'Scheduled for later',
};

const GROUP_STYLE: Record<DisplayGroup, string> = {
  PROTECTED: 'border-emerald-600 bg-emerald-950/40',
  RUNNING: 'border-sky-700 bg-sky-950/30',
  SHED: 'border-neutral-700 bg-transparent',
  DEFERRED: 'border-amber-700 bg-amber-950/20',
};

function label(id: string): string {
  return id.replace(/_\d+$/, '').replace(/_/g, ' ');
}

interface Props {
  appliances: ApplianceState[];
  intent: Intent | null;
  status: ConnectionStatus;
  onOverride: (applianceId: string, level: 0 | 1) => void;
}

export function ApplianceGrid({ appliances, intent, status, onOverride }: Props) {
  const grouped = GROUP_ORDER.map((group) => ({
    group,
    items: appliances.filter((a) => displayGroup(a) === group),
  })).filter((g) => g.items.length > 0);

  return (
    <section className="space-y-4">
      {grouped.map(({ group, items }) => (
        <div key={group}>
          <h3 className="mb-2 text-xs uppercase tracking-widest text-neutral-400">
            {GROUP_LABEL[group]}
            <span className="ml-2 text-neutral-600">{items.length}</span>
          </h3>

          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5">
            {items.map((a) => {
              const refusal = intent?.shield_reasons?.[a.appliance_id]?.[0];
              const override = overrideAvailable(a, status);

              return (
                <article
                  key={a.appliance_id}
                  className={`rounded-lg border p-3 ${GROUP_STYLE[group]}`}
                >
                  <div className="flex items-baseline justify-between">
                    <span className="truncate text-sm capitalize">
                      {label(a.appliance_id)}
                    </span>
                    {/* Simulated, and labelled so. There is no meter. */}
                    <span className="text-xs text-neutral-500" title="simulated">
                      {a.level > 0 ? `${Math.round(a.power_15min_mean_w)} W` : '—'}
                    </span>
                  </div>

                  <div className="mt-1 text-xs text-neutral-400">
                    {a.service_class}
                  </div>

                  {/* Rule 1: no off control on a protected load. Absent, not
                      greyed - the affordance itself must not exist. */}
                  {group === 'PROTECTED' ? (
                    <p className="mt-3 text-xs text-emerald-400">
                      Essential — never shed
                    </p>
                  ) : (
                    <button
                      type="button"
                      disabled={!override.enabled}
                      onClick={() => onOverride(a.appliance_id, a.level > 0 ? 0 : 1)}
                      title={override.reason}
                      className="mt-3 w-full rounded border border-neutral-700 px-2 py-1
                                 text-xs disabled:cursor-not-allowed disabled:opacity-40"
                    >
                      {a.level > 0 ? 'Turn off' : 'Turn on'}
                    </button>
                  )}

                  {/* A refusal is evidence, not an error. Show it. */}
                  {refusal && (
                    <p className="mt-2 text-xs text-amber-400">
                      {REJECTION_TEXT[refusal]}
                    </p>
                  )}

                  {/* A phone override can fail silently where a switch cannot,
                      so say why the control is unavailable. */}
                  {!override.enabled && override.reason && !refusal && (
                    <p className="mt-2 text-xs text-neutral-500">{override.reason}</p>
                  )}

                  {!a.actuation_verified && (
                    <p className="mt-2 text-xs text-red-400">
                      Relay disagrees with command
                    </p>
                  )}
                </article>
              );
            })}
          </div>
        </div>
      ))}
    </section>
  );
}
