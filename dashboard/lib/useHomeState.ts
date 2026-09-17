'use client';

import { useState, useCallback, useTransition } from 'react';
import {
  HomeState,
  PeakEvent,
  SystemConnectionStatus,
  ActionLevel,
  CommandAck,
} from './contracts';
import {
  NORMAL_HOME_STATE,
  PEAK_ACTIVE_HOME_STATE,
  INITIAL_PEAK_EVENT,
  MOCK_SYSTEM_STATUS,
} from './mockState';

export interface UseHomeStateReturn {
  homeState: HomeState;
  peakEvent: PeakEvent;
  systemStatus: SystemConnectionStatus;
  activeScenario: 'normal' | 'peak' | 'outage';
  setScenario: (scenario: 'normal' | 'peak' | 'outage') => void;
  triggerOverride: (applianceId: string, requestedLevel: ActionLevel) => Promise<CommandAck>;
  declarePeakEvent: (severity: number, durationMinutes: number) => void;
  cancelPeakEvent: () => void;
  lastAck: CommandAck | null;
  clearLastAck: () => void;
}

export function useHomeState(): UseHomeStateReturn {
  const [activeScenario, setActiveScenario] = useState<'normal' | 'peak' | 'outage'>('peak');
  const [homeState, setHomeState] = useState<HomeState>(PEAK_ACTIVE_HOME_STATE);
  const [peakEvent, setPeakEvent] = useState<PeakEvent>(INITIAL_PEAK_EVENT);
  const [systemStatus] = useState<SystemConnectionStatus>(MOCK_SYSTEM_STATUS);
  const [lastAck, setLastAck] = useState<CommandAck | null>(null);
  const [, startTransition] = useTransition();

  const setScenario = useCallback((scenario: 'normal' | 'peak' | 'outage') => {
    startTransition(() => {
      setActiveScenario(scenario);
      if (scenario === 'normal') {
        setHomeState(NORMAL_HOME_STATE);
        setPeakEvent((prev) => ({
          ...prev,
          severity: 0.15,
          is_active: false,
        }));
      } else if (scenario === 'peak') {
        setHomeState(PEAK_ACTIVE_HOME_STATE);
        setPeakEvent({
          ...INITIAL_PEAK_EVENT,
          is_active: true,
          severity: 0.67,
          declared_at: '14:05 IST',
          expires_at: '16:00 IST',
        });
      } else if (scenario === 'outage') {
        // Islanded outage mode
        setHomeState({
          ...PEAK_ACTIVE_HOME_STATE,
          operating_mode: 'islanded_outage',
          grid_absent: true,
          aggregate_power_kw: 0.116, // only inverter loads
          appliances: PEAK_ACTIVE_HOME_STATE.appliances.map((app) => {
            if (app.is_necessity) {
              return { ...app, level: 1 };
            }
            return {
              ...app,
              level: 0,
              shed_reason: 'GRID_OUTAGE_BATTERY_RESERVE',
            };
          }),
        });
        setPeakEvent((prev) => ({ ...prev, is_active: false }));
      }
    });
  }, []);

  const declarePeakEvent = useCallback((severity: number, durationMinutes: number) => {
    startTransition(() => {
      const now = new Date();
      const declaredStr = '14:05 IST';
      const expireHour = 14 + Math.floor((5 + durationMinutes) / 60);
      const expireMin = (5 + durationMinutes) % 60;
      const expireStr = `${expireHour.toString().padStart(2, '0')}:${expireMin
        .toString()
        .padStart(2, '0')} IST`;

      const newEvent: PeakEvent = {
        event_id: `apcpdcl-${now.getFullYear()}-09-15-${Math.floor(Math.random() * 900 + 100)}`,
        sequence: 48,
        region: 'APCPDCL / Guntur Urban Sub-12',
        severity,
        declared_at: declaredStr,
        expires_at: expireStr,
        reason: 'system_peak',
        is_active: true,
        homes_responding: 431,
        mw_relieved: Number((severity * 0.42).toFixed(2)),
      };
      setPeakEvent(newEvent);

      // Reflect in home state: if severity > 0.4, shed flexible loads
      if (severity >= 0.4) {
        setHomeState((prev) => ({
          ...prev,
          grid_peak_severity: severity,
          aggregate_power_kw: 0.145,
          appliances: prev.appliances.map((app) => {
            if (app.is_necessity) {
              return app; // Protected loads NEVER shed
            }
            if (app.appliance_type === 'washing_machine') {
              return {
                ...app,
                level: 0,
                shed_reason: 'PEAK_EVENT',
                deferred_until: '22:00 IST',
              };
            }
            return {
              ...app,
              level: 0,
              shed_reason: 'GRID_PEAK',
            };
          }),
        }));
      }
    });
  }, []);

  const cancelPeakEvent = useCallback(() => {
    startTransition(() => {
      setPeakEvent((prev) => ({ ...prev, is_active: false, severity: 0.15 }));
      setHomeState(NORMAL_HOME_STATE);
    });
  }, []);

  const triggerOverride = useCallback(
    async (applianceId: string, requestedLevel: ActionLevel): Promise<CommandAck> => {
      const targetApp = homeState.appliances.find((a) => a.appliance_id === applianceId);

      // SAFETY CHECK 1: Necessity load shedding refused (E8 shield rule)
      if (targetApp?.is_necessity && requestedLevel === 0) {
        const rejectionAck: CommandAck = {
          command_id: `cmd-${Date.now()}`,
          appliance_id: applianceId,
          accepted: false,
          applied_level: targetApp.level,
          rejected_reason: 'NECESSITY_MASK',
          gpio_state: targetApp.gpio_state,
          measured_w: null,
          verification: 'MATCH',
          acked_at: '14:05:12.184 IST',
          latency_ms: 184,
        };
        setLastAck(rejectionAck);
        return rejectionAck;
      }

      // Check if grid peak active and trying to turn on luxury load (in APCPDCL rules)
      if (peakEvent.is_active && requestedLevel === 1 && targetApp && !targetApp.is_necessity) {
        // Can be either honoured with advisory cost or refused by shield
        // Here we simulate the ACK round trip:
        const simulatedAck: CommandAck = {
          command_id: `cmd-${Date.now()}`,
          appliance_id: applianceId,
          accepted: true,
          applied_level: requestedLevel,
          rejected_reason: null,
          gpio_state: 1,
          measured_w: null,
          verification: 'MATCH',
          acked_at: '14:05:15.142 IST',
          latency_ms: 142,
        };

        startTransition(() => {
          setHomeState((prev) => ({
            ...prev,
            aggregate_power_kw: Number((prev.aggregate_power_kw + (targetApp.power_15min_mean_w || 200) / 1000).toFixed(3)),
            appliances: prev.appliances.map((app) =>
              app.appliance_id === applianceId
                ? { ...app, level: requestedLevel, shed_reason: null, gpio_state: 1 }
                : app
            ),
          }));
          setLastAck(simulatedAck);
        });

        return simulatedAck;
      }

      // Default normal override
      const defaultAck: CommandAck = {
        command_id: `cmd-${Date.now()}`,
        appliance_id: applianceId,
        accepted: true,
        applied_level: requestedLevel,
        rejected_reason: null,
        gpio_state: requestedLevel === 0 ? 0 : 1,
        measured_w: null,
        verification: 'MATCH',
        acked_at: '14:05:18.098 IST',
        latency_ms: 98,
      };

      startTransition(() => {
        setHomeState((prev) => ({
          ...prev,
          appliances: prev.appliances.map((app) =>
            app.appliance_id === applianceId
              ? {
                  ...app,
                  level: requestedLevel,
                  shed_reason: requestedLevel === 0 ? 'MANUAL_OVERRIDE' : null,
                  gpio_state: requestedLevel === 0 ? 0 : 1,
                }
              : app
          ),
        }));
        setLastAck(defaultAck);
      });

      return defaultAck;
    },
    [homeState.appliances, peakEvent.is_active]
  );

  const clearLastAck = useCallback(() => setLastAck(null), []);

  return {
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
  };
}
