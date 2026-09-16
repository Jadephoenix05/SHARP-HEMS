/**
 * Browser MQTT client. Runs in the BROWSER, never on Vercel's server.
 *
 * Vercel functions are request-scoped: they wake, answer, freeze. A frozen
 * function drops a long-lived subscription, and you get intermittent missing
 * telemetry that is painful to diagnose. So the client lives here, in the page.
 *
 * Port 443, not 1883. Campus and hostel Wi-Fi frequently block plain MQTT;
 * MQTT over WebSockets Secure uses the same port as HTTPS and passes through
 * anything that allows web browsing. Test this on campus before demo day.
 */

import mqtt, { MqttClient } from 'mqtt';
import { Feed, HomeState, Intent } from './contracts';

const HOUSE = process.env.NEXT_PUBLIC_HOUSE_ID ?? 'demo';

/** Two control intervals. Past this the data is stale and must look stale. */
const STALE_AFTER_SECONDS = 30 * 60;

export interface FeedHandlers {
  onFeed: (feed: Feed) => void;
}

export function connectFeed({ onFeed }: FeedHandlers): () => void {
  let state: HomeState | null = null;
  let intent: Intent | null = null;
  let lastMessage = 0;
  let client: MqttClient | null = null;

  const url = process.env.NEXT_PUBLIC_MQTT_URL;
  if (!url) {
    onFeed({ status: 'offline', dataAgeSeconds: 0, state: null, intent: null });
    return () => undefined;
  }

  const emit = (status: Feed['status']) => {
    const age = lastMessage ? (Date.now() - lastMessage) / 1000 : 0;
    // Never report 'live' on data this old, whatever the socket thinks. A
    // dashboard that renders stale data as live is worse than one showing
    // nothing, because nobody knows to distrust it.
    const effective =
      status === 'live' && lastMessage && age > STALE_AFTER_SECONDS ? 'stale' : status;
    onFeed({ status: effective, dataAgeSeconds: age, state, intent });
  };

  client = mqtt.connect(url, {
    // These credentials are PUBLIC - anything in NEXT_PUBLIC_* is visible to
    // anyone who opens the page. This user must be subscribe-only on
    // home/<house>/#. If it leaks, someone reads demo telemetry. Put the
    // publishing credential here instead and they can command your relays.
    username: process.env.NEXT_PUBLIC_MQTT_USER,
    password: process.env.NEXT_PUBLIC_MQTT_PASS,
    reconnectPeriod: 2000,
    connectTimeout: 10_000,
    clean: true,
    // One client id per session, or the free tier's connection limit is reached
    // and new browsers are silently refused.
    clientId: `sharp-dash-${Math.random().toString(16).slice(2, 10)}`,
  });

  client.on('connect', () => {
    client?.subscribe([`home/${HOUSE}/state`, `home/${HOUSE}/intent`], { qos: 0 });
    emit('live');
  });

  client.on('message', (topic, payload) => {
    try {
      const parsed = JSON.parse(payload.toString());
      if (topic.endsWith('/state')) state = parsed as HomeState;
      else if (topic.endsWith('/intent')) intent = parsed as Intent;
      lastMessage = Date.now();
      emit('live');
    } catch {
      // A malformed message must not take the page down. Keep the last good
      // state and let the age indicator show that nothing fresh has arrived.
    }
  });

  client.on('reconnect', () => emit('connecting'));
  client.on('offline', () => emit('offline'));
  client.on('error', () => emit('offline'));

  // Re-emit on a timer so the age counter keeps moving even when no message
  // arrives. Without this the UI looks live forever after the feed dies.
  const tick = setInterval(() => emit(client?.connected ? 'live' : 'offline'), 1000);

  return () => {
    clearInterval(tick);
    client?.end(true);
  };
}
