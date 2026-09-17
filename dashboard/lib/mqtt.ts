/**
 * Browser MQTT client. Runs in the BROWSER, never on Vercel's server.
 *
 * Vercel functions are request-scoped: they wake, answer, freeze. A frozen
 * function drops a long-lived subscription, and you get intermittent missing
 * telemetry that is painful to diagnose. So the client lives here, in the page.
 *
 * MQTT over WebSockets Secure is used so the browser can connect to HiveMQ.
 */

import mqtt, { MqttClient } from 'mqtt';
import { Feed, HomeState, Intent } from './contracts';

const HOUSE = process.env.NEXT_PUBLIC_HOUSE_ID ?? 'demo';

/** Past this point, received telemetry is considered stale. */
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

  // ------------------------------------------------------------
  // MQTT URL CHECK
  // ------------------------------------------------------------

  if (!url) {
    console.error('❌ NEXT_PUBLIC_MQTT_URL is not configured');

    onFeed({
      status: 'offline',
      dataAgeSeconds: 0,
      state: null,
      intent: null,
    });

    return () => undefined;
  }

  console.log('🔌 Connecting to HiveMQ...');

  // ------------------------------------------------------------
  // MQTT CONFIG DIAGNOSTIC
  // ------------------------------------------------------------

  console.log('🔎 MQTT CONFIG:', {
    url,
    user: process.env.NEXT_PUBLIC_MQTT_USER,
    hasPassword: !!process.env.NEXT_PUBLIC_MQTT_PASS,
  });

  // ------------------------------------------------------------
  // FEED EMITTER
  // ------------------------------------------------------------

  const emit = (status: Feed['status']) => {
    const age = lastMessage
      ? (Date.now() - lastMessage) / 1000
      : 0;

    const effective =
      status === 'live' &&
      lastMessage &&
      age > STALE_AFTER_SECONDS
        ? 'stale'
        : status;

    onFeed({
      status: effective,
      dataAgeSeconds: age,
      state,
      intent,
    });
  };

  // ------------------------------------------------------------
  // CONNECT TO HIVEMQ
  // ------------------------------------------------------------

  client = mqtt.connect(url, {
    username: process.env.NEXT_PUBLIC_MQTT_USER,
    password: process.env.NEXT_PUBLIC_MQTT_PASS,

    reconnectPeriod: 2000,
    connectTimeout: 10_000,

    clean: true,

    clientId: `sharp-dash-${Math.random()
      .toString(16)
      .slice(2, 10)}`,
  });

  // ------------------------------------------------------------
  // CONNECT EVENT
  // ------------------------------------------------------------

  client.on('connect', () => {
    console.log('✅ MQTT CONNECTED TO HIVEMQ');

    const topics = [
      `home/${HOUSE}/state`,
      `home/${HOUSE}/intent`,
    ];

    console.log('📡 Subscribing to:', topics);

    client?.subscribe(
      topics,
      { qos: 0 },
      (error) => {
        if (error) {
          console.error(
            '❌ MQTT SUBSCRIBE ERROR:',
            error
          );
          return;
        }

        console.log(
          '✅ MQTT SUBSCRIBED:',
          topics
        );

        emit('live');
      }
    );
  });

  // ------------------------------------------------------------
  // MESSAGE EVENT
  // ------------------------------------------------------------

  client.on('message', (topic, payload) => {
    console.log(
      '📨 MQTT MESSAGE:',
      topic,
      payload.toString()
    );

    try {
      const parsed = JSON.parse(
        payload.toString()
      );

      if (topic.endsWith('/state')) {
        state = parsed as HomeState;

        console.log(
          '🏠 HomeState received'
        );
      } else if (topic.endsWith('/intent')) {
        intent = parsed as Intent;

        console.log(
          '🤖 Intent received'
        );
      }

      lastMessage = Date.now();

      emit('live');
    } catch (error) {
      console.error(
        '❌ Invalid MQTT JSON:',
        error
      );
    }
  });

  // ------------------------------------------------------------
  // RECONNECT
  // ------------------------------------------------------------

  client.on('reconnect', () => {
    console.log('🔄 MQTT reconnecting...');
    emit('connecting');
  });

  // ------------------------------------------------------------
  // OFFLINE
  // ------------------------------------------------------------

  client.on('offline', () => {
    console.log('⚠️ MQTT offline');
    emit('offline');
  });

  // ------------------------------------------------------------
  // ERROR
  // ------------------------------------------------------------

  client.on('error', (error) => {
    console.error(
      '❌ MQTT ERROR:',
      error
    );

    emit('offline');
  });

  // ------------------------------------------------------------
  // DATA AGE TIMER
  // ------------------------------------------------------------

  const tick = setInterval(() => {
    emit(
      client?.connected
        ? 'live'
        : 'offline'
    );
  }, 1000);

  // ------------------------------------------------------------
  // CLEANUP
  // ------------------------------------------------------------

  return () => {
    console.log(
      '🔌 Disconnecting MQTT...'
    );

    clearInterval(tick);

    client?.end(true);
  };
}